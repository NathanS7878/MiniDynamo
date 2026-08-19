//! MiniDynamo router — HTTP frontend.
//!
//! OpenAI-compatible frontend that fans requests out to a pool of Python
//! workers using a pluggable routing policy (round-robin or KV-cache-aware).
//! Routing logic lives in `router.rs`; see docs/ARCHITECTURE.md.

mod router;

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::router::Pool;

// ---------- OpenAI-compatible wire types ----------

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default = "default_model")]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

fn default_model() -> String {
    "minidynamo".to_string()
}
fn default_max_tokens() -> u32 {
    32
}

// ---------- worker protocol ----------

#[derive(Serialize)]
struct GenerateRequest {
    request_id: String,
    prompt: String,
    max_tokens: u32,
    block_hashes: Vec<u64>,
    block_size: usize,
}

#[derive(Deserialize)]
struct WorkerEvent {
    event: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    evicted: Option<Vec<u64>>,
}

// ---------- helpers ----------

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Flatten chat messages into a single prompt. Deterministic ordering matters
/// for KV-cache reuse: identical conversation prefixes render identically.
fn render_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str(&m.role);
        s.push_str(": ");
        s.push_str(&m.content);
        s.push('\n');
    }
    s.push_str("assistant: ");
    s
}

fn openai_chunk(id: &str, created: u64, model: &str, delta_content: Option<&str>, finish: bool) -> String {
    let delta = match delta_content {
        Some(c) => serde_json::json!({ "content": c }),
        None => serde_json::json!({ "role": "assistant" }),
    };
    serde_json::json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{
            "index": 0, "delta": delta,
            "finish_reason": if finish { serde_json::Value::String("stop".into()) } else { serde_json::Value::Null }
        }]
    })
    .to_string()
}

// ---------- handlers ----------

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

async fn metrics(State(pool): State<Arc<Pool>>) -> String {
    pool.metrics_text()
}

async fn chat_completions(State(pool): State<Arc<Pool>>, Json(req): Json<ChatRequest>) -> Response {
    let prompt = render_prompt(&req.messages);
    let tokens = router::tokenize(&prompt);
    let hashes = router::block_hashes(&tokens, pool.block_size);

    let decision = pool.pick(&hashes);
    pool.record_route(&decision, hashes.len());
    pool.begin(decision.worker);
    // The worker caches these blocks as soon as it receives the request, so
    // update the mirror now — not when the response stream finishes (the client
    // may disconnect after the first token).
    pool.insert_blocks(decision.worker, &hashes);

    let worker = &pool.workers[decision.worker];
    let request_id = format!("chatcmpl-{}", now_unix());
    let model = req.model.clone();

    let gen = GenerateRequest {
        request_id: request_id.clone(),
        prompt,
        max_tokens: req.max_tokens,
        block_hashes: hashes.clone(),
        block_size: pool.block_size,
    };

    tracing::info!(
        worker = %worker.name,
        reason = ?decision.reason,
        overlap = decision.overlap_blocks,
        blocks = hashes.len(),
        "routed {request_id}"
    );

    let resp = match pool
        .http
        .post(format!("{}/generate", worker.base_url))
        .json(&gen)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            pool.end(decision.worker);
            tracing::error!("worker {} request failed: {e}", worker.name);
            return (axum::http::StatusCode::BAD_GATEWAY, format!("worker unavailable: {e}")).into_response();
        }
    };

    if req.stream {
        stream_response(pool.clone(), resp, decision.worker, hashes, request_id, model).await
    } else {
        aggregate_response(pool.clone(), resp, decision.worker, hashes, request_id, model).await
    }
}

/// Decrements a worker's active-request count when dropped (i.e. when the
/// response stream finishes or the client disconnects).
struct ActiveGuard {
    pool: Arc<Pool>,
    idx: usize,
}
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.pool.end(self.idx);
    }
}

async fn stream_response(
    pool: Arc<Pool>,
    resp: reqwest::Response,
    worker_idx: usize,
    sent_hashes: Vec<u64>,
    request_id: String,
    model: String,
) -> Response {
    let created = now_unix();
    let start = Instant::now();
    let guard = ActiveGuard { pool: pool.clone(), idx: worker_idx };

    let sse = async_stream::stream! {
        let _guard = guard; // held until the stream ends
        yield Ok::<Event, Infallible>(Event::default().data(openai_chunk(&request_id, created, &model, None, false)));

        let mut resp = resp;
        let mut buf: Vec<u8> = Vec::new();
        let mut ttft_recorded = false;
        let mut evicted: Vec<u64> = Vec::new();

        loop {
            match resp.chunk().await {
                Ok(Some(bytes)) => {
                    buf.extend_from_slice(&bytes);
                    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = buf.drain(..=pos).collect();
                        let line = &line[..line.len() - 1];
                        if line.is_empty() { continue; }
                        if let Ok(ev) = serde_json::from_slice::<WorkerEvent>(line) {
                            match ev.event.as_str() {
                                "token" => {
                                    if !ttft_recorded {
                                        pool.record_ttft(start.elapsed().as_millis() as u64);
                                        ttft_recorded = true;
                                    }
                                    if let Some(t) = ev.text {
                                        yield Ok(Event::default().data(openai_chunk(&request_id, created, &model, Some(&t), false)));
                                    }
                                }
                                "done" => {
                                    if let Some(ev_list) = ev.evicted { evicted = ev_list; }
                                    yield Ok(Event::default().data(openai_chunk(&request_id, created, &model, None, true)));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => { tracing::error!("stream error: {e}"); break; }
            }
        }
        let _ = &sent_hashes; // already inserted at dispatch
        pool.apply_evictions(worker_idx, &evicted);
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(sse).into_response()
}

async fn aggregate_response(
    pool: Arc<Pool>,
    resp: reqwest::Response,
    worker_idx: usize,
    sent_hashes: Vec<u64>,
    request_id: String,
    model: String,
) -> Response {
    let created = now_unix();
    let start = Instant::now();
    let _guard = ActiveGuard { pool: pool.clone(), idx: worker_idx };
    let mut resp = resp;
    let mut buf: Vec<u8> = Vec::new();
    let mut content = String::new();
    let mut ttft_recorded = false;
    let mut evicted: Vec<u64> = Vec::new();

    loop {
        match resp.chunk().await {
            Ok(Some(bytes)) => {
                buf.extend_from_slice(&bytes);
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = &line[..line.len() - 1];
                    if line.is_empty() { continue; }
                    if let Ok(ev) = serde_json::from_slice::<WorkerEvent>(line) {
                        match ev.event.as_str() {
                            "token" => {
                                if !ttft_recorded {
                                    pool.record_ttft(start.elapsed().as_millis() as u64);
                                    ttft_recorded = true;
                                }
                                if let Some(t) = ev.text { content.push_str(&t); }
                            }
                            "done" => { if let Some(ev_list) = ev.evicted { evicted = ev_list; } }
                            _ => {}
                        }
                    }
                }
            }
            Ok(None) => break,
            Err(e) => { tracing::error!("aggregate stream error: {e}"); break; }
        }
    }
    let _ = &sent_hashes; // already inserted at dispatch
    pool.apply_evictions(worker_idx, &evicted);

    Json(serde_json::json!({
        "id": request_id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": content }, "finish_reason": "stop" }]
    }))
    .into_response()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let pool = Arc::new(Pool::from_env());
    tracing::info!(
        "MiniDynamo router: policy={}, block_size={}, {} worker(s)",
        pool.policy_name(),
        pool.block_size,
        pool.workers.len()
    );
    for w in &pool.workers {
        tracing::info!("  {} -> {}", w.name, w.base_url);
    }
    let _ = pool.metrics.requests_total.load(Ordering::Relaxed); // touch metrics

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(pool);

    let addr = std::env::var("MD_LISTEN").unwrap_or_else(|_| "0.0.0.0:8000".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, app).await.expect("serve");
}
