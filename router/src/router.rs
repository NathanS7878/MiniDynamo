//! Worker pool, KV-cache mirror, and routing policies.
//!
//! The KV-aware policy is the heart of MiniDynamo. It mirrors how NVIDIA
//! Dynamo's smart router works: prompts are tokenized and split into fixed-size
//! blocks; each block gets a prefix-dependent hash (identical prefixes hash
//! identically, exactly like a paged-attention KV cache is keyed); the router
//! tracks which worker holds which blocks and routes each request to the worker
//! that can reuse the longest cached prefix, tie-broken by current load.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Deterministic whitespace tokenizer. The exact tokenizer doesn't matter for
/// the routing algorithm — only that it's deterministic and that shared text
/// prefixes yield shared tokens. (A real deployment would use the model's own
/// tokenizer; that's a drop-in replacement here.)
pub fn tokenize(text: &str) -> Vec<u64> {
    text.split_whitespace()
        .map(|w| {
            let mut h = DefaultHasher::new();
            w.to_lowercase().hash(&mut h);
            h.finish()
        })
        .collect()
}

/// Split tokens into blocks and hash each block *together with its prefix*, so a
/// block's hash depends on every token before it. A cached block is therefore
/// only reusable when the entire preceding context matches — KV-cache semantics.
pub fn block_hashes(tokens: &[u64], block_size: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut acc: u64 = 0xcbf29ce484222325; // FNV offset basis, used as a seed
    for chunk in tokens.chunks(block_size.max(1)) {
        let mut h = DefaultHasher::new();
        acc.hash(&mut h);
        for &t in chunk {
            t.hash(&mut h);
        }
        acc = h.finish();
        out.push(acc);
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteReason {
    /// Chosen because it holds the longest cached prefix.
    Cache,
    /// No useful cache overlap; chosen because it was least loaded.
    Load,
}

#[derive(Debug)]
pub struct Decision {
    pub worker: usize,
    pub reason: RouteReason,
    pub overlap_blocks: usize,
}

pub enum Policy {
    RoundRobin(AtomicUsize),
    KvAware,
}

pub struct WorkerState {
    pub name: String,
    pub base_url: String,
    /// In-flight requests dispatched to this worker (load signal).
    pub active: AtomicUsize,
    /// Router's mirror of the block hashes this worker currently holds.
    pub cache: Mutex<HashSet<u64>>,
}

#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub cache_hit_blocks: AtomicU64,
    pub cache_miss_blocks: AtomicU64,
    pub route_cache_total: AtomicU64,
    pub route_load_total: AtomicU64,
    pub ttft_ms_sum: AtomicU64,
    pub ttft_count: AtomicU64,
}

pub struct Pool {
    pub workers: Vec<WorkerState>,
    pub http: reqwest::Client,
    pub block_size: usize,
    pub policy: Policy,
    pub metrics: Metrics,
    /// Rotating counter used to break ties fairly among equally-good workers,
    /// so that under no cache overlap the policy degrades to round-robin rather
    /// than always piling onto worker 0.
    tie: AtomicUsize,
    /// How many blocks of cache overlap one in-flight request is "worth" in the
    /// KV-aware cost function `score = overlap - load_weight * active_load`.
    /// 0 = pure cache affinity (can overload a hot worker); higher = more load
    /// balancing. Tunable via MD_LOAD_WEIGHT.
    load_weight: f64,
}

/// Longest contiguous run of leading blocks the cache holds.
fn contiguous_overlap(cache: &HashSet<u64>, hashes: &[u64]) -> usize {
    let mut n = 0;
    for h in hashes {
        if cache.contains(h) {
            n += 1;
        } else {
            break;
        }
    }
    n
}

impl Pool {
    pub fn from_env() -> Self {
        let workers = load_workers()
            .into_iter()
            .map(|(name, base_url)| WorkerState {
                name,
                base_url,
                active: AtomicUsize::new(0),
                cache: Mutex::new(HashSet::new()),
            })
            .collect();

        let policy = match std::env::var("MD_POLICY").as_deref() {
            Ok("round_robin") | Ok("rr") => Policy::RoundRobin(AtomicUsize::new(0)),
            _ => Policy::KvAware,
        };

        let block_size = std::env::var("MD_BLOCK_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_BLOCK_SIZE);

        let load_weight = std::env::var("MD_LOAD_WEIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);

        Pool {
            workers,
            http: reqwest::Client::new(),
            block_size,
            policy,
            metrics: Metrics::default(),
            tie: AtomicUsize::new(0),
            load_weight,
        }
    }

    pub fn policy_name(&self) -> &'static str {
        match self.policy {
            Policy::RoundRobin(_) => "round_robin",
            Policy::KvAware => "kv_aware",
        }
    }

    /// Choose a worker for a request described by its block hashes.
    pub fn pick(&self, hashes: &[u64]) -> Decision {
        match &self.policy {
            Policy::RoundRobin(ctr) => {
                let idx = ctr.fetch_add(1, Ordering::Relaxed) % self.workers.len();
                // Round-robin ignores the cache when *choosing* a worker, but we
                // still measure the reuse it happens to land on, so the hit-rate
                // metric reflects real cache behavior rather than the policy's
                // blind spot (otherwise it reads 0% purely by construction).
                let overlap = {
                    let cache = self.workers[idx].cache.lock().unwrap();
                    contiguous_overlap(&cache, hashes)
                };
                let reason = if overlap > 0 { RouteReason::Cache } else { RouteReason::Load };
                Decision { worker: idx, reason, overlap_blocks: overlap }
            }
            Policy::KvAware => {
                let n = self.workers.len();
                // Cost function: score = cache_overlap - load_weight * in_flight.
                // Cache affinity pulls a request toward the worker holding its
                // prefix; the load term pushes it away when that worker is busy,
                // so a hot prefix can't overload one worker. Iterate from a
                // rotating offset so exact ties spread round-robin.
                let start = self.tie.load(Ordering::Relaxed);
                let mut best_idx = start % n;
                let mut best_score = f64::MIN;
                let mut best_overlap = 0usize;
                let mut chose_for_cache = false;
                for k in 0..n {
                    let i = (start + k) % n;
                    let w = &self.workers[i];
                    let overlap = {
                        let cache = w.cache.lock().unwrap();
                        contiguous_overlap(&cache, hashes)
                    };
                    let load = w.active.load(Ordering::Relaxed);
                    let score = overlap as f64 - self.load_weight * load as f64;
                    // Strict `>` keeps the first candidate in rotated order on a tie.
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_overlap = overlap;
                        // "Cache" only if this worker's own cache is why we're here.
                        chose_for_cache = overlap > 0;
                    }
                }
                let reason = if chose_for_cache { RouteReason::Cache } else { RouteReason::Load };
                // Advance the round-robin cursor only when we fell back to load
                // balancing, so consecutive no-overlap requests spread evenly.
                if matches!(reason, RouteReason::Load) {
                    self.tie.fetch_add(1, Ordering::Relaxed);
                }
                Decision { worker: best_idx, reason, overlap_blocks: best_overlap }
            }
        }
    }

    /// Mark a request as dispatched; returns a guard that decrements on drop.
    pub fn begin(&self, idx: usize) {
        self.workers[idx].active.fetch_add(1, Ordering::Relaxed);
    }

    pub fn end(&self, idx: usize) {
        self.workers[idx].active.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record that a worker now holds these blocks. Called at *dispatch* time,
    /// because the worker populates its KV cache the moment `/generate` starts —
    /// so the mirror must not depend on the client reading the whole response.
    pub fn insert_blocks(&self, idx: usize, sent: &[u64]) {
        let mut cache = self.workers[idx].cache.lock().unwrap();
        for h in sent {
            cache.insert(*h);
        }
    }

    /// Apply evictions the worker reported (best-effort; arrives on the `done`
    /// event when the client reads the full stream).
    pub fn apply_evictions(&self, idx: usize, evicted: &[u64]) {
        if evicted.is_empty() {
            return;
        }
        let mut cache = self.workers[idx].cache.lock().unwrap();
        for h in evicted {
            cache.remove(h);
        }
    }

    pub fn record_route(&self, d: &Decision, total_blocks: usize) {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        let hit = d.overlap_blocks.min(total_blocks) as u64;
        let miss = (total_blocks as u64).saturating_sub(hit);
        self.metrics.cache_hit_blocks.fetch_add(hit, Ordering::Relaxed);
        self.metrics.cache_miss_blocks.fetch_add(miss, Ordering::Relaxed);
        match d.reason {
            RouteReason::Cache => self.metrics.route_cache_total.fetch_add(1, Ordering::Relaxed),
            RouteReason::Load => self.metrics.route_load_total.fetch_add(1, Ordering::Relaxed),
        };
    }

    pub fn record_ttft(&self, ms: u64) {
        self.metrics.ttft_ms_sum.fetch_add(ms, Ordering::Relaxed);
        self.metrics.ttft_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn metrics_text(&self) -> String {
        let m = &self.metrics;
        let hit = m.cache_hit_blocks.load(Ordering::Relaxed);
        let miss = m.cache_miss_blocks.load(Ordering::Relaxed);
        let hit_rate = if hit + miss > 0 { hit as f64 / (hit + miss) as f64 } else { 0.0 };
        let ttft_sum = m.ttft_ms_sum.load(Ordering::Relaxed);
        let ttft_count = m.ttft_count.load(Ordering::Relaxed);
        let avg_ttft = if ttft_count > 0 { ttft_sum as f64 / ttft_count as f64 } else { 0.0 };

        let mut out = String::new();
        out.push_str(&format!(
            "# HELP minidynamo_requests_total Total chat completion requests\n\
             # TYPE minidynamo_requests_total counter\n\
             minidynamo_requests_total {}\n",
            m.requests_total.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "# HELP minidynamo_cache_hit_blocks_total KV blocks reused via cache\n\
             minidynamo_cache_hit_blocks_total {hit}\n\
             minidynamo_cache_miss_blocks_total {miss}\n\
             # HELP minidynamo_cache_hit_rate Fraction of prompt blocks served from cache\n\
             # TYPE minidynamo_cache_hit_rate gauge\n\
             minidynamo_cache_hit_rate {hit_rate:.4}\n"
        ));
        out.push_str(&format!(
            "minidynamo_route_reason_total{{reason=\"cache\"}} {}\n\
             minidynamo_route_reason_total{{reason=\"load\"}} {}\n",
            m.route_cache_total.load(Ordering::Relaxed),
            m.route_load_total.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "# HELP minidynamo_ttft_ms_avg Average time to first token (ms)\n\
             # TYPE minidynamo_ttft_ms_avg gauge\n\
             minidynamo_ttft_ms_avg {avg_ttft:.2}\n"
        ));
        for w in &self.workers {
            out.push_str(&format!(
                "minidynamo_worker_active_requests{{worker=\"{}\"}} {}\n",
                w.name,
                w.active.load(Ordering::Relaxed)
            ));
        }
        for w in &self.workers {
            out.push_str(&format!(
                "minidynamo_worker_cache_blocks{{worker=\"{}\"}} {}\n",
                w.name,
                w.cache.lock().unwrap().len()
            ));
        }
        out
    }
}

fn load_workers() -> Vec<(String, String)> {
    match std::env::var("MD_WORKERS") {
        Ok(spec) if !spec.trim().is_empty() => spec
            .split(',')
            .map(|entry| {
                let entry = entry.trim();
                if let Some((name, url)) = entry.split_once('=') {
                    (name.trim().to_string(), url.trim().trim_end_matches('/').to_string())
                } else {
                    (entry.to_string(), entry.trim_end_matches('/').to_string())
                }
            })
            .collect(),
        _ => vec![
            ("worker-0".to_string(), "http://127.0.0.1:9001".to_string()),
            ("worker-1".to_string(), "http://127.0.0.1:9002".to_string()),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefix_hashes_match() {
        let a = tokenize("you are a helpful assistant. what is rust?");
        let b = tokenize("you are a helpful assistant. what is go?");
        let ha = block_hashes(&a, 4);
        let hb = block_hashes(&b, 4);
        // The first block (shared prefix) must hash identically...
        assert_eq!(ha[0], hb[0]);
        // ...and once the text diverges, later blocks must differ.
        assert_ne!(ha.last(), hb.last());
    }

    #[test]
    fn overlap_is_contiguous_prefix() {
        let mut cache = HashSet::new();
        let h = block_hashes(&tokenize("a b c d e f g h"), 2);
        cache.insert(h[0]);
        cache.insert(h[1]);
        cache.insert(h[3]); // hole at index 2 -> overlap stops at 2
        assert_eq!(contiguous_overlap(&cache, &h), 2);
    }
}
