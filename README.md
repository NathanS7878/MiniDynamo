# MiniDynamo

A scaled-down, self-hostable reimplementation of the core ideas in
[**NVIDIA Dynamo**](https://github.com/ai-dynamo/dynamo) — a distributed LLM
inference **router** that routes each request to the worker that can serve it
fastest, primarily by reusing the **KV cache** those workers already hold.

- **Rust** async frontend (`axum`) exposing an **OpenAI-compatible** API
- **Python** workers (`FastAPI`) with a real LRU KV-block cache and a **simulated** model (see below)
- A **KV-cache-aware smart router**: a prefix-dependent block-hash cache mirror + a load-aware cost function (`score = overlap − λ·load`)
- Prometheus `/metrics`, a live dashboard at `/`, and routing-decision headers on every response
- A **benchmark** showing KV-aware routing beats round-robin on TTFT & cache hits
- Runs on **CPU** — no NVIDIA GPU required

> **The workers are simulated.** Each one keeps a real LRU cache of KV blocks,
> but "running the model" is a timed sleep: prefill costs 30 ms per *uncached*
> block and decode 20 ms per token. That makes the benchmark a test of the
> router's decisions under a cost model, not of a real model's latency.
> Replacing the sleep with a real backend is Phase 3.

> Why it exists: I wanted to understand how modern inference-serving systems
> actually work, so I rebuilt a faithful slice of one. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and how each
> piece maps to real Dynamo.

**The layer below:** [nano-infer](https://github.com/NathanS7878/nano-infer) is the engine a real worker would run on the GPU — a from-scratch single-GPU
inference engine with custom CUDA kernels, a paged KV cache and INT8/INT4
quantization. MiniDynamo decides *which* worker runs a request; nano-infer is
what that worker does. (The two are separate projects and are not wired
together yet.)

## Quickstart

Prereqs: Rust (`rustup`), Python 3.11+.

```bash
# 1. Python workers
python -m venv .venv
.venv/Scripts/pip install -r workers/requirements.txt   # (bin/pip on macOS/Linux)

# 2. Launch 2 workers + the router (builds the Rust router on first run)
python scripts/dev.py --workers 2
```

Then, in another terminal, hit the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minidynamo","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Metrics are at `http://localhost:8000/metrics` (Prometheus format).

## Results

On a workload of 24 multi-turn conversations (96 requests, 4 workers) with
shared, growing prefixes — the workload class Dynamo targets — pooled over 5
repeats to cut run-to-run noise:

| metric | round_robin | kv_aware | improvement |
|---|---|---|---|
| TTFT p50 (ms) | 72.8 | 45.4 | **38% lower** |
| TTFT p90 (ms) | 129.5 | 103.4 | **20% lower** |
| cache hit rate (blocks) | 68.7% (1199/1745) | **89.7% (1565/1745)** | +21 pts |
| throughput (rps) | 238.8 | 283.9 | **19% higher** |

![TTFT comparison](bench/results/ttft.png)

Cache hit rate is **block-weighted** (KV blocks reused ÷ total prompt blocks),
not per-request. Round-robin still gets substantial reuse here because 4 system
prompts are shared across the 24 conversations; KV-aware wins by concentrating
each conversation's *growing* prefix on one worker.

Under a **heavier, saturating** workload (40 conversations × 6 turns) KV-aware
converges to round-robin on latency/throughput (≈tied) while keeping a higher
cache-hit rate — the load-aware cost function correctly trades cache affinity
for balance once every worker is busy. Without that load term, an earlier
version *lost* ~2× under load by overloading cache-rich workers.

Reproduce: `python bench/bench.py --workers 4 --conversations 24 --turns 4 --repeat 5`

## Status

- [x] **Phase 1** — Rust OpenAI-compatible frontend, Python workers, round-robin routing, streaming
- [x] **Phase 2** — KV-cache-aware routing (prefix-dependent block-hash mirror, longest-prefix match, load-aware cost function, fair tie-break)
- [x] **Benchmark** — KV-aware vs round-robin harness with TTFT/cache-hit/throughput + chart
- [x] **Observability** — Prometheus metrics, live dashboard, routing-decision headers
- [ ] **Phase 3** — Real model backend (llama.cpp on CPU, or nano-infer on GPU)
- [ ] **Phase 4** — Disaggregated prefill / decode
- [ ] **Phase 5** — Docker/Kubernetes deployment

## Layout

```
router/     Rust router (axum): API, routing policies, per-worker cache mirror, metrics, dashboard
workers/    Python workers (FastAPI): simulated model backend + LRU KV-block cache
bench/      Benchmark harness, committed results and chart
docs/       Architecture
scripts/    Local dev launcher
```
