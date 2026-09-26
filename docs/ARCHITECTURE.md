# MiniDynamo — Architecture

MiniDynamo is a scaled-down, self-hostable reimplementation of the core ideas in
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo): an inference **router**
that sits in front of a pool of model **workers** and sends each request to the
worker that can serve it fastest — primarily by reusing the **KV cache** that
worker already holds.

The workers are **simulated**: they model prefill and decode cost with timed
sleeps instead of running a real model (see [Workers](#workers)). The part being
studied is the *routing layer*, which doesn't depend on what the worker computes.
The real GPU worker this router is designed to sit on top of is
[nano-infer](https://github.com/NathanS7878/nano-infer).

## How this maps to real Dynamo

| Real Dynamo | MiniDynamo | Status |
|---|---|---|
| OpenAI-compatible frontend | Rust `axum` frontend, `POST /v1/chat/completions` (streaming + non-streaming) | Built |
| KV-aware smart router | Per-worker mirror of cached block hashes; longest-contiguous-prefix match; `score = overlap − λ·load` | Built |
| KV cache events | Workers report evicted block hashes on each request's final `done` event; the router applies them to its mirror | Built (piggybacked on the response stream, not a separate event bus) |
| Metrics | Prometheus-format `/metrics` | Built |
| Python workers | `FastAPI` workers with a simulated model backend | Built (simulated) |
| Radix tree of block hashes | Not used — a flat hash set per worker is enough at this scale | Not built |
| Disaggregated prefill/decode | — | Not built |
| Kubernetes serving platform | — | Not built |

## Components

```
                         ┌──────────────────────────────────────┐
      OpenAI client ───► │  Rust frontend + router (axum/Tokio)  │
  POST /v1/chat/...      │   • tokenize prompt                   │
                         │   • split into prefix-dependent       │
                         │     block hashes                      │
                         │   • score each worker (cache − load)  │
                         │   • stream response back              │
                         │  /  (dashboard)  /metrics  /health    │
                         └───────┬───────────────┬───────────────┘
                                 │               │
                       ┌─────────▼───┐     ┌─────▼─────────┐
                       │ Worker 0     │     │ Worker 1      │   ...
                       │ (Python)     │     │ (Python)      │
                       │ LRU KV cache │     │ LRU KV cache  │
                       │ simulated    │     │ simulated     │
                       │ prefill/decode│    │ prefill/decode│
                       └──────────────┘     └───────────────┘
```

## Routing algorithm (the core idea)

1. The frontend receives a chat request and **tokenizes** the prompt. The
   tokenizer is a deterministic whitespace tokenizer — the routing logic only
   needs identical text to produce identical tokens, so a real BPE tokenizer is
   a drop-in replacement.
2. Tokens are chunked into fixed-size **blocks** (default 16). Each block is
   hashed together with the hash of the block before it, so a block's hash
   depends on its whole prefix — the same way paged KV caches are keyed.
3. For each worker the router keeps a **mirror**: a hash set of the block hashes
   it believes that worker holds.
4. For an incoming request it computes, per worker, the **longest contiguous
   run of leading blocks** in that mirror — how many blocks the worker could skip
   recomputing. This is the *cache overlap*.
5. It scores each worker as `overlap − λ · in_flight_requests` and picks the
   highest (`λ` = `MD_LOAD_WEIGHT`, default 1.0). The load term stops a hot
   prefix from piling every request onto one worker. Exact ties are broken by a
   rotating cursor, so with no overlap the policy degrades to round-robin.
6. After the request finishes, the router adds the request's blocks to that
   worker's mirror and removes any blocks the worker reports it evicted.

Baseline for comparison: plain **round-robin**, which ignores the cache when
choosing a worker. The router still *measures* the cache reuse round-robin
happens to land on, so the two policies' hit rates are directly comparable.

### Why the load term exists

An earlier version picked the highest overlap and only used load to break ties.
Under a saturating workload it lost to round-robin by about 2×: cache-rich
workers accumulated a queue while others sat idle. Scoring cache and load
together fixed that — under saturation KV-aware now ties round-robin on latency
and throughput while keeping a higher cache-hit rate.

### A consistency bug the metrics caught

Originally the router added a request's blocks to its mirror only when the
response stream finished. But a worker fills its KV cache the moment generation
starts — so if a client disconnected early, the worker held blocks the router
didn't know about, and the mirror under-counted cache that really existed. The
Prometheus cache metrics exposed the gap. The fix records blocks at **dispatch** time,
independent of whether the client reads the whole response.

One known gap remains: evictions still arrive on the `done` event, so an early
disconnect can leave evicted blocks in the mirror until the next eviction report.
That errs toward over-estimating cache, which the load term partly absorbs.

## Workers

Each worker is a `FastAPI` app with a real LRU cache of block hashes
(`MD_KV_CAPACITY_BLOCKS`, default 512) and a **simulated** model:

- **Prefill** sleeps `MD_PREFILL_SEC_PER_BLOCK` (default 30 ms) per *uncached*
  block — cache hits are free, which is the effect routing is trying to exploit.
- **Decode** sleeps `MD_DECODE_SEC_PER_TOKEN` (default 20 ms) per token and
  streams deterministic filler text.

So the benchmark measures the router's decisions under a cost model, not a real
model's latency. Swapping the sleep for a real backend (llama.cpp on CPU, or
nano-infer on GPU) is the next phase.

## Metrics

`/metrics` (Prometheus text format):
- `minidynamo_cache_hit_blocks_total`, `minidynamo_cache_miss_blocks_total`,
  `minidynamo_cache_hit_rate` — block-weighted
- `minidynamo_ttft_ms_avg`
- `minidynamo_requests_total`, `minidynamo_route_reason_total{reason="cache|load"}`
- `minidynamo_worker_active_requests{worker="..."}`
- `minidynamo_worker_cache_blocks{worker="..."}` — size of each mirror

`/` serves a small live dashboard. Every response also carries `X-MD-Worker`,
`X-MD-Reason`, `X-MD-Overlap` and `X-MD-Blocks` headers describing the routing
decision.

## Not built (yet)

- **Real model backend** — the workers are simulated.
- **Disaggregated prefill/decode** — Dynamo runs prefill and decode on separate
  workers and hands KV state between them. Not implemented here.
- **Radix tree / global index** — real Dynamo indexes blocks in a radix tree
  across the fleet; a per-worker hash set is sufficient at 4 workers.
- **Container / Kubernetes deployment.**
