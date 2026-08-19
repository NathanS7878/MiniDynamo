# MiniDynamo — Architecture

MiniDynamo is a scaled-down, self-hostable reimplementation of the core ideas in
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo): a distributed inference
**router** that sits in front of a pool of model **workers** and routes each
request to the worker that can serve it fastest — primarily by reusing the
**KV cache** that workers already hold.

It runs on CPU with a tiny model (no NVIDIA GPU required); the interesting part
is the *serving infrastructure*, which is hardware-independent.

## How this maps to real Dynamo

| Real Dynamo | MiniDynamo |
|---|---|
| OpenAI-compatible frontend | Rust `axum` frontend, `POST /v1/chat/completions` |
| Rust Runtime Core Library | Rust router: routing policies, KV radix tree, metrics |
| Python SDK workers | Python `FastAPI` workers wrapping a model backend |
| KV-aware smart router | Global radix tree of token-block hashes → workers; longest-prefix match with load tie-break |
| Disaggregated serving (prefill/decode split) | Separate `prefill` and `decode` worker roles; KV state handed off between them |
| KV cache events over NATS | Workers report cached/evicted block hashes to the router over HTTP |
| Kubernetes serving platform | `docker-compose` + `k8s/` manifests |

## Components

```
                         ┌──────────────────────────────────────┐
      OpenAI client ───► │  Rust Frontend  (axum)                │
  POST /v1/chat/...      │  ┌────────────────────────────────┐  │
                         │  │  Router                         │  │
                         │  │   • tokenize prompt             │  │
                         │  │   • split into KV blocks        │  │
                         │  │   • radix-tree prefix match     │  │
                         │  │   • pick worker (cache + load)  │  │
                         │  └────────────────────────────────┘  │
                         │  /metrics  /health                    │
                         └───────┬───────────────┬───────────────┘
                                 │               │
                       ┌─────────▼───┐     ┌─────▼─────────┐
                       │ Worker 0     │     │ Worker 1      │   ...
                       │ (Python)     │     │ (Python)      │
                       │ model + KV   │     │ model + KV    │
                       └──────────────┘     └───────────────┘
```

## Routing algorithm (the core idea)

1. The frontend receives a chat request and **tokenizes** the full prompt.
2. Tokens are chunked into fixed-size **blocks** (default 16 tokens). Each block
   is hashed together with its prefix, so identical prefixes hash identically —
   exactly how paged-attention KV caches are keyed.
3. The router keeps a **radix tree**: each node is a block hash, and each node
   records the set of workers currently holding that block in their KV cache.
4. For an incoming request, the router walks the tree along the request's block
   hashes and finds, per worker, the **longest cached prefix** (how many blocks
   it could skip recomputing). This is the *cache overlap score*.
5. It picks the worker maximizing `overlap_score` and, on ties or low overlap,
   the **least-loaded** worker (fewest active requests). This is the
   cost function Dynamo's KV router minimizes.
6. Workers report which blocks they cached (and evicted) so the tree stays
   consistent with reality.

Baseline for comparison: plain **round-robin** routing, which ignores cache
locality. The benchmark shows KV-aware routing wins on **time-to-first-token
(TTFT)** and **cache-hit rate** for workloads with shared prefixes (multi-turn
chats, shared system prompts, few-shot templates) — the workloads Dynamo targets.

## Disaggregated serving (Phase 4)

LLM inference has two phases with different resource profiles:
- **Prefill**: compute-bound, processes the whole prompt once, produces the
  first token + KV cache.
- **Decode**: memory-bandwidth-bound, generates tokens one at a time.

Dynamo can run these on *separate* workers so each is scheduled independently.
MiniDynamo mirrors this: a `prefill` worker computes the KV state and first
token, hands the KV state to a `decode` worker, which streams the rest.

## Metrics

The router exposes `/metrics` (Prometheus-style):
- `cache_hit_blocks_total`, `cache_miss_blocks_total` → cache-hit rate
- `ttft_seconds` histogram
- `worker_active_requests{worker="..."}`
- `requests_total{route_reason="cache|load"}`
