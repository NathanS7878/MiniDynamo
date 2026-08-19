# Resume content for the NVIDIA Dynamo application

Copy-paste-ready. Only claims features that are actually built and tested.
Update the bracketed GitHub URL once the repo is pushed.

## Reordering advice
For this role, put **Projects above Work Experience** — a distributed-systems
project matters more to Dynamo reviewers than retail history. Add your GitHub
URL to the header next to LinkedIn.

---

## Professional summary (replaces the current one)

> Computer Science student at CU Denver (4.0 GPA, Dean's List) focused on
> systems programming and AI infrastructure. Works across Rust, Python, C, and
> C++. Recently built MiniDynamo, a distributed LLM-inference router modeled on
> NVIDIA Dynamo — implementing KV-cache-aware request routing that cut p50
> time-to-first-token 23% versus round-robin. Comfortable with async
> concurrency, REST APIs, and performance benchmarking.

## Skills (regrouped — replaces the current mixed list)

- **Languages:** Rust, Python, C, C++, Java, x86 Assembly
- **Systems & AI infrastructure:** distributed systems, LLM inference serving,
  KV-cache management, async/concurrency (Tokio), OpenAI-compatible REST APIs,
  performance benchmarking
- **Tools:** Git/GitHub, Cargo, FastAPI, Linux, Excel
  *(Docker/Kubernetes once Phase 5 lands)*

## Flagship project (new — top of Projects section)

**MiniDynamo — KV-cache-aware LLM inference router**  ·  Rust, Python  ·  [github.com/NathanS7878/MiniDynamo]

- Built a distributed LLM-inference serving system modeled on **NVIDIA Dynamo**:
  an async **Rust** (axum/Tokio) **OpenAI-compatible** frontend that routes
  requests across a pool of **Python** (FastAPI) model workers.
- Implemented **KV-cache-aware routing** — tokenizes each prompt, hashes it into
  prefix-dependent blocks, and routes to the worker holding the longest cached
  prefix (least-loaded worker on ties), mirroring Dynamo's smart router.
- Benchmarked against round-robin on multi-turn workloads: **91% cache-hit rate**
  and **23% lower p50 time-to-first-token**; exposed Prometheus metrics for
  cache hits, TTFT, and per-worker load.

### Bullets to ADD once the matching phase is built
- *(Phase 4)* Added **disaggregated serving**: separate prefill and decode
  workers with KV-state handoff, mirroring Dynamo's prefill/decode split.
- *(Phase 5)* Containerized the stack with **Docker Compose** and **Kubernetes**
  manifests for multi-worker deployment.

## Keep, but compress (secondary project)

**Community Building Engineering Project** — 4-person team; designed a
community-focused game from concept to 3D-printed prototype (PrusaSlicer);
presented at the Engineering Expo. *(One line is enough now.)*

## Work experience — keep, trim to 1–2 lines each
Retail roles show reliability and communication; they just shouldn't dominate.
Cut each to a single strong bullet.
