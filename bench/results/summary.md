# MiniDynamo benchmark results

Workload: 24 conversations x 4 turns (96 requests), 4 workers.

| metric | round_robin | kv_aware | improvement |
|---|---|---|---|
| TTFT p50 (ms) | 67.6 | 52.1 | 23% lower |
| TTFT p90 (ms) | 132.0 | 107.4 | 19% lower |
| TTFT mean (ms) | 76.9 | 59.7 | 22% lower |
| cache hit rate | 0.0% | 91.1% | +91 pts |
| throughput (rps) | 226.3 | 278.4 | 19% faster |
