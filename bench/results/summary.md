# MiniDynamo benchmark results

Workload: 24 conversations x 4 turns (96 requests), 4 workers, 5 repeat(s) pooled. KV-aware load_weight = 1.0.

| metric | round_robin | kv_aware | improvement |
|---|---|---|---|
| TTFT p50 (ms) | 72.8 | 45.4 | 38% lower |
| TTFT p90 (ms) | 129.5 | 103.4 | 20% lower |
| TTFT mean (ms) | 76.9 | 55.0 | 28% lower |
| cache hit rate (blocks) | 68.7% (1199/1745) | 89.7% (1565/1745) | +21 pts |
| throughput (rps) | 238.8 | 283.9 | 19% higher |
| wall time (s) | 0.40 | 0.34 | 15% faster |

Cache hit rate is **block-weighted**: numerator = KV blocks reused, denominator = total prompt blocks across all requests (not a per-request average). TTFT percentiles are pooled across repeats to reduce run-to-run noise.
