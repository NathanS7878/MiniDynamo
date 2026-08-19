"""
MiniDynamo worker (Phase 2: mock backend + LRU KV cache).

A worker owns one model instance and its KV cache. In Phase 1/2 the "model" is a
mock that streams deterministic tokens with realistic-ish timing so the whole
pipeline can be exercised end to end. Phase 3 swaps the mock for a real tiny
model behind the same interface.

The worker owns an LRU cache of KV *blocks* (identified by the prefix-dependent
block hashes the router computes and sends along). Prefill time drops for blocks
already resident — so cache-aware routing produces measurably lower TTFT. The
worker reports evicted block hashes back so the router's mirror stays honest.

Run:
    python worker.py --port 9001 --name worker-0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import OrderedDict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

WORKER_NAME = os.environ.get("MD_WORKER_NAME", "worker")
# Simulated timing so TTFT/throughput differences are observable.
PREFILL_SEC_PER_BLOCK = float(os.environ.get("MD_PREFILL_SEC_PER_BLOCK", "0.03"))
DECODE_SEC_PER_TOKEN = float(os.environ.get("MD_DECODE_SEC_PER_TOKEN", "0.02"))
# KV cache capacity in blocks. Small enough that eviction happens under load,
# which is what makes routing decisions matter.
KV_CAPACITY_BLOCKS = int(os.environ.get("MD_KV_CAPACITY_BLOCKS", "512"))

app = FastAPI(title=f"MiniDynamo Worker ({WORKER_NAME})")

_active_requests = 0
# LRU of resident block hashes: key = block hash, value unused. Insertion order
# is LRU order; move_to_end marks most-recently-used.
_kv: "OrderedDict[int, None]" = OrderedDict()

_LOREM = (
    "the quick brown fox jumps over the lazy dog while distributed systems "
    "route requests to cache locality aware workers minimizing time to first "
    "token across a pool of gpus in a disaggregated serving topology"
).split()


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = 32
    block_hashes: list[int] = []
    block_size: int = 16


def _touch_cache(block_hashes: list[int]) -> tuple[int, list[int]]:
    """Insert/refresh blocks in the LRU. Returns (leading cache-hit blocks,
    evicted block hashes)."""
    # Contiguous leading prefix already resident (real KV reuse is prefix-only).
    hit = 0
    for h in block_hashes:
        if h in _kv:
            hit += 1
        else:
            break
    # Mark all requested blocks most-recently-used (inserting misses).
    for h in block_hashes:
        if h in _kv:
            _kv.move_to_end(h)
        else:
            _kv[h] = None
    # Evict LRU beyond capacity.
    evicted: list[int] = []
    while len(_kv) > KV_CAPACITY_BLOCKS:
        k, _ = _kv.popitem(last=False)
        evicted.append(k)
    return hit, evicted


@app.get("/status")
async def status() -> dict:
    return {
        "name": WORKER_NAME,
        "active_requests": _active_requests,
        "backend": "mock",
        "kv_blocks_resident": len(_kv),
        "kv_capacity_blocks": KV_CAPACITY_BLOCKS,
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


async def _generate_stream(req: GenerateRequest):
    global _active_requests
    _active_requests += 1
    try:
        hit_blocks, evicted = _touch_cache(req.block_hashes)
        uncached = max(0, len(req.block_hashes) - hit_blocks)

        # Prefill cost is proportional to *uncached* blocks — cache hits are free.
        prefill_time = uncached * PREFILL_SEC_PER_BLOCK
        await asyncio.sleep(prefill_time)
        yield json.dumps({
            "event": "prefill_done",
            "worker": WORKER_NAME,
            "prefill_sec": prefill_time,
            "cached_hit_blocks": hit_blocks,
            "total_blocks": len(req.block_hashes),
        }) + "\n"

        for i in range(req.max_tokens):
            await asyncio.sleep(DECODE_SEC_PER_TOKEN)
            word = _LOREM[i % len(_LOREM)]
            token = word if i == 0 else " " + word
            yield json.dumps({"event": "token", "text": token}) + "\n"

        yield json.dumps({
            "event": "done",
            "cached_hit_blocks": hit_blocks,
            "evicted": evicted,
        }) + "\n"
    finally:
        _active_requests -= 1


@app.post("/generate")
async def generate(req: GenerateRequest) -> StreamingResponse:
    return StreamingResponse(_generate_stream(req), media_type="application/x-ndjson")


def _main() -> None:
    global WORKER_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--name", default=WORKER_NAME)
    args = parser.parse_args()
    WORKER_NAME = args.name
    os.environ["MD_WORKER_NAME"] = args.name

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
