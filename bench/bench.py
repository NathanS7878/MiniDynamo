"""
MiniDynamo benchmark: KV-aware routing vs round-robin.

Replays a realistic workload with strong prefix locality (multi-turn
conversations that share a growing prefix, plus one-off requests) against the
router under each policy, and reports time-to-first-token (TTFT), cache-hit
rate, and throughput.

For each policy it boots a fresh worker pool + router, warms up, replays the
*same* workload concurrently (one thread per conversation, turns sequential
within a conversation), then tears everything down.

Usage:
    python bench/bench.py                 # 4 workers, default workload
    python bench/bench.py --workers 6 --conversations 32 --turns 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
ROUTER_BIN = ROOT / "router" / "target" / "debug" / ("router.exe" if IS_WIN else "router")
RESULTS = ROOT / "bench" / "results"

SYSTEM_PROMPTS = [
    "You are a senior distributed-systems engineer. Answer precisely, note tradeoffs, and prefer concrete numbers over hand-waving in every single response you give.",
    "You are a terse Rust assistant. Output only idiomatic Rust with zero prose, no comments, and no explanation whatsoever under any circumstances at all.",
    "You are a patient math tutor for first-year calculus. Explain each step, define notation, and never skip algebra even when it seems obvious to an expert.",
    "You are a cautious SRE reviewing production changes. Call out blast radius, rollback plans, and monitoring gaps before approving anything that touches serving.",
]
USER_TURNS = [
    "walk me through the first idea",
    "now compare it to the alternative",
    "what breaks under high load",
    "summarize the tradeoffs as bullets",
    "and how would you monitor it",
]


def build_workload(n_conversations: int, turns: int) -> list[list[dict]]:
    """Return conversations; each is a list of OpenAI-style request bodies where
    turn t includes the full history so the shared prefix grows every turn."""
    conversations = []
    for c in range(n_conversations):
        system = SYSTEM_PROMPTS[c % len(SYSTEM_PROMPTS)]
        history = [{"role": "system", "content": system}]
        reqs = []
        for t in range(turns):
            history = history + [{"role": "user", "content": USER_TURNS[t % len(USER_TURNS)]}]
            reqs.append({"messages": list(history), "max_tokens": 8, "stream": True})
            # Pretend the assistant replied, so the next turn's prefix includes it.
            history = history + [{"role": "assistant", "content": "ok noted, continuing the thread"}]
        conversations.append(reqs)
    return conversations


def stream_ttft(url: str, body: dict) -> float:
    """Send one request; return time-to-first-content-token in seconds."""
    t0 = time.perf_counter()
    with requests.post(url, json=body, stream=True, timeout=60) as r:
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue
            payload = s[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def run_conversation(url: str, reqs: list[dict], ttfts: list[float]) -> None:
    for body in reqs:
        ttfts.append(stream_ttft(url, body))


def start_stack(policy: str, workers: int, base_port: int, router_port: int) -> list[subprocess.Popen]:
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    procs: list[subprocess.Popen] = []
    specs = []
    for i in range(workers):
        port = base_port + i
        procs.append(subprocess.Popen(
            [py, str(ROOT / "workers" / "worker.py"), "--port", str(port), "--name", f"worker-{i}"],
            cwd=str(ROOT / "workers"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        specs.append(f"worker-{i}=http://127.0.0.1:{port}")

    env = dict(os.environ)
    env["MD_WORKERS"] = ",".join(specs)
    env["MD_LISTEN"] = f"127.0.0.1:{router_port}"
    env["MD_POLICY"] = policy
    env["RUST_LOG"] = "warn"
    procs.append(subprocess.Popen([str(ROUTER_BIN)], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return procs


def stop_stack(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()


def wait_healthy(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.3)
    raise RuntimeError(f"router at {url} never became healthy")


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def run_policy(policy: str, workload: list[list[dict]], workers: int, router_port: int) -> dict:
    procs = start_stack(policy, workers, base_port=9101, router_port=router_port)
    try:
        wait_healthy(f"http://127.0.0.1:{router_port}/health")
        url = f"http://127.0.0.1:{router_port}/v1/chat/completions"

        # Warmup (not measured): prime caches/JIT.
        stream_ttft(url, {"messages": [{"role": "user", "content": "warmup"}], "max_tokens": 1, "stream": True})

        ttfts: list[float] = []
        buckets = [[] for _ in workload]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(workload)) as ex:
            futs = [ex.submit(run_conversation, url, conv, buckets[i]) for i, conv in enumerate(workload)]
            for f in futs:
                f.result()
        wall = time.perf_counter() - t0
        for b in buckets:
            ttfts.extend(b)

        metrics = requests.get(f"http://127.0.0.1:{router_port}/metrics", timeout=5).text
        hit_rate = 0.0
        for ln in metrics.splitlines():
            if ln.startswith("minidynamo_cache_hit_rate"):
                hit_rate = float(ln.split()[-1])

        return {
            "policy": policy,
            "requests": len(ttfts),
            "ttft_p50_ms": pctl(ttfts, 50) * 1000,
            "ttft_p90_ms": pctl(ttfts, 90) * 1000,
            "ttft_mean_ms": statistics.mean(ttfts) * 1000 if ttfts else 0.0,
            "cache_hit_rate": hit_rate,
            "wall_sec": wall,
            "throughput_rps": len(ttfts) / wall if wall else 0.0,
            "_ttfts_ms": [x * 1000 for x in ttfts],
        }
    finally:
        stop_stack(procs)
        time.sleep(1.0)  # let ports free up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conversations", type=int, default=24)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--router-port", type=int, default=8100)
    args = ap.parse_args()

    if not ROUTER_BIN.exists():
        sys.exit(f"router binary not found at {ROUTER_BIN}; run `cargo build` in router/ first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    workload = build_workload(args.conversations, args.turns)
    total = args.conversations * args.turns
    print(f"workload: {args.conversations} conversations x {args.turns} turns = {total} requests, {args.workers} workers\n")

    results = []
    for policy in ("round_robin", "kv_aware"):
        print(f"running policy = {policy} ...")
        res = run_policy(policy, workload, args.workers, args.router_port)
        results.append(res)
        print(f"  p50 TTFT = {res['ttft_p50_ms']:.1f} ms | p90 = {res['ttft_p90_ms']:.1f} ms | "
              f"cache hit = {res['cache_hit_rate']*100:.1f}% | throughput = {res['throughput_rps']:.1f} rps\n")

    # Persist + summarize
    with open(RESULTS / "results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in results], f, indent=2)

    rr, kv = results[0], results[1]
    def improvement(a, b):  # percent reduction from a -> b
        return (a - b) / a * 100 if a else 0.0
    summary = (
        "| metric | round_robin | kv_aware | improvement |\n"
        "|---|---|---|---|\n"
        f"| TTFT p50 (ms) | {rr['ttft_p50_ms']:.1f} | {kv['ttft_p50_ms']:.1f} | {improvement(rr['ttft_p50_ms'], kv['ttft_p50_ms']):.0f}% lower |\n"
        f"| TTFT p90 (ms) | {rr['ttft_p90_ms']:.1f} | {kv['ttft_p90_ms']:.1f} | {improvement(rr['ttft_p90_ms'], kv['ttft_p90_ms']):.0f}% lower |\n"
        f"| TTFT mean (ms) | {rr['ttft_mean_ms']:.1f} | {kv['ttft_mean_ms']:.1f} | {improvement(rr['ttft_mean_ms'], kv['ttft_mean_ms']):.0f}% lower |\n"
        f"| cache hit rate | {rr['cache_hit_rate']*100:.1f}% | {kv['cache_hit_rate']*100:.1f}% | +{(kv['cache_hit_rate']-rr['cache_hit_rate'])*100:.0f} pts |\n"
        f"| throughput (rps) | {rr['throughput_rps']:.1f} | {kv['throughput_rps']:.1f} | {improvement(rr['wall_sec'], kv['wall_sec']):.0f}% faster |\n"
    )
    print(summary)
    with open(RESULTS / "summary.md", "w") as f:
        f.write("# MiniDynamo benchmark results\n\n")
        f.write(f"Workload: {args.conversations} conversations x {args.turns} turns "
                f"({total} requests), {args.workers} workers.\n\n")
        f.write(summary)

    try:
        make_chart(results)
        print(f"chart written to {RESULTS / 'ttft.png'}")
    except Exception as e:  # matplotlib optional
        print(f"(skipping chart: {e})")


def make_chart(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["policy"] for r in results]
    p50 = [r["ttft_p50_ms"] for r in results]
    p90 = [r["ttft_p90_ms"] for r in results]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], p50, w, label="TTFT p50", color="#76b900")  # NVIDIA green
    ax.bar([i + w / 2 for i in x], p90, w, label="TTFT p90", color="#004831")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("time to first token (ms)")
    ax.set_title("MiniDynamo: KV-aware routing vs round-robin")
    ax.legend()
    for i, v in enumerate(p50):
        ax.text(i - w / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(p90):
        ax.text(i + w / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "ttft.png", dpi=130)


if __name__ == "__main__":
    main()
