#!/usr/bin/env python3
"""Concurrency sweep (C1..CN) against an OpenAI-compatible endpoint.

Purpose: measure what co-tenanting a video render on the same node costs DS4.
Run the same sweep against a clean lane and a loaded lane at the same moment,
so the comparison is harness-for-harness rather than against a number measured
some other way on some other day.

Streams every request so TTFT is the real time-to-first-token, and decode rate
is measured first-token -> last-token. Total wall time includes prefill and
would understate decode; that distinction is the whole point of the exercise.

Usage: bench_conc.py <host:port> <model> [label] [levels] [max_tokens]
Stdlib only, so it runs anywhere without a venv.
"""
import json
import sys
import threading
import time
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-flash"
LABEL = sys.argv[3] if len(sys.argv) > 3 else HOST
LEVELS = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else
                           ["1", "2", "3", "4", "5"])]
MAX_TOK = int(sys.argv[5]) if len(sys.argv) > 5 else 700

# Distinct prompts per stream: identical prompts would share a prefix-cache hit
# and flatter the result, which is not what we are trying to find out.
# COUNT PROMPTS, deliberately. On a speculative-decoding model the throughput
# you measure is a function of how PREDICTABLE the output is, not just the
# hardware: measured on this box, same model, same minute, dense prose gets
# 35.9 tok/s at 2.45 accepted tokens per pass while counting gets 91.6 at 5.94.
# A 2.6x spread from the prompt alone. Counting is the CEILING, so it is the
# honest basis for "what does adding a video render cost me" — both sides of
# the comparison sit at the same point on that curve. Each stream counts a
# different range so no two share a prefix-cache entry.
PROMPTS = [
    "Count from 1 to 300. Output only the numbers separated by commas, nothing else.",
    "Count from 301 to 600. Output only the numbers separated by commas, nothing else.",
    "Count from 601 to 900. Output only the numbers separated by commas, nothing else.",
    "Count from 901 to 1200. Output only the numbers separated by commas, nothing else.",
    "Count from 1201 to 1500. Output only the numbers separated by commas, nothing else.",
    "Count from 1501 to 1800. Output only the numbers separated by commas, nothing else.",
]


def one_stream(idx, out):
    """Run a single streaming completion. Records TTFT and decode rate."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
        "max_tokens": MAX_TOK, "temperature": 0.0, "stream": True,
        # REQUIRED for a correct token count. With speculative decoding on
        # (dspark, 5 draft tokens) vLLM emits every ACCEPTED token in a single
        # SSE delta — measured at 2.47 tokens per delta on this model. Counting
        # deltas as tokens therefore understates throughput by ~2.5x. Take the
        # count from vLLM's own usage block instead of counting chunks.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://{HOST}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ndelta = 0
    usage = None
    tlast = t0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]          # final chunk carries the truth
                ch = d.get("choices") or []
                delta = (ch[0].get("delta") or {}) if ch else {}
                # Reasoning models emit reasoning_content deltas too; both are
                # generated tokens and both cost decode time, so count either.
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    ndelta += 1
                    tlast = now
    except Exception as e:  # noqa - a failed stream must not kill the sweep
        out[idx] = {"ok": False, "err": str(e)[:120]}
        return
    decode_s = max(1e-9, tlast - (t0 + (ttft or 0)))
    # Trust vLLM's completion_tokens; fall back to the delta count only if the
    # server did not return usage (then the number is a floor, not the truth).
    ntok = (usage or {}).get("completion_tokens") or ndelta
    out[idx] = {
        "ok": ndelta > 0,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "tokens": ntok,
        "deltas": ndelta,
        "tok_per_delta": round(ntok / ndelta, 2) if ndelta else None,
        # first-token -> last-token, i.e. pure decode, prefill excluded
        "decode_tps": round((ntok - 1) / decode_s, 2) if ntok > 1 else 0.0,
        "wall_s": round(tlast - t0, 3),
    }


def sweep(level):
    out = {}
    threads = [threading.Thread(target=one_stream, args=(i, out))
               for i in range(level)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    good = [v for v in out.values() if v.get("ok")]
    if not good:
        return {"level": level, "ok": 0, "err": [v.get("err") for v in out.values()][:2]}
    toks = sum(v["tokens"] for v in good)
    return {
        "level": level,
        "ok": len(good),
        "failed": level - len(good),
        # Aggregate is total tokens over the wall clock of the whole batch —
        # the number that matters when several agents hit the box at once.
        "agg_tps": round(toks / wall, 2),
        "per_stream_tps": round(sum(v["decode_tps"] for v in good) / len(good), 2),
        "mean_ttft_s": round(sum(v["ttft_s"] or 0 for v in good) / len(good), 3),
        "max_ttft_s": round(max(v["ttft_s"] or 0 for v in good), 3),
        "tokens": toks,
        "wall_s": round(wall, 2),
    }


if __name__ == "__main__":
    print(f"# {LABEL}  ({HOST}, {MODEL}, max_tokens={MAX_TOK})")
    print(f"{'lvl':>4} {'agg tok/s':>10} {'per-stream':>11} {'TTFT mean':>10} "
          f"{'TTFT max':>9} {'tokens':>7} {'wall':>7}  ok")
    rows = []
    for lv in LEVELS:
        r = sweep(lv)
        rows.append(r)
        if not r.get("agg_tps"):
            print(f"{lv:>4}  FAILED  {r.get('err')}")
            continue
        print(f"{lv:>4} {r['agg_tps']:>10} {r['per_stream_tps']:>11} "
              f"{r['mean_ttft_s']:>10} {r['max_ttft_s']:>9} {r['tokens']:>7} "
              f"{r['wall_s']:>7}  {r['ok']}/{lv}")
        time.sleep(2)  # let the scheduler drain between levels
    print("JSON " + json.dumps({"label": LABEL, "host": HOST, "model": MODEL,
                                "max_tokens": MAX_TOK, "rows": rows}))
