#!/usr/bin/env python3
"""Queue a heretic-enhanced H3 render (Sage+Sol+Spectrum+FBC+ESRGAN).

Usage:
  python3 render_enhanced.py --host 10.100.10.2:8188 --prompt "..." [--no-upscale]
  python3 render_enhanced.py --host 10.100.10.2:8188 --prompt-file p.txt --prefix video/run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# local import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_graph import build_enhanced  # noqa: E402


def http_json(url, data=None, timeout=120):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.100.10.2:8188")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prefix", default="video/H3_enhanced")
    ap.add_argument("--first-frame", default=None, help="filename already in Comfy input/")
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--upscale", default="RealESRGAN_x2plus.pth")
    ap.add_argument("--no-spectrum", action="store_true")
    ap.add_argument("--sage", default="auto", help="auto|disabled|…")
    ap.add_argument("--wait", action="store_true", default=True)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    elif args.prompt:
        prompt = args.prompt
    else:
        sys.exit("need --prompt or --prompt-file")

    graph = build_enhanced(
        prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        prefix=args.prefix,
        first_frame=args.first_frame,
        sage=args.sage,
        spectrum=not args.no_spectrum,
        upscale=None if args.no_upscale else args.upscale,
    )

    try:
        r = http_json(
            f"http://{args.host}/prompt",
            {"prompt": graph, "client_id": "h3-enhanced"},
            timeout=60,
        )
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:3000])
        sys.exit(1)

    pid = r.get("prompt_id")
    print(json.dumps(r, indent=1))
    if not pid or args.no_wait:
        return

    t0 = time.time()
    while time.time() - t0 < 7200:
        hist = http_json(f"http://{args.host}/history/{pid}", timeout=30)
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "success" or hist[pid].get("outputs"):
                print(f"SUCCESS in {time.time()-t0:.0f}s")
                print(json.dumps(hist[pid].get("outputs"), indent=1)[:2000])
                return
            if st.get("status_str") == "error":
                print("ERROR", json.dumps(st)[:2000])
                sys.exit(2)
        if int(time.time() - t0) % 30 < 2:
            print(f"  … {time.time()-t0:.0f}s", flush=True)
        time.sleep(5)
    sys.exit("timeout")


if __name__ == "__main__":
    main()
