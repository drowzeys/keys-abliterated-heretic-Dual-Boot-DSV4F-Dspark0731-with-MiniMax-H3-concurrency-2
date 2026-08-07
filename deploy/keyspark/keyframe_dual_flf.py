#!/usr/bin/env python3
"""Parallel dual-H3 ~10s story via quality keyframes + first/last-frame (FLF).

Built on Tony's dual-H3 + DS4 co-tenancy factory:
  https://github.com/tonyd2wild/ds4-h3-video-gen-factory  (tonyd2wild)
Keyspark extension: quality-first parallel path on that foundation.

Priority: **character / identity quality first**. Parallelism is only for wall-clock.

Quality-first design (default H3_QUALITY_ID=1):
  Phase 0  PARALLEL quality keyframes (independent — no cheap FLF scouts):
             .2  I0  human start  — full steps, detailed identity prompt
                 (optional H3_I0_REF image seed for face lock)
             .3  I1  mid dragon   — full/high steps, hall window pose
  Phase 1  I2 end dragon on free box (first_frame=I1 for continuity)
  Phase 2  PARALLEL full ~5s FLF clips:
             .2  first=I0 last=I1  → part A (transform)
             .3  first=I1 last=I2  → part B (flight)
  Phase 3  xfade stitch → ~10s

Fast mode (H3_QUALITY_ID=0): old short 10-step scouts (speed over face).

Env:
  H3_QUALITY_ID=1          quality-first (default)
  H3_I0_REF=/path/to.png   seed human face from a known-good frame
  H3_STEPS / H3_LEN        full clip denoise (default 20 / 124)
  H3_STEPS_I0 / H3_LEN_I0  human keyframe (default = full steps / ~2s)
  H3_STEPS_ID / H3_LEN_ID  mid+end keyframes (default 20 / ~1.4s in quality)
  H3_STEPS_KF / H3_LEN_KF  used only in fast mode
  H3_UPSCALE=0             skip RealESRGAN if face looks over-processed
  SEED, H3_HEAD, H3_WORKER, OUT_DIR, WORK_DIR, H3_SPECTRUM

Usage:
  PYTHONPATH=... python3 keyframe_dual_flf.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_graph import build_enhanced  # noqa: E402

HEAD = os.environ.get("H3_HEAD", "10.100.10.2:8188")
WORKER = os.environ.get("H3_WORKER", "10.100.10.3:8188")
W = int(os.environ.get("H3_W", "864"))
H = int(os.environ.get("H3_H", "480"))
LEN_FULL = int(os.environ.get("H3_LEN", "124"))
STEPS_FULL = int(os.environ.get("H3_STEPS", "20"))

# Quality-first is default: identity beats scout speed.
QUALITY_ID = os.environ.get("H3_QUALITY_ID", "1") not in ("0", "false", "no", "fast")
# Fast-mode scout defaults (only when QUALITY_ID=0)
LEN_KF = int(os.environ.get("H3_LEN_KF", "22"))
STEPS_KF = int(os.environ.get("H3_STEPS_KF", "10"))
# Identity keyframes — full denoise for human face by default
if QUALITY_ID:
    STEPS_I0 = int(os.environ.get("H3_STEPS_I0", str(STEPS_FULL)))
    LEN_I0 = int(os.environ.get("H3_LEN_I0", "49"))   # ~2.0s @ 24fps — room for a clean portrait
    STEPS_ID = int(os.environ.get("H3_STEPS_ID", str(STEPS_FULL)))
    LEN_ID = int(os.environ.get("H3_LEN_ID", "33"))    # ~1.4s mid/end still-energy
else:
    STEPS_I0 = int(os.environ.get("H3_STEPS_I0", str(STEPS_KF)))
    LEN_I0 = int(os.environ.get("H3_LEN_I0", str(LEN_KF)))
    STEPS_ID = int(os.environ.get("H3_STEPS_ID", str(STEPS_KF)))
    LEN_ID = int(os.environ.get("H3_LEN_ID", str(LEN_KF)))

SEED = int(os.environ.get("SEED", "42042"))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "hp_dragon_parallel")))
WORK = Path(os.environ.get("WORK_DIR", "/tmp/hp_dragon_parallel"))
SPECTRUM = os.environ.get("H3_SPECTRUM", "1") not in ("0", "false", "no")
UPSCALE = os.environ.get("H3_UPSCALE", "RealESRGAN_x2plus.pth")
if UPSCALE in ("0", "none", "None"):
    UPSCALE = None
# Optional known-good face/frame to seed I0 (e.g. frame from sequential original)
I0_REF = os.environ.get("H3_I0_REF", "").strip() or None
if I0_REF:
    I0_REF = str(Path(I0_REF).expanduser().resolve())

# ---------------------------------------------------------------------------
# Prompts — quality-first identity (ported / aligned with hp_dragon_dual.py)
# ---------------------------------------------------------------------------

LOOK = """integrated_multimodal_description: Photorealistic live-action cinema, 8K IMAX detail, natural anamorphic lens, 24fps, 180° shutter motion blur. Cinematography like Alfonso Cuarón x Roger Deakins. Strictly practical and motivated light — grey English daylight through tall Gothic stone windows, warm candle and floating-candle spill, cool stone bounce, soft volumetric dust in the shafts of light. No cartoon, no anime, no CGI plastic skin, no game-engine look, no text, no subtitles, no watermark.

Identity lock: same teenage wizard whenever human — messy black hair, round wire spectacles, lightning-bolt scar on forehead, black Hogwarts-style school robe, red-and-gold striped Gryffindor scarf, white shirt, grey sweater. Highly detailed anatomy: correct facial proportions, visible pores, individual hair strands, fabric weave, realistic glass reflection in the spectacles. Same emerald-and-obsidian Western dragon scale pattern, horn shape, and body proportions whenever the dragon appears."""

# I0 — human portrait only (NO dragon in frame). Quality of this frame sets part A face.
PROMPT_I0 = f"""{LOOK}

KEYFRAME / living portrait (minimal motion). Subject only: the teenage wizard boy alone in a grand Hogwarts-like great hall / common room — tall arched windows looking over misty Scottish Highlands, stone walls, wooden beams, floating candles, fireplace glow. Medium shot, center frame, locked camera with very slight handheld breath.

He looks at his hands in quiet fear and wonder as golden sparks begin to crawl under his skin; robe fabric lifts in an unseen wind; candles flicker hard. Spectacles on; scar visible; face fully readable and photoreal. NO dragon in frame yet. NO second character.

overall_soundscape: crackling fireplace, wind against stone, soft fabric. No music. No dialogue.
non_diegetic_music: N/A"""

PROMPT_I1 = f"""{LOOK}

KEYFRAME / still-energy shot (minimal motion): a fully transformed mid-sized Western dragon fills the same Hogwarts-like great hall, wings half-unfurled scraping beams, head turned toward a tall open Gothic window with misty Scottish hills outside. Emerald-and-obsidian scales catch window light with subsurface green fire; chest heaving, smoke curling from nostrils, muscles coiled to leap. Single continuous slow push-in, almost a living tableau. Same chamber architecture as the human keyframe (candles, stone, window).

overall_soundscape: fireplace, deep reptilian inhale, wind. No music. No dialogue.
non_diegetic_music: N/A"""

PROMPT_I2 = f"""{LOOK}

KEYFRAME / living aerial still (minimal motion): the same emerald-and-obsidian dragon banks in a hover over a Hogwarts-like castle — turrets, Great Hall roof, lake, misty Highlands below. Hero aerial angle, wings fully extended with correct membrane stretch, tail counterbalancing. Exact same dragon identity (scale pattern, horns, proportions) as the mid hall keyframe.

overall_soundscape: wind, distant water, wing tension. No music. No dialogue.
non_diegetic_music: N/A"""

# Full 5s arms — storyboard + identity (match sequential dual prompts)
PROMPT_A = f"""{LOOK}

ONE continuous take, no cuts. FIRST FRAME is the human wizard at the start of the metamorphosis (match face, glasses, scarf, robe, hall lighting exactly). LAST FRAME must match the fully transformed dragon at the open window (identity, pose, lighting locked).

Storyboard — ONE continuous take:
[0.0s–1.5s] Medium shot: the boy stands center frame, looking at his hands in quiet fear and wonder as golden sparks crawl under his skin; robe lifts in wind; candles flicker hard.
[1.5s–3.5s] Transformation: spine lengthens, shoulders broaden into a reptilian shoulder girdle, skin becomes overlapping iridescent emerald-and-obsidian dragon scales with anatomical accuracy; fingers fuse into clawed wing digits; spectacles fall and clatter; face elongates into a noble dragon snout with correct jaw articulation, horns emerging, pupils becoming vertical slits — painful, visceral metamorphosis with weight and inertia.
[3.5s–5.17s] Final form holds: fully transformed mid-sized Western dragon fills the chamber, wings half-unfurled, head toward the tall open window, muscles coiled to leap. Camera holds.

overall_soundscape: crackling fireplace, wind against stone, fabric tearing, wet organic morphing, bone and scale settling, deep reptilian inhale. No music. No dialogue.
non_diegetic_music: N/A"""

PROMPT_B = f"""{LOOK}

ONE continuous take, no cuts. FIRST FRAME is the fully transformed dragon at the open window (exact identity match to mid keyframe). LAST FRAME is the same dragon hovering over the castle.

Storyboard — ONE continuous take:
[0.0s–1.5s] The dragon coils and launches through the open arched window; stone dust and a hanging tapestry whip in the slipstream; camera pushes after him into open air.
[1.5s–3.5s] Exterior: full wings, banks into a wide hover over the Hogwarts-like castle — turrets, lake, mist; correct wing aerodynamics and body weight in the hover.
[3.5s–5.17s] Hero aerial hold: slow hover circle around the main keep, heavy downstroke, mist swirling, scale detail sharp in diffuse daylight.

Physical consistency: same dragon identity as the first frame, rigid castle geometry, gravity respected, no morphing into another creature, no extra limbs, no text.
overall_soundscape: wind roar, heavy wing beats, distant lake water, deep dragon breath. No music. No dialogue.
non_diegetic_music: N/A"""


def http_json(url, data=None, timeout=120):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def upload_image(host: str, path: Path, name: str) -> str:
    import mimetypes
    import uuid
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        raw,
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    data = b"".join(parts)
    req = urllib.request.Request(
        f"http://{host}/upload/image", data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read().decode())
    return resp.get("name", name)


def queue(host: str, graph: dict, client_id: str) -> str:
    try:
        r = http_json(f"http://{host}/prompt", {"prompt": graph, "client_id": client_id}, timeout=60)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"queue {host}: {e.read().decode()[:2500]}") from e
    if "error" in r:
        raise RuntimeError(r)
    return r["prompt_id"]


def wait_done(host: str, pid: str, label: str, timeout: int = 7200) -> dict:
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout:
        try:
            hist = http_json(f"http://{host}/history/{pid}", timeout=30)
        except Exception:
            time.sleep(5)
            continue
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status", {})
            s = st.get("status_str", "")
            if s == "success" or entry.get("outputs"):
                print(f"  [{label}] SUCCESS {time.time()-t0:.0f}s", flush=True)
                return entry
            if s == "error":
                for m in st.get("messages") or []:
                    if isinstance(m, list) and m and m[0] == "execution_error":
                        raise RuntimeError(f"{label}: {m[1].get('exception_message','')[:1200]}")
                raise RuntimeError(f"{label}: {json.dumps(st)[:1200]}")
        if time.time() - last > 20:
            print(f"  [{label}] {time.time()-t0:.0f}s …", flush=True)
            last = time.time()
        time.sleep(4)
    raise TimeoutError(label)


def find_video(entry: dict) -> tuple[str, str, str]:
    for _nid, out in (entry.get("outputs") or {}).items():
        for key, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                fn = it.get("filename") or ""
                if any(fn.lower().endswith(ext) for ext in (".mp4", ".webm", ".mkv")):
                    return fn, it.get("subfolder", ""), it.get("type", "output")
    raise RuntimeError(f"no video: {json.dumps(entry.get('outputs'))[:1500]}")


def download(host: str, fn: str, sub: str, typ: str, dest: Path):
    q = f"filename={urllib.request.quote(fn)}&subfolder={urllib.request.quote(sub)}&type={typ}"
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=600) as r:
        dest.write_bytes(r.read())
    print(f"  saved {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)


def extract_frame(video: Path, out: Path, when: str = "last"):
    """when: first | last | mid | best_face (early stable frame for portraits)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if when == "first":
        cmd = ["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    elif when in ("mid", "best_face"):
        # Prefer a slightly settled frame (not frame 0 artifacts) for face keyframes
        cmd = ["ffmpeg", "-y", "-ss", "0.35", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-sseof", "-0.05", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  frame[{when}] → {out}", flush=True)


def prepare_i0_ref() -> str | None:
    """If H3_I0_REF set, copy into work dir and upload name left to caller."""
    if not I0_REF:
        return None
    src = Path(I0_REF)
    if not src.is_file():
        raise FileNotFoundError(f"H3_I0_REF not found: {src}")
    dest = WORK / "I0_ref_input.png"
    # Normalize to png via ffmpeg (handles jpg/mp4 frame paths already being images)
    if src.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv"):
        subprocess.check_call(
            ["ffmpeg", "-y", "-ss", "0.5", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    print(f"  I0_REF prepared {src} → {dest}", flush=True)
    return str(dest)


def render(
    host: str,
    label: str,
    prompt: str,
    seed: int,
    prefix: str,
    length: int,
    steps: int,
    first: str | None,
    last: str | None,
    upscale: str | None,
) -> Path:
    g = build_enhanced(
        prompt, seed=seed, width=W, height=H, length=length, steps=steps,
        prefix=prefix, first_frame=first, last_frame=last,
        sage="auto", spectrum=SPECTRUM, upscale=upscale,
    )
    pid = queue(host, g, label)
    print(
        f"  [{label}] queued {pid} on {host}  "
        f"len={length} steps={steps} first={bool(first)} last={bool(last)}",
        flush=True,
    )
    entry = wait_done(host, pid, label)
    fn, sub, typ = find_video(entry)
    dest = WORK / f"{label}.mp4"
    download(host, fn, sub, typ, dest)
    return dest


def parallel_pair(job_a, job_b):
    """job_* are callables returning a path; run in threads, raise if either fails."""
    results: dict[str, Path | BaseException] = {}

    def run(key, fn):
        try:
            results[key] = fn()
        except BaseException as e:
            results[key] = e

    ta = threading.Thread(target=run, args=("a", job_a))
    tb = threading.Thread(target=run, args=("b", job_b))
    t0 = time.time()
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    wall = time.time() - t0
    if isinstance(results.get("a"), BaseException):
        raise results["a"]
    if isinstance(results.get("b"), BaseException):
        raise results["b"]
    print(f"  parallel wall {wall:.0f}s", flush=True)
    return results["a"], results["b"], wall


def stitch(a: Path, b: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    def dur(p):
        return float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(p),
        ], text=True).strip())

    d1, d2 = dur(a), dur(b)
    xfade = 0.25
    offset = max(0.0, d1 - xfade)
    fc = (
        f"[0:v]fps=24,format=yuv420p,setsar=1[v0];"
        f"[1:v]fps=24,format=yuv420p,setsar=1[v1];"
        f"[v0][v1]xfade=transition=fade:duration={xfade}:offset={offset}[v];"
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0];"
        f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
        f"[a0][a1]acrossfade=d={xfade}[a]"
    )
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", str(a), "-i", str(b),
            "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        lst = WORK / "concat.txt"
        lst.write_text(f"file '{a.resolve()}'\nfile '{b.resolve()}'\n")
        subprocess.check_call([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c:v", "libx264", "-crf", "16", "-c:a", "aac", "-movflags", "+faststart", str(out),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  FINAL {out} ({out.stat().st_size/1e6:.1f} MB) d1={d1:.2f}s d2={d2:.2f}s", flush=True)


def run_quality(t_all: float) -> tuple[Path, Path, Path, float, float, float]:
    """Quality-first: full-step independent I0+I1 in parallel, then I2, then dual FLF."""
    # Optional face seed
    i0_first_name = None
    ref_path = prepare_i0_ref()
    if ref_path:
        i0_first_name = upload_image(HEAD, Path(ref_path), "I0_ref.png")
        print(f"  seeding I0 from H3_I0_REF → {i0_first_name}", flush=True)

    print(
        f"\n=== PHASE 0: quality I0 (human) + I1 (mid dragon) in parallel ===\n"
        f"  I0: len={LEN_I0} steps={STEPS_I0}  |  I1: len={LEN_ID} steps={STEPS_ID}\n"
        f"  (no FLF constraint on I0 — face is free T2V / optional ref)",
        flush=True,
    )
    t0 = time.time()

    def job_i0():
        return render(
            HEAD, "kf_start", PROMPT_I0, SEED, "video/kf_i0_human",
            LEN_I0, STEPS_I0, i0_first_name, None, None,  # no last_frame — pure identity
        )

    def job_i1():
        return render(
            WORKER, "kf_mid", PROMPT_I1, SEED + 3, "video/kf_i1_mid",
            LEN_ID, STEPS_ID, None, None, None,
        )

    start_vid, mid_vid, wall0 = parallel_pair(job_i0, job_i1)
    i0 = WORK / "I0_start.png"
    i1 = WORK / "I1_mid.png"
    # Settled portrait frame for face; last frame for mid dragon pose
    extract_frame(start_vid, i0, "best_face")
    extract_frame(mid_vid, i1, "last")
    i0_h = upload_image(HEAD, i0, "I0_start.png")
    i1_h = upload_image(HEAD, i1, "I1_mid.png")
    i1_w = upload_image(WORKER, i1, "I1_mid.png")
    print(f"  phase0 wall {wall0:.0f}s  I0+I1 ready", flush=True)

    # Phase 1: I2 on worker (head free) — continuity from I1
    print(
        f"\n=== PHASE 1: quality I2 (end dragon) on {WORKER} ===\n"
        f"  I2: len={LEN_ID} steps={STEPS_ID} first=I1",
        flush=True,
    )
    t1 = time.time()
    end_vid = render(
        WORKER, "kf_end", PROMPT_I2, SEED + 2, "video/kf_i2_end",
        LEN_ID, STEPS_ID, i1_w, None, None,
    )
    i2 = WORK / "I2_end.png"
    extract_frame(end_vid, i2, "last")
    i2_w = upload_image(WORKER, i2, "I2_end.png")
    wall1 = time.time() - t1
    print(f"  phase1 wall {wall1:.0f}s (total {time.time()-t_all:.0f}s)", flush=True)

    return i0_h, i1_h, i1_w, i2_w, wall0, wall1


def run_fast(t_all: float) -> tuple[str, str, str, str, float, float]:
    """Legacy fast scouts (speed over face). Kept for H3_QUALITY_ID=0."""
    print("\n=== PHASE 0 (fast): mid keyframe I1 on", HEAD, "===", flush=True)
    t0 = time.time()
    mid_vid = render(
        HEAD, "kf_mid", PROMPT_I1, SEED, "video/kf_mid",
        LEN_KF, STEPS_KF, None, None, None,
    )
    i1 = WORK / "I1_mid.png"
    extract_frame(mid_vid, i1, "last")
    i1_name_h = upload_image(HEAD, i1, "I1_mid.png")
    i1_name_w = upload_image(WORKER, i1, "I1_mid.png")
    wall0 = time.time() - t0
    print(f"  phase0 wall {wall0:.0f}s", flush=True)

    print("\n=== PHASE 1 (fast): parallel I0 + I2 scouts ===", flush=True)
    t1 = time.time()

    def job_start():
        return render(
            HEAD, "kf_start", PROMPT_I0, SEED + 1, "video/kf_start",
            LEN_KF, STEPS_KF, None, i1_name_h, None,
        )

    def job_end():
        return render(
            WORKER, "kf_end", PROMPT_I2, SEED + 2, "video/kf_end",
            LEN_KF, STEPS_KF, i1_name_w, None, None,
        )

    start_vid, end_vid, wall1 = parallel_pair(job_start, job_end)
    i0 = WORK / "I0_start.png"
    i2 = WORK / "I2_end.png"
    extract_frame(start_vid, i0, "first")
    extract_frame(end_vid, i2, "last")
    i0_h = upload_image(HEAD, i0, "I0_start.png")
    i2_w = upload_image(WORKER, i2, "I2_end.png")
    print(f"  phase1 wall {wall1:.0f}s (total {time.time()-t_all:.0f}s)", flush=True)
    return i0_h, i1_name_h, i1_name_w, i2_w, wall0, wall1


def main():
    t_all = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mode = "QUALITY-FIRST" if QUALITY_ID else "FAST-SCOUT"
    print(
        f"mode={mode}  spectrum={SPECTRUM} upscale={UPSCALE}\n"
        f"  full: len={LEN_FULL} steps={STEPS_FULL}\n"
        f"  I0:   len={LEN_I0} steps={STEPS_I0}  ref={I0_REF or 'none'}\n"
        f"  I1/I2:len={LEN_ID} steps={STEPS_ID}",
        flush=True,
    )

    for host, name in ((HEAD, ".2"), (WORKER, ".3")):
        try:
            s = http_json(f"http://{host}/system_stats", timeout=5)
            print(f"H3 {name} {host} ram_free={s['system']['ram_free']/1e9:.1f}G", flush=True)
        except Exception as e:
            print(f"H3 {name} DOWN: {e}", file=sys.stderr)
            sys.exit(1)

    if QUALITY_ID:
        i0_h, i1_h, i1_w, i2_w, wall0, wall1 = run_quality(t_all)
    else:
        i0_h, i1_h, i1_w, i2_w, wall0, wall1 = run_fast(t_all)

    # --- Phase 2: parallel full 5s FLF clips ---
    print("\n=== PHASE 2: parallel full FLF 5s clips ===", flush=True)
    t2 = time.time()

    def job_a():
        return render(
            HEAD, "partA_transform", PROMPT_A, SEED + 10, "video/partA_flf",
            LEN_FULL, STEPS_FULL, i0_h, i1_h, UPSCALE,
        )

    def job_b():
        return render(
            WORKER, "partB_flight", PROMPT_B, SEED + 11, "video/partB_flf",
            LEN_FULL, STEPS_FULL, i1_w, i2_w, UPSCALE,
        )

    part_a, part_b, wall2 = parallel_pair(job_a, job_b)
    shutil.copy2(part_a, OUT_DIR / "partA_transform.mp4")
    shutil.copy2(part_b, OUT_DIR / "partB_flight.mp4")
    # keep keyframes in out for QA
    for name in ("I0_start.png", "I1_mid.png", "I2_end.png"):
        src = WORK / name
        if src.is_file():
            shutil.copy2(src, OUT_DIR / name)
    print(f"  phase2 wall {wall2:.0f}s", flush=True)

    # --- Phase 3: stitch ---
    print("\n=== PHASE 3: stitch ===", flush=True)
    final = OUT_DIR / "harry_potter_dragon_parallel_10s.mp4"
    stitch(part_a, part_b, final)
    lst = WORK / "concat.txt"
    lst.write_text(f"file '{part_a.resolve()}'\nfile '{part_b.resolve()}'\n")
    hard = OUT_DIR / "harry_potter_dragon_parallel_10s_hardcut.mp4"
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(hard),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total = time.time() - t_all
    print(f"\nDone in {total:.0f}s wall (~{total/60:.1f} min)  mode={mode}", flush=True)
    print(f"  continuous: {final}", flush=True)
    print(f"  hardcut:    {hard}", flush=True)
    print(
        f"  timing: phase0={wall0:.0f}s phase1={wall1:.0f}s phase2={wall2:.0f}s  "
        f"(parallel full arms = half of sequential dual)",
        flush=True,
    )
    (OUT_DIR / "TIMING.txt").write_text(
        f"mode={mode}\n"
        f"total_wall_s={total:.1f}\n"
        f"phase0_identity_keyframes_s={wall0:.1f}\n"
        f"phase1_end_keyframe_s={wall1:.1f}\n"
        f"phase2_parallel_full_s={wall2:.1f}\n"
        f"spectrum={SPECTRUM} upscale={UPSCALE}\n"
        f"len_full={LEN_FULL} steps_full={STEPS_FULL}\n"
        f"len_i0={LEN_I0} steps_i0={STEPS_I0} len_id={LEN_ID} steps_id={STEPS_ID}\n"
        f"i0_ref={I0_REF or ''}\n"
        f"quality_id={QUALITY_ID}\n"
        f"note=quality-first: I0 is free T2V (or ref) at full steps; FLF only on full 5s arms\n"
    )


if __name__ == "__main__":
    main()
