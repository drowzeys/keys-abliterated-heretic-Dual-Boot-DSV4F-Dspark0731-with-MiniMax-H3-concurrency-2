#!/usr/bin/env python3
"""Dual-H3 continuous 10s clip: Harry Potter → dragon, then dragon over Hogwarts.

Part 1  → .2  (10.100.10.2:8188)  text-to-video, ~5.17s (124 frames)
Part 2  → .3  (10.100.10.3:8188)  first_frame = last frame of part 1 for seam
Stitch  → ffmpeg concat with short crossfade, output ~/Videos/

Co-tenancy: 832x480, length=124, steps=20, res_multistep, --disable-pinned-memory already on servers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HEAD = os.environ.get("H3_HEAD", "10.100.10.2:8188")
WORKER = os.environ.get("H3_WORKER", "10.100.10.3:8188")
# 864x480 (~0.41 MP) then RealESRGAN x2 → ~1728x960 (enhanced stack default)
WIDTH = int(os.environ.get("H3_W", "864"))
HEIGHT = int(os.environ.get("H3_H", "480"))
LENGTH = int(os.environ.get("H3_LEN", "124"))  # 17*7+5 = 124 ≈ 5.17s @ 24fps
STEPS = int(os.environ.get("H3_STEPS", "20"))
SEED1 = int(os.environ.get("SEED1", "42042"))
SEED2 = int(os.environ.get("SEED2", "42043"))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "hp_dragon_dual")))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/hp_dragon_dual"))

PROMPT_P1 = """integrated_multimodal_description: Photorealistic live-action cinema, 8K IMAX detail, natural anamorphic lens, 24fps, 180° shutter motion blur. Cinematography like Alfonso Cuarón x Roger Deakins. Strictly practical and motivated light — grey English daylight through tall Gothic stone windows, warm candle and floating-candle spill, cool stone bounce, soft volumetric dust in the shafts of light. No cartoon, no anime, no CGI plastic skin, no game-engine look, no text, no subtitles, no watermark.

Subject: a teenage wizard boy with messy black hair, round wire spectacles, a lightning-bolt scar on his forehead, wearing a black Hogwarts-style school robe with a red-and-gold striped scarf, white shirt, grey sweater. Highly detailed anatomy: correct facial proportions, visible pores, individual hair strands, fabric weave on the robe, realistic glass reflection in the spectacles.

Setting: a grand Hogwarts-like castle common room / tower chamber — tall arched windows looking over misty Scottish Highlands, stone walls, wooden beams, floating candles, fireplace glow. Continuous single shot, locked camera with very slight handheld breath.

Storyboard — ONE continuous take, no cuts:
[0.0s–1.5s] Medium shot: the boy stands center frame, looking at his hands in quiet fear and wonder as golden sparks begin to crawl under his skin; robe fabric lifts in an unseen wind; candles flicker hard.
[1.5s–3.5s] Transformation: his spine lengthens, shoulders broaden into a powerful reptilian shoulder girdle, skin becomes overlapping iridescent emerald-and-obsidian dragon scales with anatomical accuracy (scapula, deltoid mass, rib cage expanding), fingers fuse into clawed wing digits, spectacles fall and clatter; the face elongates into a noble dragon snout with correct jaw articulation, horns emerging from the skull, pupils becoming vertical slits — painful, visceral metamorphosis, every joint relocating with weight and inertia.
[3.5s–5.17s] Final form holds: a fully transformed mid-sized Western dragon fills the chamber, wings half-unfurled scraping beams, chest heaving, smoke curling from nostrils, scales catching window light with subsurface green fire; he turns his massive head toward the tall open window, muscles coiled to leap. Camera holds.

overall_soundscape: crackling fireplace, wind against stone, fabric tearing, wet organic morphing, bone and scale settling, a deep reptilian inhale, distant castle ambience. No music score; diegetic only. No dialogue.
non_diegetic_music: N/A"""

PROMPT_P2 = """integrated_multimodal_description: Photorealistic live-action cinema, seamless CONTINUATION of the previous shot. Same emerald-and-obsidian Western dragon with exact same scale pattern, horn shape, body proportions and lighting continuity. 8K IMAX, natural anamorphic lens, 24fps, 180° shutter. Practical light: overcast Highland sky, wet stone, soft volumetric mist, no cartoon, no anime, no CGI plastic, no text, no subtitles, no watermark.

The first frame is the last moment of the prior clip: the fully transformed dragon inside a Hogwarts-like tower chamber, wings half-unfurled, head turned toward a tall open Gothic window. Match that pose and identity exactly, then continue motion forward without a hard cut feel.

Storyboard — ONE continuous take, no cuts:
[0.0s–1.5s] The dragon coils and launches through the open arched window in a single powerful push; stone dust and a hanging tapestry whip in the slipstream; camera pushes after him through the window into open air.
[1.5s–3.5s] Exterior: the dragon spreads full wings and banks into a wide hover over the Hogwarts-like castle — turrets, Great Hall roof, lake and black forest below, mist in the valleys; correct wing aerodynamics, membrane stretch, primary feathers of membrane, body weight shifting in the hover, tail counterbalancing.
[3.5s–5.17s] Hero aerial hold: the dragon hangs in a slow hover circle around the main castle keep, wings beating with heavy downstroke, mist swirling, scale detail sharp in diffuse daylight, castle architecture stable and rigid beneath him. Camera orbits gently with him.

Physical consistency: same dragon identity as the first frame, rigid castle geometry, gravity and inertia respected, no morphing into another creature, no extra limbs, no text.
overall_soundscape: wind roar, heavy wing beats, distant lake water, castle flag snap, deep dragon breath. No music. No dialogue.
non_diegetic_music: N/A"""


def http_json(url: str, data=None, timeout=120):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def upload_image(host: str, path: Path, name: str | None = None) -> str:
    """Upload image to ComfyUI input folder via /upload/image."""
    import mimetypes
    import uuid

    name = name or path.name
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    raw = path.read_bytes()
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n".encode())
    parts.append(raw)
    parts.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())
    data = b"".join(parts)
    req = urllib.request.Request(
        f"http://{host}/upload/image",
        data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read().decode())
    # {"name": "...", "subfolder": "", "type": "input"}
    return resp.get("name", name)


def build_t2v(prompt: str, seed: int, prefix: str, first_frame: str | None = None) -> dict:
    """Heretic-enhanced graph: Sage → SolAttnPatch → Spectrum → FBC → ESRGAN x2."""
    # Prefer enhanced_graph if present (same package)
    try:
        from enhanced_graph import build_enhanced
        return build_enhanced(
            prompt,
            seed=seed,
            width=WIDTH,
            height=HEIGHT,
            length=LENGTH,
            steps=STEPS,
            prefix=prefix,
            first_frame=first_frame,
            sage="auto",
            spectrum=True,  # ComfyUI v0.30.1+
            upscale="RealESRGAN_x2plus.pth",
            te="H3/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        )
    except ImportError:
        pass
    # Fallback minimal chain
    g = {
        "6": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "50": {"class_type": "PathchSageAttentionKJ",
               "inputs": {"model": ["6", 0], "sage_attention": "auto", "allow_compile": False}},
        "7": {"class_type": "SolAttnPatch",
              "inputs": {"model": ["50", 0], "tau": 1.3, "start_percent": 0.2,
                         "end_percent": 0.9, "min_tokens": 4096, "int8_qk": True,
                         "sink_conditioning": "exact_kv_and_rows", "morton": False,
                         "morton_curve": "2d_frame", "verbose": False, "use_tma": False}},
        "51": {"class_type": "SpectrumApplyMiniMaxH3",
               "inputs": {"model": ["7", 0], "enabled": True, "blend_weight": 0.5,
                          "degree": 4, "ridge_lambda": 0.1, "window_size": 2.0,
                          "flex_window": 0.75, "warmup_steps": 5, "tail_actual_steps": 1,
                          "max_history": 8, "debug": False, "history_storage": "system_ram"}},
        "8": {"class_type": "H3FirstBlockCache",
              "inputs": {"model": ["51", 0], "threshold": 0.08, "start_step": 3,
                         "end_dense_steps": 2, "max_consecutive_skips": 2}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "H3/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
                          "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "104": {"class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "clip": ["13", 0], "vae": ["11", 0],
                    "prompt": prompt,
                    "width": WIDTH, "height": HEIGHT, "length": LENGTH,
                }},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["8", 0], "scheduler": "simple",
                         "steps": STEPS, "denoise": 1.0}},
        "16": {"class_type": "BasicGuider",
               "inputs": {"model": ["8", 0], "conditioning": ["104", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0],
                          "sampler": ["17", 0], "sigmas": ["9", 0],
                          "latent_image": ["104", 1]}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "60": {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": "RealESRGAN_x2plus.pth"}},
        "61": {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["60", 0], "image": ["10", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["61", 0], "audio": ["23", 0], "fps": 24.0}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": prefix,
                          "format": "auto", "codec": "auto"}},
    }
    if first_frame:
        g["105"] = {"class_type": "LoadImage",
                    "inputs": {"image": first_frame, "upload": "image"}}
        g["104"]["inputs"]["first_frame"] = ["105", 0]
    return g


def queue(host: str, graph: dict, client_id: str) -> str:
    try:
        r = http_json(f"http://{host}/prompt",
                      {"prompt": graph, "client_id": client_id}, timeout=60)
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:4000]
        raise RuntimeError(f"queue failed HTTP {e.code}: {body}") from e
    if "error" in r:
        raise RuntimeError(f"queue error: {r}")
    return r["prompt_id"]


def wait_done(host: str, pid: str, label: str, timeout: int = 7200) -> dict:
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout:
        try:
            hist = http_json(f"http://{host}/history/{pid}", timeout=30)
        except Exception:
            time.sleep(8)
            continue
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            st = status.get("status_str", "")
            if st == "success" or entry.get("outputs"):
                print(f"  [{label}] SUCCESS in {time.time()-t0:.0f}s", flush=True)
                return entry
            if st == "error" or status.get("completed") is False and status.get("status_str") == "error":
                raise RuntimeError(f"{label} failed: {json.dumps(status)[:1500]}")
            # some builds mark error differently
            msgs = status.get("messages") or []
            for m in msgs:
                if isinstance(m, list) and m and m[0] == "execution_error":
                    raise RuntimeError(f"{label} execution_error: {m}")
        # progress ping
        if time.time() - last > 30:
            elapsed = time.time() - t0
            try:
                q = http_json(f"http://{host}/queue", timeout=10)
                running = q.get("queue_running") or []
                pending = q.get("queue_pending") or []
                print(f"  [{label}] {elapsed:.0f}s elapsed  running={len(running)} pending={len(pending)}", flush=True)
            except Exception:
                print(f"  [{label}] {elapsed:.0f}s elapsed", flush=True)
            last = time.time()
        time.sleep(8)
    raise TimeoutError(f"{label} timed out after {timeout}s")


def find_output_video(entry: dict) -> tuple[str, str, str]:
    """Return (filename, subfolder, folder_type) from history outputs."""
    outputs = entry.get("outputs") or {}
    for _nid, out in outputs.items():
        for key in ("gifs", "videos", "images"):
            items = out.get(key) or []
            for it in items:
                fn = it.get("filename") or ""
                if fn.lower().endswith((".mp4", ".webm", ".mkv", ".mov", ".avi")):
                    return fn, it.get("subfolder", ""), it.get("type", "output")
                # CreateVideo sometimes stores as images entry with .mp4
                if "mp4" in fn.lower() or "webm" in fn.lower():
                    return fn, it.get("subfolder", ""), it.get("type", "output")
    # fallback: scan all
    for _nid, out in outputs.items():
        for key, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                fn = it.get("filename") or ""
                if any(fn.lower().endswith(ext) for ext in (".mp4", ".webm", ".mkv")):
                    return fn, it.get("subfolder", ""), it.get("type", "output")
    raise RuntimeError(f"no video in outputs: {json.dumps(outputs)[:2000]}")


def download_output(host: str, filename: str, subfolder: str, folder_type: str, dest: Path):
    q = f"filename={urllib.request.quote(filename)}&subfolder={urllib.request.quote(subfolder)}&type={folder_type}"
    url = f"http://{host}/view?{q}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=600) as r:
        dest.write_bytes(r.read())
    print(f"  saved {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)


def extract_last_frame(video: Path, frame_path: Path):
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    # last frame
    subprocess.check_call([
        "ffmpeg", "-y", "-sseof", "-0.05", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(frame_path),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  last frame → {frame_path}", flush=True)


def stitch(part1: Path, part2: Path, out: Path):
    """Seamless stitch: short xfade across the join for continuity."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # probe durations
    def dur(p):
        o = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(p),
        ], text=True).strip()
        return float(o)

    d1, d2 = dur(part1), dur(part2)
    xfade = 0.35  # seconds of crossfade
    offset = max(0.0, d1 - xfade)
    # normalize both to same params then xfade
    filtergraph = (
        f"[0:v]fps=24,format=yuv420p,setsar=1[v0];"
        f"[1:v]fps=24,format=yuv420p,setsar=1[v1];"
        f"[v0][v1]xfade=transition=fade:duration={xfade}:offset={offset}[v];"
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0];"
        f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
        f"[a0][a1]acrossfade=d={xfade}[a]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(part1), "-i", str(part2),
        "-filter_complex", filtergraph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    print("  stitching:", " ".join(cmd[:8]), "...", flush=True)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        # fallback hard cut if audio missing on one side
        print("  xfade failed; hard concat fallback", flush=True)
        lst = WORK_DIR / "concat.txt"
        lst.write_text(f"file '{part1}'\nfile '{part2}'\n")
        subprocess.check_call([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c:v", "libx264", "-crf", "16", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out),
        ])
    print(f"  FINAL → {out} ({out.stat().st_size/1e6:.1f} MB)  d1={d1:.2f}s d2={d2:.2f}s", flush=True)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for host, label in ((HEAD, "head"), (WORKER, "worker")):
        try:
            s = http_json(f"http://{host}/system_stats", timeout=5)
            print(f"H3 {label} {host}: ram_free={s['system']['ram_free']/1e9:.1f}G", flush=True)
        except Exception as e:
            print(f"H3 {label} DOWN: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Part 1: transform on .2 ---
    print("\n=== PART 1: Potter → dragon on", HEAD, "===", flush=True)
    g1 = build_t2v(PROMPT_P1, SEED1, "video/hp_dragon_p1")
    pid1 = queue(HEAD, g1, "hp-dragon-p1")
    print(f"  queued prompt_id={pid1}", flush=True)
    entry1 = wait_done(HEAD, pid1, "p1")
    fn1, sub1, typ1 = find_output_video(entry1)
    part1 = WORK_DIR / "part1_transform.mp4"
    download_output(HEAD, fn1, sub1, typ1, part1)
    shutil.copy2(part1, OUT_DIR / "part1_transform.mp4")

    # last frame for continuity
    last = WORK_DIR / "p1_last.png"
    extract_last_frame(part1, last)

    # --- Part 2: flight on .3 with first_frame ---
    print("\n=== PART 2: dragon over Hogwarts on", WORKER, "===", flush=True)
    uploaded = upload_image(WORKER, last, "hp_dragon_p1_last.png")
    print(f"  uploaded first_frame as {uploaded}", flush=True)
    g2 = build_t2v(PROMPT_P2, SEED2, "video/hp_dragon_p2", first_frame=uploaded)
    pid2 = queue(WORKER, g2, "hp-dragon-p2")
    print(f"  queued prompt_id={pid2}", flush=True)
    entry2 = wait_done(WORKER, pid2, "p2")
    fn2, sub2, typ2 = find_output_video(entry2)
    part2 = WORK_DIR / "part2_flight.mp4"
    download_output(WORKER, fn2, sub2, typ2, part2)
    shutil.copy2(part2, OUT_DIR / "part2_flight.mp4")

    # --- stitch ---
    print("\n=== STITCH ===", flush=True)
    final = OUT_DIR / "harry_potter_dragon_hogwarts_10s.mp4"
    stitch(part1, part2, final)
    # also leave a hard-cut master
    hard = OUT_DIR / "harry_potter_dragon_hogwarts_10s_hardcut.mp4"
    lst = WORK_DIR / "concat.txt"
    lst.write_text(f"file '{part1.resolve()}'\nfile '{part2.resolve()}'\n")
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(hard),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\nDone.\n  continuous: {final}\n  hardcut:    {hard}", flush=True)


if __name__ == "__main__":
    main()
