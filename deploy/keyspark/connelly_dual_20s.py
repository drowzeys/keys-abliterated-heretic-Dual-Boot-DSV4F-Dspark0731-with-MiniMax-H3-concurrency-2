#!/usr/bin/env python3
"""Parallel dual-H3 ~20s clip: Jennifer Connelly / Career Opportunities.

Mounts the keyspark dual-boot concurrency-2 stack (two H3 boxes at once):
  .2:8188  arm A  ->  shot1 + shot2 (chained, first-frame seeded)
  .3:8188  arm B  ->  shot3 + shot4 (chained, first-frame seeded)
        parallel; then single 4-way xfade stitch -> final ~20s clip.

Uses the repo build_enhanced() graph (Sage->Sol->Spectrum->FBC->ESRGAN x2)
and DS4 ablit stays co-tenant on .2:8888.  H3 co-tenancy rule: one heavy
FLF job per Spark at a time (concurrency=2 = two boxes, not 2 jobs/GPU).

Usage:
  python3 connelly_dual_20s.py
  UPSCALE=0 python3 connelly_dual_20s.py        # skip ESRGAN (faster, softer)
  STEPS=16 python3 connelly_dual_20s.py

Output: ~/Videos/connelly_pony_20s/connelly_pony_20s.mp4 (+ per-shot files)

Based on tonyd2wild/ds4-h3-video-gen-factory (keyspark dual-boot ext).
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
FRAMES = int(os.environ.get("H3_LEN", "124"))   # ~5.17s @ 24fps per clip
STEPS = int(os.environ.get("STEPS", "20"))
SEED = int(os.environ.get("SEED", "99171"))
SPECTRUM = os.environ.get("H3_SPECTRUM", "1") not in ("0", "false", "no")
UPSCALE = os.environ.get("UPSCALE", "RealESRGAN_x2plus.pth")
if UPSCALE in ("0", "none", "None"):
    UPSCALE = None
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "connelly_pony_20s")))
WORK = Path(os.environ.get("WORK_DIR", "/tmp/connelly_pony_20s"))
# Optional Jennifer-Connelly face reference (identity lock). If unset, falls back
# to pure T2V (weaker likeness). Should be a CLEAR FRONTAL FACE image.
I0_REF = os.environ.get("H3_I0_REF", "").strip() or None
if I0_REF:
    I0_REF = str(Path(I0_REF).expanduser().resolve())

# ---------------------------------------------------------------------------
# Identity lock + look (kept in every prompt; same woman throughout)
# ---------------------------------------------------------------------------
LOOK = """integrated_multimodal_description: Photorealistic live-action cinema, 35mm film grain, natural anamorphic lens, 24fps, 180 shutter, soft practical retail lighting of a closed department store at night — dim warm tungsten from a few ceiling spots, cool blue moonlight washing through tall display windows, soft fluorescents on the escalator. No cartoon, no anime, no CGI plastic skin, no game-engine look, no text, no subtitles, no watermark.  Ample natural bare fair skin with visible pore texture, soft subsurface light, delicate natural body hair, no makeup shine.  Motion discipline: EVERY gesture is SLOW and languid, dreamlike and seductive — minimal, held motion, gentle breathing, slow drifting camera. Nothing quick, nothing jerky; the whole mood is unhurried.

Identity lock (same woman in every shot): the face must be the young Jennifer Connelly — luminous green eyes, strikingly fair pale unblemished skin, youthful smooth face with a delicate jaw, full soft lips, long dark chestnut curls falling over her face. She sits astride a life-size mechanical bucking horse (a chrome-and-fiberglass carousel pony under a sign reading SKIP), wearing an open black leather jacket over a bare torso with a long front zipper, and dark tight jeans. Hair, eye colour, skin tone and the leather jacket must stay identical across all shots."""

# Shot 1 — establish her astride the pony, night store, face readable, slow ease
P1 = f"""{LOOK}

ONE continuous medium shot that slowly pushes closer, no cuts.  The young woman sits astride the mechanical pony in the dim closed store, one hand resting on the chrome pommel, the other hanging loosely.  She rolls her hips in a slow, lazy rhythm with the pony's gentle sway, chest rising and falling with a soft breath, and turns her head slowly to look almost over her bare shoulder at camera — a drowsy, knowing half-smile, eyes half-lidded, hair slipping over her shoulder.

[0-1.7s] She is still, riding the pony's slow rock, settling her weight, a strand of dark hair tucked behind her ear — very gradual.
[1.7-3.4s] She lets her eyelashes lower and slowly trails one fingertip up the outside of her thigh, drawing it to rest on the zipper pull; the camera inches in, framing her shoulder and chest.
[3.4-5.17s] She bites her lower lip in a long, held, unhurried motion and pulls the zipper pull DOWN one single slow inch — just a narrow, subtle opening of fair bare skin at the top of her cleavage.  She holds the look, breathing softly.

overall_soundscape: soft mechanical servo hum of the pony, faint distant store ventilation, one slow quiet fwip of the zipper, her soft breath. No music. No dialogue.
non_diegetic_music: N/A"""

# Shot 2 — the langorous full unzip + cleavage (first_frame = shot1 last). TIGHT, SLOW.
P2 = f"""{LOOK}

ONE continuous TIGHT medium-close shot of her bare chest and throat, no cuts, gentle slow drift.  CONTINUATION: she draws the zipper DOWN over the full length of her torso in one extremely slow, deliberate pull; the black leather parts and falls open across her chest, exposing her generous cleavage, the soft inside curves of her breasts, natural fair skin, the lower edge of her collarbone and sternum.  Camera holds close on the skin — every pore, every soft shadow on the bare fair chest visible.

[0-1.7s] Her fingers work the metal pull down with deliberate slowness; leather creaks and the two sides of the jacket open a few centimetres, her breasts settling free inside.
[1.7-3.4s] The jacket edge droops open wider as she gives the smallest shrug — the view lingers on the soft valley between her breasts and the skin of her upper stomach; she presses her breasts gently together with her upper arms.
[3.4-5.17s] Very slowly she tilts her head back, lips parting, and drags her tongue across her lower lip while biting it gently; she looks at camera through lowered lashes.  Almost no motion — just the slow rise and fall of her bare chest.

overall_soundscape: the long slow zipper fwip, soft servo sway, leather creak, her slow deep breathing. No music. No dialogue.
non_diegetic_music: N/A"""

# Shot 3 — leather falls fully open; bosom to hip framing; areolae + fair skin (first_frame = shot2 last)
P3 = f"""{LOOK}

ONE continuous close shot framed LOW on her bare TORSO from just under her bosom down toward her hips, no cuts, the camera drifting down very slowly to reveal soft bare fair skin.  CONTINUATION: the black leather jacket now hangs wide open off her shoulders and elbows; her pale bare torso is fully revealed — the soft underside curves of her breasts with natural rose areolae and nipples plainly visible, the gentle slope of her ribs, the flat soft plane of her stomach, porcelain fair skin with faint freckles and finely visible pore texture, a glisten of sweat along the sternum.

[0-1.7s] The camera moves down her bare torso at a whisper-slow pace, passing the underside of her breasts and the areolae, lingering on the skin of her ribs.
[1.7-3.4s] She inhales slowly, her lower belly rising; the jacket shifts open further at her sides; she drags one hand slowly down her sternum, thumb grazing the inside curve of one breast.
[3.4-5.17s] The camera settles at her waist as she hooks a finger low at her jeans — a tiny, slow tug exposing only the barest top edge of her pubic hair line above the waistband, fair skin taut over her lower belly.  She breathes, held and still.

overall_soundscape: soft servo sway and pneumatic hiss, leather sliding slowly, her slow breathing. No music. No dialogue.
non_diegetic_music: N/A"""

# Shot 4 — slow final: full bare torso, top of pubic hair, sultry held finale (first_frame = shot3 last)
P4 = f"""{LOOK}

ONE continuous slow medium-close shot on her bare torso and face, no cuts, camera holding and barely drifting.  CONTINUATION: the final, unhurried reveal — she reclines back over the pony's rump, the open black leather jacket pooling at her hips; her whole pale bare torso and the clean top line of her dark pubic hair sit visible above her unbuttoned-tight jeans waistband, natural fair skin, the soft underside of her breasts with areolae visible, one knee hooked lazily over the saddle.

[0-1.7s] She arches her back in a long, slow, luxurious stretch, hair spilling, eyes closing then slowly opening to find the camera.
[1.7-3.4s] She hooks a thumb under her jeans waistband and eases it DOWN one slow inch, uncovering the full clean line of her pubic hair and flat lower belly; the pony keeps its gentle mechanical sway beneath her.
[3.4-5.17s] She bites her lower lip in a long held motion and gives the camera a slow, possessive, memorable smile — green eyes locked, fair skin flushed faintly, bare chest rising and falling slowly.  The camera holds on her face and bare torso as she waits, watchful.

overall_soundscape: soft pony-sway mechanism, her slow breathing, a faint store hum. No music. No dialogue.
non_diegetic_music: N/A"""

# ---------------------------------------------------------------------------
# Networking / Comfy helpers (ported from repo)
# ---------------------------------------------------------------------------
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
        return json.loads(r.read().decode()).get("name", name)


def queue(host: str, graph: dict, client_id: str) -> str:
    try:
        r = http_json(f"http://{host}/prompt", {"prompt": graph, "client_id": client_id}, timeout=60)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"queue {host}: {e.read().decode()[:2500]}") from e
    if "error" in r:
        raise RuntimeError(r)
    return r["prompt_id"]


def wait_done(host: str, pid: str, label: str, timeout: int = 7800) -> dict:
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
            print(f"  [{label}] {time.time()-t0:.0f}s ...", flush=True)
            last = time.time()
        time.sleep(4)
    raise TimeoutError(label)


def find_video(entry: dict) -> tuple[str, str, str]:
    for _nid, out in (entry.get("outputs") or {}).items():
        for key, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and any(it.get("filename", "").lower().endswith(e) for e in (".mp4", ".webm", ".mkv")):
                    return it["filename"], it.get("subfolder", ""), it.get("type", "output")
    raise RuntimeError(f"no video: {json.dumps(entry.get('outputs'))[:1500]}")


def download(host: str, fn: str, sub: str, typ: str, dest: Path):
    from urllib.parse import quote
    q = f"filename={quote(fn)}&subfolder={quote(sub)}&type={typ}"
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=600) as r:
        dest.write_bytes(r.read())
    print(f"  saved {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)


def extract_frame(video: Path, out: Path, when: str = "last"):
    out.parent.mkdir(parents=True, exist_ok=True)
    if when == "first":
        cmd = ["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    elif when == "best_face":
        cmd = ["ffmpeg", "-y", "-ss", "0.35", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-sseof", "-0.05", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  frame[{when}] -> {out}", flush=True)


def render(host, label, prompt, seed, prefix, first: str | None, last: str | None) -> Path:
    g = build_enhanced(
        prompt, seed=seed, width=W, height=H, length=FRAMES, steps=STEPS,
        prefix=prefix, first_frame=first or None, last_frame=last or None,
        sage="auto", spectrum=SPECTRUM, upscale=UPSCALE,
    )
    pid = queue(host, g, label)
    print(f"  [{label}] queued {pid} on {host} len={FRAMES} steps={STEPS} first={bool(first)} last={bool(last)}", flush=True)
    entry = wait_done(host, pid, label)
    fn, sub, typ = find_video(entry)
    dest = WORK / f"{label}.mp4"
    download(host, fn, sub, typ, dest)
    return dest


def run_on(host, label, shot, seed, first_ref) -> tuple[Path, Path]:
    """Render a 2-shot chained arm on one box; returns (clip1, clip2)."""
    c1 = render(host, f"{label}_c1", shot[0], seed, f"video/{label}_c1", first_ref, None)
    # chain: next shot's first frame = this clip's last frame (upload to same host)
    first_png = WORK / f"{label}_c1_last.png"
    extract_frame(c1, first_png, "last")
    first_name = upload_image(host, first_png, f"{label}_c1_last.png")
    c2 = render(host, f"{label}_c2", shot[1], seed + 7, f"video/{label}_c2", first_name, None)
    return c1, c2


def run_arms() -> tuple[Path, Path, Path, Path]:
    """Launch arm A (.2) and arm B (.3) in parallel threads."""
    results: dict[str, object] = {}

    # Identity lock: seed BOTH arms with the same face reference so the
    # likeness can't drift between arm A (P1/P2) and arm B (P3/P4).
    head_ref = None
    worker_ref = None
    if I0_REF:
        src = Path(I0_REF)
        if not src.is_file():
            raise FileNotFoundError(f"H3_I0_REF not found: {src}")
        # normalize/scale to a clean square-ish face png (H3 wants ~64 multiple dims)
        norm = WORK / "face_ref.png"
        subprocess.check_call([
            "ffmpeg", "-y", "-i", str(src), "-vf",
            "scale=640:960:force_original_aspect_ratio=decrease,pad=640:960:(ow-iw)/2:(oh-ih)/2",
            "-frames:v", "1", str(norm),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  identity ref prepared -> {norm}", flush=True)
        head_ref = upload_image(HEAD, norm, "face_ref.png")
        worker_ref = upload_image(WORKER, norm, "face_ref.png")
        print("  identity seeded on BOTH hosts", flush=True)

    def run(key, host, label, shots, base_seed, ref):
        try:
            results[key] = run_on(host, label, shots, base_seed, ref)
        except BaseException as e:
            results[key] = e

    a = threading.Thread(target=run, args=("a", HEAD, "shotA", (P1, P2), SEED, head_ref))
    b = threading.Thread(target=run, args=("b", WORKER, "shotB", (P3, P4), SEED + 1, worker_ref))
    t0 = time.time()
    print(f"\n=== dual-H3 parallel: arm A (.2) = P1+P2  |  arm B (.3) = P3+P4 (identity-seeded) ===", flush=True)
    a.start(); b.start(); a.join(); b.join()
    wall = time.time() - t0
    for k in ("a", "b"):
        if isinstance(results[k], BaseException):
            raise results[k]  # type: ignore
    c1, c2 = results["a"]  # type: ignore
    c3, c4 = results["b"]  # type: ignore
    print(f"  dual-arm wall {wall:.0f}s", flush=True)
    return c1, c2, c3, c4


def dur(p: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(p)]) .strip())


def stitch4(a: Path, b: Path, c: Path, d: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    da, db, dc, dd = dur(a), dur(b), dur(c), dur(d)
    # 3 xfades (0.4s) -> 20.7-1.2 = ~19.5s  (near 20s; keep duration honest)
    xf = 0.4
    o1 = max(0.0, da - xf)
    o2 = max(0.0, da + db - 2 * xf)
    o3 = max(0.0, da + db + dc - 3 * xf)
    fc = (
        f"[0:v]fps=24,format=yuv420p,setsar=1[v0];"
        f"[1:v]fps=24,format=yuv420p,setsar=1[v1];"
        f"[2:v]fps=24,format=yuv420p,setsar=1[v2];"
        f"[3:v]fps=24,format=yuv420p,setsar=1[v3];"
        f"[v0][v1]xfade=transition=fade:duration={xf}:offset={o1}[x1];"
        f"[x1][v2]xfade=transition=fade:duration={xf}:offset={o2}[x2];"
        f"[x2][v3]xfade=transition=fade:duration={xf}:offset={o3}[v];"
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0];"
        f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
        f"[2:a]aformat=sample_rates=44100:channel_layouts=stereo[a2];"
        f"[3:a]aformat=sample_rates=44100:channel_layouts=stereo[a3];"
        f"[a0][a1]acrossfade=d={xf}[x1a];"
        f"[x1a][a2]acrossfade=d={xf}[x2a];"
        f"[x2a][a3]acrossfade=d={xf}[a]"
    )
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(a), "-i", str(b), "-i", str(c), "-i", str(d),
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  FINAL {out} ({out.stat().st_size/1e6:.1f} MB)  clips {da:.2f}/{db:.2f}/{dc:.2f}/{dd:.2f}s", flush=True)


def main():
    t_all = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"connelly dual-H3 20s  spectrum={SPECTRUM} upscale={UPSCALE} len={FRAMES} steps={STEPS} seed={SEED}", flush=True)

    for host, name in ((HEAD, ".2"), (WORKER, ".3")):
        try:
            s = http_json(f"http://{host}/system_stats", timeout=6)
            print(f"H3 {name} {host} up  ram_free={s['system']['ram_free']/1e9:.1f}G", flush=True)
        except Exception as e:
            print(f"H3 {name} DOWN: {e}", file=sys.stderr); sys.exit(1)

    # clear any stuck queues
    for host in (HEAD, WORKER):
        try:
            http_json(f"http://{host}/interrupt", {}, timeout=10)
            http_json(f"http://{host}/queue", {"clear": True}, timeout=10)
        except Exception:
            pass

    c1, c2, c3, c4 = run_arms()
    for tag, p in (("01_mount", c1), ("02_unzip", c2), ("03_open", c3), ("04_reveal", c4)):
        shutil.copy2(p, OUT_DIR / f"shot_{tag}.mp4")

    print("\n=== stitching 4 clips -> ~20s ===", flush=True)
    final = OUT_DIR / "connelly_pony_20s.mp4"
    stitch4(c1, c2, c3, c4, final)
    # also a hard-cut concat (no xfade) for review
    lst = WORK / "concat4.txt"
    lst.write_text("\n".join(f"file '{p.resolve()}'" for p in (c1, c2, c3, c4)))
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(OUT_DIR / "connelly_pony_20s_hardcut.mp4"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total = time.time() - t_all
    (OUT_DIR / "TIMING.txt").write_text(
        f"mode=concurrency2-dual-arm\n"
        f"total_wall_s={total:.1f}\n"
        f"len_frames={FRAMES} steps={STEPS}\n"
        f"spectrum={SPECTRUM} upscale={UPSCALE}\n"
        f"note=arm A (.2)=P1/P2 chained; arm B (.3)=P3/P4 chained; 4-way xfade\n"
    )
    print(f"\nDone in {total:.0f}s wall (~{total/60:.1f} min)", flush=True)
    print(f"  continuous: {final}", flush=True)
    print(f"  hardcut:    {OUT_DIR / 'connelly_pony_20s_hardcut.mp4'}", flush=True)
    print(f"  shots:      {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
