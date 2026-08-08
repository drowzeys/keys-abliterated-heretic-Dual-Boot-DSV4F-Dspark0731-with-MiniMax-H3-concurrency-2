#!/usr/bin/env python3
"""Parallel dual-H3 ~20s Jennifer Connelly / Career Opportunities clip — FLF-correct.

Correct architecture (user-guided, mirrors tonyd2wild factory keyframe_dual_flf.py):

  Phase 0 (SEQUENTIAL, single box .2, full quality) — generate the shared KEY
          frames up front and make them CONSISTENT:
            K0 establish on pony   K1 zipper->cleavage   K2 leather open (areolae)
            K3 torso low (pubic-top tease)               K4 final reclined skin+face
          Each K_i is seeded by K_{i-1}'s last frame so identity/skin/pose stay
          consistent across the whole chain. Same box -> no cross-box drift.
          Every seam key is ONE image, used as both A-end and B-start.

  Phase 2 (PARALLEL, concurrency=2) — generate the motion arms from the shared
          keys, one per box, two waves:
            wave1  arm1: K0->K1 (HEAD .2)   ||   arm2: K1->K2 (WORKER .3)
            wave2  arm3: K2->K3 (HEAD .2)   ||   arm4: K3->K4 (WORKER .3)
          All arms are temporally independent given the keys exist, so they run
          on both boxes in parallel. The parallelism is leveraged HERE.

  Phase 3 — HARD-CUT concat (NO xfade): the seams share exact key frames, so a
          straight concat is continuous by construction.

Identity lock via H3_I0_REF (Jennifer Connelly face). Slow seductive motion.
Usage:
  H3_I0_REF=/path/to/face.png python3 connelly_flf_20s.py
  UPSCALE=0 python3 connelly_flf_20s.py
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
KF_FRAMES = int(os.environ.get("H3_LEN_KF", "49"))    # keyframe short ~2s
ARM_FRAMES = int(os.environ.get("H3_LEN", "124"))     # motion arm ~5.17s
STEPS = int(os.environ.get("STEPS", "20"))
KF_STEPS = int(os.environ.get("H3_STEPS_KF", "20"))
SEED = int(os.environ.get("SEED", "77119"))
SPECTRUM = os.environ.get("H3_SPECTRUM", "1") not in ("0", "false", "no")
UPSCALE = os.environ.get("UPSCALE", "RealESRGAN_x2plus.pth")
if UPSCALE in ("0", "none", "None"):
    UPSCALE = None
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "connelly_flf_20s")))
WORK = Path(os.environ.get("WORK_DIR", "/tmp/connelly_flf_20s"))
I0_REF = os.environ.get("H3_I0_REF", "").strip() or None
if I0_REF:
    I0_REF = str(Path(I0_REF).expanduser().resolve())

# ---------------------------------------------------------------------------
# Prompts — identity locked, slow seductive, tight torso framing
# ---------------------------------------------------------------------------
LOOK = """integrated_multimodal_description: photorealistic live-action cinema, 35mm film grain, natural anamorphic lens, 24fps, 180 shutter, soft practical retail lighting of a closed department store at night — dim warm tungsten, cool blue moonlight through tall display windows. No cartoon, no anime, no CGI plastic, no airbrushed or retouched skin, no game-engine look, no text, no subtitles, no watermark. Skin MUST look REAL and alive: natural visible pore texture, fine soft skin folds and creases at the collarbone and waist, gentle subsurface scattering, subtle sheen and delicate natural body hair, faint freckling, no blur-smooth porcelain, every inch of bare body reads as a real woman's skin with weight. Embrace the body's natural curves — the genuine rounding of the breasts, the soft dip and swell of the waist and hips, the gentle curve of the belly; no idealized flat mannequin, a woman with soft natural volume and realistic silhouette. Motion discipline: EVERY gesture SLOW and languid, dreamlike, minimal held motion, gentle breathing, slow drifting camera. Nothing quick or jerky.

Identity lock (same woman in every key and arm): the face and body of a young Jennifer Connelly — luminous green eyes, strikingly fair pale skin, youthful smooth face, delicate jaw, full soft lips, long dark chestnut curls. She sits astride a life-size mechanical bucking horse (chrome-and-fiberglass carousel pony, sign reading SKIP), wearing an open black leather jacket over a bare torso with a long front zipper, and dark tight jeans. Hair, eyes, skin tone and the jacket stay identical everywhere."""

def key_prompt(scene):
    return f"""{LOOK}\n\nKEYFRAME still (near-frozen, minimal motion), high quality: {scene} overall_soundscape: soft servo hum, her slow breath. No music. No dialogue.\nnon_diegetic_music: N/A"""

K_PROMPTS = [
    # K0
    key_prompt("The young woman sits astride the mechanical pony in the dim closed store, one hand on the chrome pommel, hips rocked slightly into the pony's slow sway, head turned to camera with a drowsy knowing half-smile, hair over her shoulder, zipper still closed, black leather jacket open at the collar showing the top of her cleavage."),
    # K1
    key_prompt("Same woman on the pony, now holding the zipper pull drawn halfway down her chest; the leather is parted and open at her cleavage, generous fair skin and the soft inside curve of her breasts beginning to show, her other hand resting on her thigh, lips parted, eyes on camera."),
    # K2
    key_prompt("Same woman on the pony, the black leather jacket now fully open off her shoulders and elbows, bare torso revealed — soft curves of her breasts with natural rose areolae and nipples plainly visible, sternum, flat stomach, porcelain fair skin with faint freckles and visible pore texture; she bites her lower lip, head tipped back slightly, hair tumbling, low angle on her bare torso."),
    # K3
    key_prompt("Same woman on the pony, close framing low on her bare torso from just beneath her breasts down to her hips: the underside of her breasts with areolae, the gentle slope of her ribs, the flat soft plane of her stomach; she hooks a finger low at her jeans tugging the waistband a fraction so the very top edge of her dark pubic hair line shows above the unbuttoned tight jeans; fair skin taut over her lower belly."),
    # K4
    key_prompt("Same young woman, final reveal: she reclines back over the pony's rump, open black leather jacket pooling at her hips, whole pale bare torso and the clean top line of her dark pubic hair visible above her jeans waistband, areolae visible, one knee hooked lazily over the saddle; she bites her lower lip and gives the camera a slow possessive memorable smile, green eyes locked, chest rising with a soft breath."),
]

# Motion arms — each a SLOW continuous short between its two key frames
# Prompts rewritten to H3-native [Shot N] + timestamps structure (technique from
# lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA methodology), H3-optimized wording.
ARM_SCRIPTS = [
    # arm1: K0 -> K1   mount, lazy rock, slow zipper start
    f"""{LOOK}

[Shot 1] Slow continuous medium shot, gentle push-in toward her chest. FIRST FRAME matches the key frame of the woman astride the pony, zipper closed. LAST FRAME matches the key frame of her with the zipper pulled halfway open at her cleavage. Slow, restrained: she rides the pony's lazy rock, one fingertip travels up her thigh to the zipper pull over ~2 seconds, then she draws the zipper down one slow inch over the next ~2 seconds — bare fair skin and the start of her cleavage appear. She bites her lower lip and holds, eyes on camera, breath slow. Pony sways gently throughout. WARDROBE/CONTINUITY: identical leather jacket, jeans, dark chestnut hair, green eyes, fair pore-textured skin in every frame.

timestamps: [0.0-1.7s] settle and ride the rock; [1.7-3.4s] fingertip path to the zipper; [3.4-5.17s] slow one-inch unzip, hold. No cuts.
overall_soundscape: soft servo hum of the pony, faint store ventilation, one slow zipper fwip, her soft long exhale. No speech.
non_diegetic_music: N/A""",
    # arm2: K1 -> K2   langorous full unzip + leather falls open (areolae)
    f"""{LOOK}

[Shot 1] Slow TIGHT medium-close shot on her bare chest and throat, gentle drift, no cuts. FIRST FRAME matches the key of her zipper open at cleavage; LAST FRAME matches the key of the leather fully open with rose areolae visible. Slow: she draws the metal zipper down the rest of her chest over ~2 seconds, leather creaks open; with the smallest shrug the jacket slips off her shoulders to her elbows over the next ~2 seconds, her bare torso fully opening, natural areolae and the soft inside curves of both breasts revealed against fair skin with visible pores and soft folds at the sternum; she tilts her head back slowly, lips parted, tongue dragging across her lower lip, chest rising and falling. WARDROBE/CONTINUITY: identical jacket skin hair throughout.

timestamps: [0.0-1.7s] slow rest-of-zipper pull; [1.7-3.4s] jacket shrugs open at shoulders; [3.4-5.17s] head back, lip bite, held breath. No cuts.
overall_soundscape: long slow zipper fwip, soft servo sway, leather creak, her slow deep breathing.
non_diegetic_music: N/A""",
    # arm3: K2 -> K3   camera drifts down torso bosom->pubic tease
    f"""{LOOK}

[Shot 1] Close shot framed LOW on her bare torso from beneath the bosom down to the hips, whisper-slow downward camera drift, no cuts. FIRST FRAME matches the key of the leather open with areolae; LAST FRAME matches the key of the low torso with a finger hooking her jeans at the top of the pubic hair line. Slow: the camera drifts down her bare torso past the underside of the breasts and areolae, lingering on the ribs over ~2 seconds; she inhales slowly, lower belly rising, a hand dragging down her sternum with the thumb grazing one breast; the camera settles at her waist as she hooks a finger low at her jeans and gives a slow tug exposing the barest top edge of her dark pubic hair line above the waistband, fair skin taut, soft natural creases at the waist.

timestamps: [0.0-1.7s] camera drift down torso; [1.7-3.4s] slow inhale, hand down sternum; [3.4-5.17s] waist settle, slow jeans tug, pubic-top edge. No cuts.
overall_soundscape: soft servo sway, leather slide, her slow breathing.
non_diegetic_music: N/A""",
    # arm4: K3 -> K4   slow final skin reveal + possessive look
    f"""{LOOK}

[Shot 1] Slow medium-close shot on her bare torso and face, camera holding and barely drifting, no cuts. FIRST FRAME matches the key of the low torso pubic-top tease; LAST FRAME matches the key of her reclined with the full top line of her pubic hair and possessive smile. Slow: she arches back in a long luxurious stretch over ~2 seconds, hair spilling, eyes opening slowly to find the camera; she eases her jeans waistband down one slow inch over the next ~2 seconds, uncovering the clean top line of her pubic hair and flat lower belly; she bites her lower lip in a long held motion and gives a slow possessive memorable smile, green eyes locked, fair skin faintly flushed, bare chest rising and falling slowly as she holds the camera. WARDROBE/CONTINUITY: identical jacket, jeans, hair, eyes throughout.

timestamps: [0.0-1.7s] slow stretch, eyes find camera; [1.7-3.4s] waistband eased down, pubic-hair clean line; [3.4-5.17s] lip bite, possessive smile, hold.
overall_soundscape: soft pony-sway mechanism, her slow breathing, faint store hum.
non_diegetic_music: N/A""",
]

# ---------------------------------------------------------------------------
def http_json(url, data=None, timeout=120):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def upload_image(host, path, name):
    import mimetypes, uuid
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(), raw,
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(f"http://{host}/upload/image", data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode()).get("name", name)

def queue(host, graph, client_id):
    try:
        r = http_json(f"http://{host}/prompt", {"prompt": graph, "client_id": client_id}, timeout=60)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"queue {host}: {e.read().decode()[:2500]}") from e
    if "error" in r:
        raise RuntimeError(r)
    return r["prompt_id"]

def wait_done(host, pid, label, timeout=7800):
    t0 = time.time(); last = 0
    while time.time() - t0 < timeout:
        try:
            hist = http_json(f"http://{host}/history/{pid}", timeout=30)
        except Exception:
            time.sleep(5); continue
        if pid in hist:
            entry = hist[pid]; st = entry.get("status", {}); s = st.get("status_str", "")
            if s == "success" or entry.get("outputs"):
                print(f"  [{label}] SUCCESS {time.time()-t0:.0f}s", flush=True); return entry
            if s == "error":
                for m in st.get("messages") or []:
                    if isinstance(m, list) and m and m[0] == "execution_error":
                        raise RuntimeError(f"{label}: {m[1].get('exception_message','')[:1200]}")
                raise RuntimeError(f"{label}: {json.dumps(st)[:1200]}")
        if time.time() - last > 20:
            print(f"  [{label}] {time.time()-t0:.0f}s ...", flush=True); last = time.time()
        time.sleep(4)
    raise TimeoutError(label)

def find_video(entry):
    for _nid, out in (entry.get("outputs") or {}).items():
        for key, items in out.items():
            if not isinstance(items, list): continue
            for it in items:
                if isinstance(it, dict) and any(it.get("filename", "").lower().endswith(e) for e in (".mp4", ".webm", ".mkv")):
                    return it["filename"], it.get("subfolder", ""), it.get("type", "output")
    raise RuntimeError(f"no video: {json.dumps(entry.get('outputs'))[:1500]}")

def download(host, fn, sub, typ, dest):
    from urllib.parse import quote
    q = f"filename={quote(fn)}&subfolder={quote(sub)}&type={typ}"
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=600) as r:
        dest.write_bytes(r.read())
    print(f"  saved {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)

def extract_frame(video, out, when="last"):
    out.parent.mkdir(parents=True, exist_ok=True)
    if when == "first":
        cmd = ["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-sseof", "-0.05", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  frame[{when}] -> {out}", flush=True)

def render(host, label, prompt, seed, prefix, length, steps, first, last, upscale=UPSCALE):
    g = build_enhanced(
        prompt, seed=seed, width=W, height=H, length=length, steps=steps,
        prefix=prefix, first_frame=first or None, last_frame=last or None,
        sage="auto", spectrum=SPECTRUM, upscale=upscale,
    )
    pid = queue(host, g, label)
    print(f"  [{label}] queued on {host} len={length} steps={steps} first={bool(first)} last={bool(last)}", flush=True)
    entry = wait_done(host, pid, label)
    fn, sub, typ = find_video(entry)
    dest = WORK / f"{label}.mp4"
    download(host, fn, sub, typ, dest)
    return dest

def prepare_ref():
    """Normalize H3_I0_REF -> face png and upload to both boxes (identity lock)."""
    if not I0_REF:
        print("  [warn] H3_I0_REF not set — pure T2V identity (weaker likeness)", flush=True)
        return None
    src = Path(I0_REF)
    if not src.is_file():
        raise FileNotFoundError(f"H3_I0_REF not found: {src}")
    norm = WORK / "face_ref.png"
    subprocess.check_call(["ffmpeg", "-y", "-i", str(src), "-vf",
        "scale=640:960:force_original_aspect_ratio=decrease,pad=640:960:(ow-iw)/2:(oh-ih)/2",
        "-frames:v", "1", str(norm)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hn = upload_image(HEAD, norm, "face_ref.png")
    upload_image(WORKER, norm, "face_ref.png")
    print(f"  identity ref {norm} seeded on BOTH hosts", flush=True)
    return hn

def run_parallel(jobs):
    """jobs: list[(host,label,callable)]. Run all concurrently (<=2 expected)."""
    results = {}
    def run(idx, host, label, fn):
        try:
            results[idx] = fn()
        except BaseException as e:
            results[idx] = e
    threads = [threading.Thread(target=run, args=(i, h, l, f)) for i, (h, l, f) in enumerate(jobs)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    for i in range(len(jobs)):
        if isinstance(results.get(i), BaseException):
            raise results[i]
    print(f"  parallel wave wall {wall:.0f}s", flush=True)
    return [results[i] for i in range(len(jobs))]

def main():
    t_all = time.time()
    WORK.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FLF 20s  spectrum={SPECTRUM} upscale={UPSCALE} kf={KF_FRAMES}/arm={ARM_FRAMES} steps={STEPS}", flush=True)
    for host, name in ((HEAD, ".2"), (WORKER, ".3")):
        try:
            s = http_json(f"http://{host}/system_stats", timeout=6)
            print(f"H3 {name} {host} up ram_free={s['system']['ram_free']/1e9:.1f}G", flush=True)
        except Exception as e:
            print(f"H3 {name} DOWN: {e}", file=sys.stderr); sys.exit(1)
    for host in (HEAD, WORKER):
        try:
            http_json(f"http://{host}/interrupt", {}, timeout=10); http_json(f"http://{host}/queue", {"clear": True}, timeout=10)
        except Exception: pass

    face = prepare_ref()

    # ---- PHASE 0: sequential, consistent keyframes on HEAD only ----
    print("\n=== PHASE 0: sequential quality keyframes K0..K4 on", HEAD, "===", flush=True)
    keys = []
    prev_key_name = face
    for i, kp in enumerate(K_PROMPTS):
        label = f"K{i}"
        vid = render(HEAD, label, kp, SEED + i, f"video/kf_{label}", KF_FRAMES, KF_STEPS,
                     prev_key_name, None, None)
        # the KEY image is the last frame of this short; keep it for seams
        key_png = WORK / f"K{i}.png"
        extract_frame(vid, key_png, "last")
        shutil.copy2(key_png, OUT_DIR / f"key_{label}.png")
        key_name = upload_image(HEAD, key_png, f"K{i}.png")
        upload_image(WORKER, key_png, f"K{i}.png")  # both boxes have every key
        keys.append(key_name)
        prev_key_name = key_name
    print(f"  phase0 wall {time.time()-t_all:.0f}s — 5 consistent keyframes", flush=True)

    # ---- PHASE 2: parallel motion arms (2 waves x 2 boxes) ----
    print("\n=== PHASE 2: parallel motion arms (concurrency=2) ===", flush=True)
    arms = []
    # arm n uses key[n-1] (first) -> key[n] (last); key names known on correct host
    for i in range(4):
        host = HEAD if i % 2 == 0 else WORKER
        fn = i + 1
        arms.append((host, f"arm{fn}",
            lambda i=i, host=host, fn=fn: render(
                host, f"arm{fn}", ARM_SCRIPTS[i], SEED + 100 + i, f"video/arm{fn}",
                ARM_FRAMES, STEPS, keys[i], keys[i + 1], UPSCALE)))
    # wave1: arm1 (HEAD) || arm2 (WORKER)   wave2: arm3 (HEAD) || arm4 (WORKER)
    wave1 = run_parallel(arms[0:2])
    wave2 = run_parallel(arms[2:4])
    for label, p in zip(["arm1", "arm2", "arm3", "arm4"], wave1 + wave2):
        shutil.copy2(p, OUT_DIR / f"{label}.mp4")

    # ---- PHASE 3: HARD-CUT concat (NO xfade — seams are exact key matches) ----
    print("\n=== PHASE 3: hard-cut concat (no xfade) ===", flush=True)
    lst = WORK / "concat_all.txt"
    allp = wave1 + wave2
    lst.write_text("\n".join(f"file '{p.resolve()}'" for p in allp))
    final = OUT_DIR / "connelly_flf_20s.mp4"
    subprocess.check_call(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(final)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  FINAL {final}", flush=True)
    total = time.time() - t_all
    (OUT_DIR / "TIMING.txt").write_text(
        f"mode=FLF-consistent-keys parallel-arms no-xfade\n"
        f"total_wall_s={total:.1f}\nkeyframes=5 arms=4 arm_frames={ARM_FRAMES}\n"
        f"note=keys K0..K4 sequential on .2, consistent; arms ran parallel .2|.3; hard-cut concat\n")
    dur = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",str(final)], text=True).strip())
    print(f"\nDone in {total:.0f}s wall (~{total/60:.1f} min)  final duration={dur:.2f}s", flush=True)
    print(f"  {final}", flush=True)

if __name__ == "__main__":
    main()
