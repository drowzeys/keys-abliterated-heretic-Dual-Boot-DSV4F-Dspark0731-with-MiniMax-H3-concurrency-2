#!/usr/bin/env python3
"""Generic dual-H3 multishot FLF engine (Hermes / keyspark architecture).

Correct seamless parallel pipeline for MiniMax-H3 concurrency=2:

  Phase 0  SEQUENTIAL quality KEYFRAMES K0..KN on one box (HEAD)
           — consistent identity / skin / lighting
           — each key can be re-anchored to H3_I0_REF (stops likeness drift)
           — OR chained so key_i first_frame = key_{i-1} last (pose continuity)

  Phase 1  PARALLEL motion arms (fleet concurrency=2):
           arm_i: first_frame=K_i  last_frame=K_{i+1}
           waves of 2 across HEAD(.2) ‖ WORKER(.3)
           — previous segment's LAST image IS the next segment's FIRST image
           — seams are exact shared pixels, not approximate

  Phase 2  HARD-CUT concat (NO xfade) — continuous by construction

This is the workflow Hermes refined for dual-instance H3. Prefer it over
naive crossfade dual or independent T2V arms.

Usage (library):
  from multishot_flf import MultishotConfig, run_multishot
  run_multishot(cfg)

Usage (CLI demo — HP dragon 2-arm story):
  PYTHONPATH=... python3 multishot_flf.py
  H3_I0_REF=/path/to/face.png python3 multishot_flf.py

Env:
  H3_HEAD / H3_WORKER   default 10.100.10.2:8188 / .3:8188
  H3_W H3_H H3_LEN H3_LEN_KF H3_STEPS H3_STEPS_KF
  H3_SPECTRUM H3_UPSCALE (0 to skip) SEED
  H3_I0_REF             face/identity image for every key (recommended)
  H3_KEY_CHAIN=1        also chain keys (first=prev last) after face seed
  H3_KEY_MODE=face|chain|both   default face if ref else chain
  OUT_DIR WORK_DIR

Credit: dual-H3 co-tenancy factory by tonyd2wild/ds4-h3-video-gen-factory;
        FLF multishot parallel architecture refined with Hermes on keyspark.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_graph import build_enhanced  # noqa: E402

# ---------------------------------------------------------------------------
# Natural skin / anti-plastic look (prepended to every prompt by default)
# ---------------------------------------------------------------------------
NATURAL_SKIN = """Skin & material truth (critical): real human (or creature) surface only —
visible pore texture, fine natural creases and folds, soft subsurface scatter,
subtle natural sheen (not plastic gloss), delicate peach-fuzz / body hair where
appropriate, faint freckling or natural unevenness. NO airbrushed porcelain,
NO CGI plastic, NO beauty-filter blur, NO mannequin smoothness, NO game-engine
shader look. Prefer 35mm film grain and practical light over digital perfection."""

HEAD = os.environ.get("H3_HEAD", "10.100.10.2:8188")
WORKER = os.environ.get("H3_WORKER", "10.100.10.3:8188")


@dataclass
class MultishotConfig:
    """N keys → N-1 motion arms; parallel waves of 2 on dual H3."""

    key_prompts: Sequence[str]
    arm_prompts: Sequence[str]
    # optional shared look bible prepended once
    look: str = ""
    # natural skin block
    natural_skin: bool = True
    # outputs
    out_dir: Path = field(default_factory=lambda: Path.home() / "Videos" / "multishot_flf")
    work_dir: Path = field(default_factory=lambda: Path("/tmp/multishot_flf"))
    final_name: str = "multishot_flf.mp4"
    # render
    width: int = int(os.environ.get("H3_W", "864"))
    height: int = int(os.environ.get("H3_H", "480"))
    kf_frames: int = int(os.environ.get("H3_LEN_KF", "49"))
    arm_frames: int = int(os.environ.get("H3_LEN", "124"))
    steps: int = int(os.environ.get("H3_STEPS", os.environ.get("STEPS", "20")))
    kf_steps: int = int(os.environ.get("H3_STEPS_KF", "20"))
    seed: int = int(os.environ.get("SEED", "42042"))
    spectrum: bool = os.environ.get("H3_SPECTRUM", "1") not in ("0", "false", "no")
    upscale: str | None = os.environ.get("H3_UPSCALE", os.environ.get("UPSCALE", "RealESRGAN_x2plus.pth"))
    # identity
    i0_ref: str | None = None
    # key generation: "face" re-anchors every key to ref; "chain" uses prev last;
    # "both" = first_frame=face (or prev) with chain for pose (uses prev last as first if no face)
    key_mode: str = "auto"  # auto | face | chain | both
    # hosts
    head: str = HEAD
    worker: str = WORKER
    # hard-cut only (seamless by shared keys)
    hardcut: bool = True
    # H3 Turbo LoRA (few-step) — QrusherZA ComfyUI-fixed / larryvrh original
    turbo: bool = os.environ.get("H3_TURBO", "0") not in ("0", "false", "no", "")
    turbo_lora: str = os.environ.get(
        "H3_TURBO_LORA", "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"
    )
    turbo_strength: float = float(os.environ.get("H3_TURBO_STRENGTH", "1.0"))
    turbo_low_vram: bool = os.environ.get("H3_TURBO_LOW_VRAM", "0") not in ("0", "false", "no", "")
    # Dual-sampler quality (ANe5s HF discussion #21) — rough ckpt850 then refine ckpt500
    # https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/21
    dual_turbo: bool = os.environ.get("H3_DUAL_TURBO", "0") not in ("0", "false", "no", "")
    turbo_lora_rough: str = os.environ.get(
        "H3_TURBO_LORA_ROUGH", "minimax_h3_turbo_4step_ckpt850.safetensors"
    )
    turbo_steps_rough: int = int(os.environ.get("H3_TURBO_STEPS_ROUGH", "6"))
    turbo_strength_rough: float = float(os.environ.get("H3_TURBO_STRENGTH_ROUGH", "1.0"))
    turbo_lora_refine: str = os.environ.get(
        "H3_TURBO_LORA_REFINE", "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"
    )
    turbo_steps_refine: int = int(os.environ.get("H3_TURBO_STEPS_REFINE", "8"))
    turbo_strength_refine: float = float(os.environ.get("H3_TURBO_STRENGTH_REFINE", "0.7"))
    # Motion Context — true audio/motion continuation (NikoDemon80)
    # https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
    # Forces sequential arms + Spectrum off.
    motion_context: bool = os.environ.get("H3_MOTION_CONTEXT", "0") not in ("0", "false", "no", "")
    context_length: int = int(os.environ.get("H3_CONTEXT_LENGTH", "22"))
    audio_context_length: int = int(os.environ.get("H3_AUDIO_CONTEXT_LENGTH", "22"))

    def __post_init__(self):
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        if self.upscale in ("0", "none", "None", ""):
            self.upscale = None
        n_keys = len(self.key_prompts)
        n_arms = len(self.arm_prompts)
        if n_keys < 2:
            raise ValueError("need at least 2 key prompts (start + end)")
        if n_arms != n_keys - 1:
            raise ValueError(f"arm_prompts must be n_keys-1 ({n_keys - 1}), got {n_arms}")
        if self.i0_ref:
            self.i0_ref = str(Path(self.i0_ref).expanduser().resolve())
        if self.key_mode == "auto":
            self.key_mode = "face" if self.i0_ref else "chain"
        if self.dual_turbo:
            self.turbo = True
            # dual path uses rough+refine step counts, not self.steps
        elif self.turbo and self.steps >= 16 and "H3_STEPS" not in os.environ and "STEPS" not in os.environ:
            self.steps = 6
        if self.turbo and self.kf_steps >= 16 and "H3_STEPS_KF" not in os.environ:
            self.kf_steps = 6
        # Motion context + dual turbo: Spectrum must stay off (set in graph)
        if self.motion_context:
            self.spectrum = False

    def full_prompt(self, body: str) -> str:
        parts = []
        if self.look:
            parts.append(self.look.strip())
        if self.natural_skin:
            parts.append(NATURAL_SKIN)
        parts.append(body.strip())
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HTTP / Comfy helpers
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
    req = urllib.request.Request(
        f"http://{host}/upload/image", data=b"".join(parts),
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


def wait_done(host: str, pid: str, label: str, timeout: int = 9500) -> dict:
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
                print(f"  [{label}] SUCCESS {time.time() - t0:.0f}s", flush=True)
                return entry
            if s == "error":
                for m in st.get("messages") or []:
                    if isinstance(m, list) and m and m[0] == "execution_error":
                        raise RuntimeError(f"{label}: {m[1].get('exception_message', '')[:1200]}")
                raise RuntimeError(f"{label}: {json.dumps(st)[:1200]}")
        if time.time() - last > 20:
            print(f"  [{label}] {time.time() - t0:.0f}s …", flush=True)
            last = time.time()
        time.sleep(4)
    raise TimeoutError(label)


def find_video(entry: dict) -> tuple[str, str, str]:
    for _nid, out in (entry.get("outputs") or {}).items():
        for _k, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and any(
                    it.get("filename", "").lower().endswith(e) for e in (".mp4", ".webm", ".mkv")
                ):
                    return it["filename"], it.get("subfolder", ""), it.get("type", "output")
    raise RuntimeError(f"no video: {json.dumps(entry.get('outputs'))[:1500]}")


def download(host: str, fn: str, sub: str, typ: str, dest: Path):
    from urllib.parse import quote
    q = f"filename={quote(fn)}&subfolder={quote(sub)}&type={typ}"
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=600) as r:
        dest.write_bytes(r.read())
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


def extract_frame(video: Path, out: Path, when: str = "last"):
    out.parent.mkdir(parents=True, exist_ok=True)
    if when == "first":
        cmd = ["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    elif when == "best_face":
        cmd = ["ffmpeg", "-y", "-ss", "0.35", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-sseof", "-0.05", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  frame[{when}] → {out}", flush=True)


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
    *,
    width: int,
    height: int,
    spectrum: bool,
    upscale: str | None,
    work: Path,
    turbo: bool = False,
    turbo_lora: str = "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors",
    turbo_strength: float = 1.0,
    turbo_low_vram: bool = False,
    dual_turbo: bool = False,
    turbo_lora_rough: str = "minimax_h3_turbo_4step_ckpt850.safetensors",
    turbo_strength_rough: float = 1.0,
    turbo_steps_rough: int = 6,
    turbo_lora_refine: str = "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors",
    turbo_strength_refine: float = 0.7,
    turbo_steps_refine: int = 8,
    motion_context: bool = False,
    prev_video: str | None = None,
    context_length: int = 22,
    audio_context_length: int = 22,
    motion_clip_index: int = 0,
) -> Path:
    g = build_enhanced(
        prompt, seed=seed, width=width, height=height, length=length, steps=steps,
        prefix=prefix, first_frame=first or None, last_frame=last or None,
        sage="auto", spectrum=spectrum, upscale=upscale,
        turbo=turbo, turbo_lora=turbo_lora, turbo_strength=turbo_strength,
        turbo_low_vram=turbo_low_vram,
        dual_turbo=dual_turbo,
        turbo_lora_rough=turbo_lora_rough,
        turbo_strength_rough=turbo_strength_rough,
        turbo_steps_rough=turbo_steps_rough,
        turbo_lora_refine=turbo_lora_refine,
        turbo_strength_refine=turbo_strength_refine,
        turbo_steps_refine=turbo_steps_refine,
        motion_context=motion_context,
        prev_video=prev_video,
        context_length=context_length,
        audio_context_length=audio_context_length,
        motion_clip_index=motion_clip_index,
    )
    pid = queue(host, g, label)
    step_info = (
        f"dual={turbo_steps_rough}+{turbo_steps_refine}"
        if dual_turbo
        else f"steps={steps}"
    )
    print(
        f"  [{label}] queued {pid} on {host}  "
        f"len={length} {step_info} first={bool(first)} last={bool(last)} "
        f"mc={bool(prev_video)} dual_turbo={dual_turbo}",
        flush=True,
    )
    entry = wait_done(host, pid, label)
    fn, sub, typ = find_video(entry)
    dest = work / f"{label}.mp4"
    download(host, fn, sub, typ, dest)
    return dest


def run_parallel(jobs: Sequence[Callable[[], Path]]) -> list[Path]:
    results: dict[int, Path | BaseException] = {}

    def run(i, fn):
        try:
            results[i] = fn()
        except BaseException as ex:
            results[i] = ex

    threads = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(jobs)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    out: list[Path] = []
    for i in range(len(jobs)):
        r = results.get(i)
        if isinstance(r, BaseException):
            raise r
        out.append(r)  # type: ignore[arg-type]
    print(f"  parallel wave wall {wall:.0f}s", flush=True)
    return out


def prepare_face_ref(cfg: MultishotConfig) -> str | None:
    if not cfg.i0_ref:
        print("  [warn] no H3_I0_REF — weaker identity lock", flush=True)
        return None
    src = Path(cfg.i0_ref)
    if not src.is_file():
        raise FileNotFoundError(f"H3_I0_REF not found: {src}")
    norm = cfg.work_dir / "face_ref.png"
    if src.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv"):
        subprocess.check_call(
            ["ffmpeg", "-y", "-ss", "0.5", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(norm)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(norm)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    name = "face_ref.png"
    upload_image(cfg.head, norm, name)
    upload_image(cfg.worker, norm, name)
    print(f"  identity ref → both hosts as {name}", flush=True)
    return name


def hardcut_concat(clips: Sequence[Path], final: Path, work: Path):
    final.parent.mkdir(parents=True, exist_ok=True)
    lst = work / "concat_hardcut.txt"
    lst.write_text("\n".join(f"file '{p.resolve()}'" for p in clips) + "\n")
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"  FINAL hard-cut {final}", flush=True)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
def run_multishot(cfg: MultishotConfig) -> Path:
    t_all = time.time()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    n_keys = len(cfg.key_prompts)
    n_arms = len(cfg.arm_prompts)
    print(
        f"mode=FLF-MULTISHOT  keys={n_keys} arms={n_arms}  "
        f"key_mode={cfg.key_mode} natural_skin={cfg.natural_skin}\n"
        f"  spectrum={cfg.spectrum} upscale={cfg.upscale}  "
        f"kf={cfg.kf_frames}x{cfg.kf_steps} arm={cfg.arm_frames}x{cfg.steps}\n"
        f"  turbo={cfg.turbo} dual_turbo={cfg.dual_turbo} "
        f"motion_context={cfg.motion_context}\n"
        f"  dual: rough={cfg.turbo_lora_rough}@{cfg.turbo_strength_rough} "
        f"x{cfg.turbo_steps_rough} → refine={cfg.turbo_lora_refine}"
        f"@{cfg.turbo_strength_refine} x{cfg.turbo_steps_refine}\n"
        f"  hardcut={cfg.hardcut}  (seam = shared key pixels, no xfade)",
        flush=True,
    )

    for host, name in ((cfg.head, ".2"), (cfg.worker, ".3")):
        try:
            s = http_json(f"http://{host}/system_stats", timeout=6)
            print(f"H3 {name} {host} ram_free={s['system']['ram_free'] / 1e9:.1f}G", flush=True)
        except Exception as e:
            print(f"H3 {name} DOWN: {e}", file=sys.stderr)
            sys.exit(1)

    # clear queues
    for host in (cfg.head, cfg.worker):
        try:
            http_json(f"http://{host}/interrupt", {}, timeout=10)
            http_json(f"http://{host}/queue", {"clear": True}, timeout=10)
        except Exception:
            pass

    face = prepare_face_ref(cfg)

    # --- Phase 0: sequential keys ---
    print(
        f"\n=== PHASE 0: sequential keyframes K0..K{n_keys - 1} on {cfg.head} ===\n"
        f"  mode={cfg.key_mode}  (face re-anchor prevents identity drift)",
        flush=True,
    )
    t0 = time.time()
    key_names: list[str] = []  # uploaded basenames on both hosts
    prev_last_name: str | None = None

    for i, kp in enumerate(cfg.key_prompts):
        label = f"K{i}"
        prompt = cfg.full_prompt(kp)
        # choose first_frame for this key
        first: str | None = None
        if cfg.key_mode == "face":
            first = face
        elif cfg.key_mode == "chain":
            first = prev_last_name
        elif cfg.key_mode == "both":
            # prefer face for identity; fall back to chain
            first = face or prev_last_name
        # first key always uses face if available
        if i == 0 and face:
            first = face

        # Keys: single turbo (fast) — dual-turbo reserved for full motion arms
        vid = render(
            cfg.head, label, prompt, cfg.seed + i, f"video/kf_{label}",
            cfg.kf_frames, cfg.kf_steps, first, None,
            width=cfg.width, height=cfg.height, spectrum=cfg.spectrum and not cfg.motion_context,
            upscale=None,
            work=cfg.work_dir,
            turbo=cfg.turbo and not cfg.dual_turbo,
            turbo_lora=cfg.turbo_lora,
            turbo_strength=cfg.turbo_strength,
            turbo_low_vram=cfg.turbo_low_vram,
            dual_turbo=False,
        )
        key_png = cfg.work_dir / f"K{i}.png"
        extract_frame(vid, key_png, "last")
        shutil.copy2(key_png, cfg.out_dir / f"key_K{i}.png")
        kn = upload_image(cfg.head, key_png, f"K{i}.png")
        upload_image(cfg.worker, key_png, f"K{i}.png")
        key_names.append(kn)
        prev_last_name = kn

    phase0 = time.time() - t0
    print(f"  phase0 wall {phase0:.0f}s  — {n_keys} shared keys on both boxes", flush=True)

    # --- Phase 1: FLF arms ---
    # Motion Context needs previous arm audio/latent → sequential on one host.
    # Dual turbo runs full ANe5s two-stage on each arm; two boxes still parallel
    # when motion_context is off.
    sequential = cfg.motion_context
    print(
        f"\n=== PHASE 1: FLF arms "
        f"({'SEQUENTIAL motion-context' if sequential else f'PARALLEL waves×2'}) ===\n"
        f"  arm_i: first=K_i last=K_{{i+1}}  dual_turbo={cfg.dual_turbo}",
        flush=True,
    )
    t1 = time.time()

    def arm_kwargs(i: int, prev_vid_name: str | None):
        # Apply Motion Context from arm2 onward (needs previous clip).
        # Always assign motion_clip_index when MC enabled so arm1 still *saves*
        # its AV latent for arm2 to load (fixes missing h3_context/ folder).
        return dict(
            width=cfg.width,
            height=cfg.height,
            spectrum=cfg.spectrum and not cfg.motion_context,
            upscale=cfg.upscale,
            work=cfg.work_dir,
            turbo=cfg.turbo,
            turbo_lora=cfg.turbo_lora,
            turbo_strength=cfg.turbo_strength,
            turbo_low_vram=cfg.turbo_low_vram,
            dual_turbo=cfg.dual_turbo,
            turbo_lora_rough=cfg.turbo_lora_rough,
            turbo_strength_rough=cfg.turbo_strength_rough,
            turbo_steps_rough=cfg.turbo_steps_rough,
            turbo_lora_refine=cfg.turbo_lora_refine,
            turbo_strength_refine=cfg.turbo_strength_refine,
            turbo_steps_refine=cfg.turbo_steps_refine,
            motion_context=cfg.motion_context and i > 0 and bool(prev_vid_name),
            prev_video=prev_vid_name,
            context_length=cfg.context_length,
            audio_context_length=cfg.audio_context_length,
            motion_clip_index=(i + 1) if cfg.motion_context else 0,
        )

    all_arms: list[Path] = []
    if sequential:
        # stay on HEAD so Motion Context latents + prev video stay on one host
        host = cfg.head
        host_ip = host.split(":")[0]
        # ComfyUI paths on the remote box (keyspark dual-boot layout)
        comfy_root = os.environ.get(
            "H3_COMFY_ROOT", "/home/keyspark/h3-cotenancy/ComfyUI"
        )
        input_dir = f"{comfy_root}/input"
        out_ctx = f"{comfy_root}/output/h3_context"
        # ensure latent folder exists before any Load
        try:
            subprocess.check_call(
                ["ssh", f"keyspark@{host_ip}", f"mkdir -p {out_ctx} {input_dir}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  [warn] mkdir comfy dirs: {e}", flush=True)

        prev_host_path: str | None = None
        for i in range(n_arms):
            fn = i + 1
            ap = cfg.full_prompt(cfg.arm_prompts[i])
            if i > 0:
                # scp previous arm mp4 to a stable absolute path for VHS_LoadVideoPath
                prev_path = all_arms[-1]
                prev_host_path = f"{input_dir}/mc_prev_arm{i}.mp4"
                subprocess.check_call(
                    ["scp", "-q", str(prev_path), f"keyspark@{host_ip}:{prev_host_path}"],
                )
                print(f"  mc prev → {host_ip}:{prev_host_path}", flush=True)
            p = render(
                host, f"arm{fn}", ap, cfg.seed + 100 + i, f"video/arm{fn}",
                cfg.arm_frames, cfg.steps, key_names[i], key_names[i + 1],
                **arm_kwargs(i, prev_host_path),
            )
            all_arms.append(p)
            shutil.copy2(p, cfg.out_dir / f"arm{fn}.mp4")
            # confirm SaveLatent wrote clip slot (arm index 1-based)
            if cfg.motion_context:
                try:
                    listing = subprocess.check_output(
                        ["ssh", f"keyspark@{host_ip}", f"ls -la {out_ctx}/ 2>/dev/null || true"],
                        text=True,
                    )
                    print(f"  h3_context after arm{fn}:\n{listing}", flush=True)
                except Exception:
                    pass
    else:
        arm_jobs: list[tuple[str, str, Callable[[], Path]]] = []
        for i in range(n_arms):
            host = cfg.head if i % 2 == 0 else cfg.worker
            fn = i + 1
            ap = cfg.full_prompt(cfg.arm_prompts[i])

            def make_job(i=i, host=host, fn=fn, ap=ap):
                return lambda: render(
                    host, f"arm{fn}", ap, cfg.seed + 100 + i, f"video/arm{fn}",
                    cfg.arm_frames, cfg.steps, key_names[i], key_names[i + 1],
                    **arm_kwargs(i, None),
                )

            arm_jobs.append((host, f"arm{fn}", make_job()))

        for w in range((n_arms + 1) // 2):
            wave = arm_jobs[w * 2 : w * 2 + 2]
            print(f"  -- wave {w + 1}: {[lab for _, lab, _ in wave]} --", flush=True)
            all_arms += run_parallel([fn for _, _, fn in wave])
        for i, p in enumerate(all_arms):
            shutil.copy2(p, cfg.out_dir / f"arm{i + 1}.mp4")

    phase1 = time.time() - t1
    print(f"  phase1 wall {phase1:.0f}s", flush=True)

    # --- Phase 2: hard-cut stitch ---
    print("\n=== PHASE 2: hard-cut concat (no xfade — shared key seams) ===", flush=True)
    final = cfg.out_dir / cfg.final_name
    hardcut_concat(all_arms, final, cfg.work_dir)

    total = time.time() - t_all
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(final)],
            text=True,
        ).strip())
    except Exception:
        dur = -1.0

    timing = (
        f"mode=FLF-MULTISHOT\n"
        f"total_wall_s={total:.1f}\n"
        f"phase0_keys_s={phase0:.1f}\n"
        f"phase1_parallel_arms_s={phase1:.1f}\n"
        f"keys={n_keys} arms={n_arms}\n"
        f"key_mode={cfg.key_mode} natural_skin={cfg.natural_skin}\n"
        f"spectrum={cfg.spectrum} upscale={cfg.upscale}\n"
        f"turbo={cfg.turbo} turbo_lora={cfg.turbo_lora if cfg.turbo else ''} "
        f"steps={cfg.steps} kf_steps={cfg.kf_steps}\n"
        f"i0_ref={cfg.i0_ref or ''}\n"
        f"hardcut=True\n"
        f"duration_s={dur:.2f}\n"
        f"note=prev last frame == next first frame (shared key PNG); concurrency=2 waves\n"
        f"credit=tonyd2wild factory + Hermes FLF + larryvrh/QrusherZA H3 Turbo\n"
    )
    (cfg.out_dir / "TIMING.txt").write_text(timing)
    print(f"\nDone in {total:.0f}s wall (~{total / 60:.1f} min)  duration={dur:.2f}s", flush=True)
    print(f"  {final}", flush=True)
    print(f"  keys: {cfg.out_dir}/key_K*.png  arms: arm*.mp4", flush=True)
    return final


# ---------------------------------------------------------------------------
# Default CLI story: HP → dragon ~10s (2 arms) using FLF multishot
# ---------------------------------------------------------------------------
def default_hp_dragon_cfg() -> MultishotConfig:
    look = """integrated_multimodal_description: Photorealistic live-action cinema, 8K IMAX detail,
natural anamorphic lens, 24fps, 180° shutter. Cinematography like Alfonso Cuarón x Roger Deakins.
Practical light only: grey Highland daylight through tall Gothic stone windows, warm floating
candles, cool stone bounce, soft volumetric dust. No cartoon, no anime, no text, no watermark.

Identity lock: same teen wizard when human — messy black hair, round wire spectacles, lightning
scar, black robe, red-and-gold Gryffindor scarf. Same emerald-and-obsidian Western dragon scale
pattern, horn shape, body proportions whenever the dragon appears."""

    keys = [
        """KEYFRAME / living portrait (minimal motion): teenage wizard alone in Hogwarts-like
great hall, medium shot center frame. Hands with golden sparks under the skin; quiet fear and
wonder; spectacles on; scar visible; REAL skin pores and fabric weave. NO dragon in frame yet.""",
        """KEYFRAME still-energy: fully transformed mid-sized Western dragon fills the same hall,
wings half-unfurled, head toward tall open Gothic window, emerald-obsidian scales with real scale
micro-detail (not plastic), coiled to leap.""",
        """KEYFRAME living aerial still: same dragon banks in hover over Hogwarts-like castle —
turrets, lake, misty Highlands. Wings fully extended, correct membrane stretch.""",
    ]
    arms = [
        """ONE continuous take, no cuts. FIRST FRAME = human wizard key. LAST FRAME = dragon at window key.
[0–1.5s] Boy, golden sparks, robe lifts, candles gutter — real skin pores, no plastic.
[1.5–3.5s] Visceral transform: spine, scales, snout, wings with weight and anatomy.
[3.5–5.2s] Fully formed dragon fills hall, head to open window.
overall_soundscape: fireplace, morphing flesh/scale, dragon inhale. No music. No dialogue.""",
        """ONE continuous take. FIRST FRAME = dragon at window. LAST FRAME = dragon over castle.
[0–1.5s] Coils and launches through Gothic window.
[1.5–3.5s] Exterior bank into hover over roofs and lake.
[3.5–5.2s] Hero aerial hold circling keep.
overall_soundscape: wind, wing beats, distant water. No music. No dialogue.""",
    ]
    return MultishotConfig(
        key_prompts=keys,
        arm_prompts=arms,
        look=look,
        natural_skin=True,
        out_dir=Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "hp_dragon_flf"))),
        work_dir=Path(os.environ.get("WORK_DIR", "/tmp/hp_dragon_flf")),
        final_name="harry_potter_dragon_flf_10s.mp4",
        i0_ref=os.environ.get("H3_I0_REF", "").strip() or None,
        key_mode=os.environ.get("H3_KEY_MODE", "auto"),
    )


def main():
    cfg = default_hp_dragon_cfg()
    run_multishot(cfg)


if __name__ == "__main__":
    main()
