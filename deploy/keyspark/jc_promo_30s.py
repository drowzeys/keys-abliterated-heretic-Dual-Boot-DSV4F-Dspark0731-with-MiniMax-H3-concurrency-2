#!/usr/bin/env python3
"""~30s Jennifer Connelly promo — FLF multishot dual-H3 (portrait AR).

Story: green-eyed beauty look-back, pitches keys + tonyd2wild ablit/heretic
DeepSeek-V4F DSpark + MiniMax H3 Turbo multishot parallel on two DGX Sparks,
asks audience to get it for her now, blows a kiss.

Architecture: multishot_flf (keys sequential → arms parallel first/last → hard-cut).
Aspect: portrait matching reference (~3:4). Identity: H3_I0_REF image.

Usage:
  H3_I0_REF=~/Videos/jc_promo_30s/I0_ref.jpg H3_TURBO=1 \\
    python3 jc_promo_30s.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multishot_flf import MultishotConfig, run_multishot  # noqa: E402

# Portrait ~3:4 (reference 526×701 → 576×768)
W = int(os.environ.get("H3_W", "576"))
H = int(os.environ.get("H3_H", "768"))

LOOK = f"""integrated_multimodal_description: Photorealistic live-action cinema, 35mm film grain,
natural anamorphic lens, 24fps, 180° shutter. Closed department store / Target-like retail aisle
at night — soft fluorescent wash, red-and-white circular logo blur in deep background bokeh,
shallow depth of field. No cartoon, no anime, no CGI plastic, no beauty-filter porcelain, no text
graphics burned into the frame, no subtitles, no watermark, no captions on screen.

Identity lock (same woman every frame — match the reference photograph exactly): young Jennifer
Connelly — luminous GREEN eyes (catchlight, vivid and memorable), dark chestnut-brown hair long
and wavy over one shoulder, fair warm skin with real visible pores and soft subsurface scatter,
full soft lips with natural rose tint, delicate jaw, subtle elegant drop earrings, white ribbed
tank top, dark high-waist bottoms. Face and figure must stay identical across all keys and arms.

Skin truth: real pore texture, fine natural creases, gentle sheen, no airbrush. Camera: portrait
vertical framing {W}x{H}, medium close-up to medium shot, elegant handheld breath only.
Motion: slow, languid, seductive confidence — never jerky."""

# 7 keys → 6 arms × ~5.17s ≈ 31s
KEY_PROMPTS = [
    # K0 — establish, green eyes
    f"""{LOOK}
KEYFRAME still (near-frozen, high quality): medium close-up portrait matching the reference —
she faces three-quarter toward camera in the store aisle, white tank top, long dark hair, a soft
knowing half-smile. Her GREEN eyes are the hero — sharp catchlights, vivid, beautiful. Shallow
DOF, red circular logo soft in background. overall_soundscape: quiet store hum, soft breath.
No music. No dialogue. non_diegetic_music: N/A""",
    # K1 — look back over shoulder
    f"""{LOOK}
KEYFRAME still: she has turned, looking BACK over her bare shoulder at camera with a warm
devastating smile; green eyes locked on the lens; hair cascading; white tank top; same store
bokeh. Beautiful, unforgettable look-back pose. overall_soundscape: soft store hum, her breath.
No music. No dialogue. non_diegetic_music: N/A""",
    # K2 — speaking to audience (pitch start)
    f"""{LOOK}
KEYFRAME still: facing camera again, medium close-up, lips slightly parted mid-speech, green eyes
earnest and bright, one hand raised gently as if explaining something important to her audience.
Same identity and wardrobe. overall_soundscape: quiet room tone. No music.
non_diegetic_music: N/A""",
    # K3 — emphatic pitch
    f"""{LOOK}
KEYFRAME still: closer framing, green eyes intense and playful, slight lean toward camera,
confident half-smile as if she just named the stack she wants — dual DGX Sparks energy in her
expression, not text on screen. overall_soundscape: quiet room tone. No music.
non_diegetic_music: N/A""",
    # K4 — "get it for me now"
    f"""{LOOK}
KEYFRAME still: pleading-gorgeous look, brows soft, green eyes wide and hopeful, hands near her
heart / clasped gently, leaning in — the "please get it for me now" expression. overall_soundscape:
quiet room tone. No music. non_diegetic_music: N/A""",
    # K5 — wind-up to kiss
    f"""{LOOK}
KEYFRAME still: she lifts one hand toward her lips, fingers poised to blow a kiss, green eyes
mischievous and warm, soft smile. overall_soundscape: quiet room tone. No music.
non_diegetic_music: N/A""",
    # K6 — kiss blown
    f"""{LOOK}
KEYFRAME still: she has just blown a kiss toward camera — hand open toward lens, lips pursed then
soft smile, green eyes sparkling, hair frame her face. Magical, affectionate, final hero hold.
overall_soundscape: soft breath, quiet store hum. No music. non_diegetic_music: N/A""",
]

ARM_PROMPTS = [
    # arm1 K0→K1 look-back
    f"""{LOOK}
[Shot 1] ONE continuous slow medium-close portrait. FIRST FRAME matches the front three-quarter
key with green eyes; LAST FRAME matches the look-back-over-shoulder key. Slow: she holds the
camera with her green eyes, then turns gracefully, looking back over her shoulder with a warm
beautiful smile, hair sliding, catchlights in her eyes the whole time.
timestamps: [0-1.7s] hold eyes; [1.7-3.4s] slow turn; [3.4-5.17s] look-back smile held.
overall_soundscape: quiet store fluorescent hum, soft fabric, her slow breath. No music. No dialogue.
non_diegetic_music: N/A""",
    # arm2 K1→K2 start speaking
    f"""{LOOK}
[Shot 1] ONE continuous slow shot. FIRST FRAME = look-back smile; LAST FRAME = facing camera
mid-speech. She turns back to camera, green eyes bright, and begins speaking warmly to her
audience with natural lip motion (clear speech audio).
She says, in a soft seductive American voice: "Hey… I need you to hear this."
timestamps: [0-1.7s] turn front; [1.7-3.4s] start speaking; [3.4-5.17s] hold earnest look.
overall_soundscape: her clear spoken line, quiet store hum. No music score.
non_diegetic_music: N/A""",
    # arm3 K2→K3 stack pitch
    f"""{LOOK}
[Shot 1] ONE continuous medium close-up, gentle push-in. FIRST FRAME = mid-speech key; LAST FRAME
= emphatic confident pitch key. Natural lip-sync speech, green eyes locked on camera.
She says clearly: "Keys and tonyd2wild's abliterated heretic DeepSeek V4 Flash with DSpark —
plus MiniMax H3 Turbo multishot running in parallel on two DGX Sparks. That is what I want."
timestamps: [0-1.5s] set up; [1.5-4.0s] full stack pitch with lip sync; [4.0-5.17s] confident smile.
overall_soundscape: her clear enthusiastic spoken lines only, soft room tone. No music.
non_diegetic_music: N/A""",
    # arm4 K3→K4 get it now
    f"""{LOOK}
[Shot 1] ONE continuous closer portrait, barely drifting. FIRST FRAME = confident pitch; LAST
FRAME = pleading beautiful "get it for me now" look. Lip-sync speech, green eyes soft and urgent.
She says: "Parallel processing. Multishot. Seamless. I want the audience to get it for me —
right now."
timestamps: [0-1.7s] lean in; [1.7-3.8s] spoken plea; [3.8-5.17s] hopeful green-eyed hold.
overall_soundscape: her clear spoken lines, soft breath. No music.
non_diegetic_music: N/A""",
    # arm5 K4→K5 wind-up kiss
    f"""{LOOK}
[Shot 1] ONE continuous medium close-up. FIRST FRAME = pleading look; LAST FRAME = hand at lips
ready to blow a kiss. She smiles mischievously, raises her hand slowly to her lips, green eyes
sparkling at camera. Optional soft line: "For me?"
timestamps: [0-1.7s] smile; [1.7-3.4s] hand rises to lips; [3.4-5.17s] poised kiss blow.
overall_soundscape: soft breath, quiet store hum, tiny playful whisper. No music.
non_diegetic_music: N/A""",
    # arm6 K5→K6 blow kiss
    f"""{LOOK}
[Shot 1] ONE continuous hero close-up. FIRST FRAME = hand at lips; LAST FRAME = kiss blown toward
camera with open hand and sparkling green eyes. Slow: she blows a kiss to her audience — lips
purse, soft kiss sound, hand opens toward the lens, then a warm unforgettable smile held.
timestamps: [0-1.5s] prepare; [1.5-3.2s] blow the kiss to camera; [3.2-5.17s] smile hold, eyes.
overall_soundscape: soft kiss sound, her happy breath, quiet store hum. No music. No dialogue after kiss.
non_diegetic_music: N/A""",
]


def main():
    ref = os.environ.get("H3_I0_REF", "").strip()
    if not ref:
        cand = Path.home() / "Videos" / "jc_promo_30s" / "I0_ref.jpg"
        if cand.is_file():
            ref = str(cand)
    if not ref:
        print("ERROR: set H3_I0_REF to the Jennifer reference image", file=sys.stderr)
        sys.exit(1)

    # Defaults for HQ re-render: dual turbo + motion context (override with =0)
    if "H3_DUAL_TURBO" not in os.environ:
        os.environ["H3_DUAL_TURBO"] = "1"
    if "H3_MOTION_CONTEXT" not in os.environ:
        os.environ["H3_MOTION_CONTEXT"] = "1"
    if "H3_TURBO" not in os.environ:
        os.environ["H3_TURBO"] = "1"

    def _env_bool(k: str, default: str = "0") -> bool:
        return os.environ.get(k, default) not in ("0", "false", "no", "")

    dual = _env_bool("H3_DUAL_TURBO", "1")
    mctx = _env_bool("H3_MOTION_CONTEXT", "1")
    turbo = _env_bool("H3_TURBO", "1") or dual

    cfg = MultishotConfig(
        key_prompts=KEY_PROMPTS,
        arm_prompts=ARM_PROMPTS,
        look="",  # already embedded in each prompt
        natural_skin=True,
        out_dir=Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "jc_promo_30s_mc"))),
        work_dir=Path(os.environ.get("WORK_DIR", "/tmp/jc_promo_30s_mc")),
        final_name="jc_promo_ablit_h3_dual_turbo_mc_30s.mp4",
        width=W,
        height=H,
        i0_ref=ref,
        key_mode=os.environ.get("H3_KEY_MODE", "face"),
        hardcut=True,
        turbo=turbo,
        dual_turbo=dual,
        motion_context=mctx,
        spectrum=not mctx,  # Motion Context requires Spectrum off
        turbo_lora_rough=os.environ.get(
            "H3_TURBO_LORA_ROUGH", "minimax_h3_turbo_4step_ckpt850.safetensors"
        ),
        turbo_strength_rough=float(os.environ.get("H3_TURBO_STRENGTH_ROUGH", "1.0")),
        turbo_steps_rough=int(os.environ.get("H3_TURBO_STEPS_ROUGH", "6")),
        turbo_lora_refine=os.environ.get(
            "H3_TURBO_LORA_REFINE", "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"
        ),
        turbo_strength_refine=float(os.environ.get("H3_TURBO_STRENGTH_REFINE", "0.7")),
        turbo_steps_refine=int(os.environ.get("H3_TURBO_STEPS_REFINE", "8")),
        context_length=int(os.environ.get("H3_CONTEXT_LENGTH", "22")),
        audio_context_length=int(os.environ.get("H3_AUDIO_CONTEXT_LENGTH", "22")),
    )
    print(
        f"JC promo 30s  {W}x{H} portrait  dual_turbo={cfg.dual_turbo} "
        f"motion_context={cfg.motion_context} turbo={cfg.turbo}  ref={ref}",
        flush=True,
    )
    run_multishot(cfg)


if __name__ == "__main__":
    main()
