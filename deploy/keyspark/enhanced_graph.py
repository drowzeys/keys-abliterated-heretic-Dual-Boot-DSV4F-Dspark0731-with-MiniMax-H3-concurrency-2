#!/usr/bin/env python3
"""Build the full heretic-enhanced MiniMax H3 API graph.

Patch chain:
  UNET → [Turbo LoRA(s)] → Sage → SolAttn → [Spectrum] → FBC
       → Guider / SamplerCustomAdvanced
  TE: heretic NVFP4
  Post: optional RealESRGAN x2

Turbo modes
-----------
* single turbo (H3_TURBO=1): one LoRA + MiniMaxH3TurboSampler, 4–8 steps
* dual turbo (H3_DUAL_TURBO=1) — @ANe5s HF discussion #21 quality recipe:
    Stage 1 rough:  ckpt850 strength 1.0, high-sigma steps (default 6)
    Stage 2 refine: ckpt500 strength 0.7, low-sigma steps (default 8)
    SplitSigmas between stages; TurboSampler both times
  Dual-box usage: each arm runs the full dual-stage graph; two Sparks run
  different arms in parallel (or stage1/stage2 pipelined — see dual_turbo_pipeline).

Motion Context (optional)
-------------------------
NikoDemon80/ComfyUI-H3-Motion-Context — true audio+motion continuation.
Forces Spectrum off. Prefer sequential arms.

Credits: tonyd2wild factory · larryvrh Turbo · QrusherZA ComfyUI pruned weights
         · @ANe5s dual-sampler recipe · NikoDemon80 Motion Context
"""
from __future__ import annotations

from typing import Any

# Stage-2 refine (QrusherZA ComfyUI-fixed pruned — matches pruned_int8 DiT)
DEFAULT_TURBO_LORA = "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"
# Stage-1 rough (ANe5s: non-EMA ckpt850 — high-variance layout/motion)
DEFAULT_TURBO_LORA_ROUGH = "minimax_h3_turbo_4step_ckpt850.safetensors"


def _sol_attn(model_ref: list, sol_tau: float) -> dict[str, Any]:
    return {
        "class_type": "SolAttnPatch",
        "inputs": {
            "model": model_ref,
            "tau": sol_tau,
            "start_percent": 0.2,
            "end_percent": 0.9,
            "min_tokens": 4096,
            "int8_qk": True,
            "sink_conditioning": "exact_kv_and_rows",
            "morton": False,
            "morton_curve": "2d_frame",
            "verbose": False,
            "use_tma": False,
        },
    }


def _attach_speed_chain(
    g: dict[str, Any],
    *,
    model_in: list,
    sage: str,
    sol_tau: float,
    spectrum: bool,
    fbc_threshold: float,
    fbc_start_step: int,
    fbc_end_dense: int,
    fbc_max_skip: int,
    id_prefix: str,
) -> list:
    """UNET/LoRA → Sage → Sol → [Spectrum] → FBC. Returns FBC model ref [id, 0]."""
    sage_id = f"{id_prefix}sage"
    sol_id = f"{id_prefix}sol"
    spec_id = f"{id_prefix}spec"
    fbc_id = f"{id_prefix}fbc"

    g[sage_id] = {
        "class_type": "PathchSageAttentionKJ",
        "inputs": {
            "model": model_in,
            "sage_attention": sage,
            "allow_compile": False,
        },
    }
    g[sol_id] = _sol_attn([sage_id, 0], sol_tau)
    if spectrum:
        g[spec_id] = {
            "class_type": "SpectrumApplyMiniMaxH3",
            "inputs": {
                "model": [sol_id, 0],
                "enabled": True,
                "blend_weight": 0.5,
                "degree": 4,
                "ridge_lambda": 0.1,
                "window_size": 2.0,
                "flex_window": 0.75,
                "warmup_steps": 5,
                "tail_actual_steps": 1,
                "max_history": 8,
                "debug": False,
                "history_storage": "system_ram",
            },
        }
        fbc_in = [spec_id, 0]
    else:
        fbc_in = [sol_id, 0]
    g[fbc_id] = {
        "class_type": "H3FirstBlockCache",
        "inputs": {
            "model": fbc_in,
            "threshold": fbc_threshold,
            "start_step": fbc_start_step,
            "end_dense_steps": fbc_end_dense,
            "max_consecutive_skips": fbc_max_skip,
        },
    }
    return [fbc_id, 0]


def build_enhanced(
    prompt: str,
    *,
    seed: int = 42,
    width: int = 864,
    height: int = 480,
    length: int = 124,
    steps: int = 20,
    prefix: str = "video/H3_enhanced",
    first_frame: str | None = None,
    last_frame: str | None = None,
    unet: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    te: str = "H3/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    # accel
    sage: str = "auto",
    sol_tau: float = 1.3,
    spectrum: bool = True,
    fbc_threshold: float = 0.08,
    fbc_start_step: int = 3,
    # single turbo
    turbo: bool = False,
    turbo_lora: str = DEFAULT_TURBO_LORA,
    turbo_strength: float = 1.0,
    turbo_low_vram: bool = False,
    # dual turbo — ANe5s discussion #21
    # https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/21
    dual_turbo: bool = False,
    turbo_lora_rough: str = DEFAULT_TURBO_LORA_ROUGH,  # stage1 ckpt850 non-EMA
    turbo_strength_rough: float = 1.0,
    turbo_steps_rough: int = 6,  # 5–7 recommended
    turbo_lora_refine: str = DEFAULT_TURBO_LORA,  # stage2 ckpt500
    turbo_strength_refine: float = 0.7,  # ANe5s: >0.7 too denoise-y
    turbo_steps_refine: int = 8,  # 7–8 sweet spot
    # motion context
    motion_context: bool = False,
    prev_video: str | None = None,
    context_length: int = 22,
    audio_context_length: int = 22,
    audio_mode: str = "timeline",
    encode_mode: str = "video",
    anchor_mode: str = "head",
    motion_clip_index: int = 0,
    motion_latent_prefix: str = "h3_context/clip",
    # post
    upscale: str | None = "RealESRGAN_x2plus.pth",
    fps: float = 24.0,
) -> dict[str, Any]:
    """Return a ComfyUI API-format prompt graph."""

    if dual_turbo:
        turbo = True  # implies turbo sampler + few-step schedule
    if motion_context and spectrum:
        spectrum = False  # pinned rows break Spectrum forecasting

    g: dict[str, Any] = {
        "6": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "13": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": te, "type": "minimax", "device": "default"},
        },
        "11": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "24": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "104": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["13", 0],
                "vae": ["11", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }

    if first_frame:
        g["105"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_frame, "upload": "image"},
        }
        g["104"]["inputs"]["first_frame"] = ["105", 0]
    if last_frame:
        g["106"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_frame, "upload": "image"},
        }
        g["104"]["inputs"]["last_frame"] = ["106", 0]

    # --- conditioning (optional motion context) ---
    cond_src: list = ["104", 0]
    trim_frames_src = None
    use_mc = bool(motion_context and prev_video)
    if use_mc:
        g["80"] = {"class_type": "LoadVideo", "inputs": {"file": prev_video}}
        g["81"] = {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["80", 0]},
        }
        mc_inputs: dict[str, Any] = {
            "conditioning": ["104", 0],
            "vae": ["11", 0],
            "latent": ["104", 1],
            "context_frames": ["81", 0],
            "context_length": int(context_length),
            "encode_mode": encode_mode,
            "anchor_mode": anchor_mode,
            "crop": "disabled",
            "audio_context_length": int(audio_context_length),
            "audio_mode": audio_mode,
            "audio_vae": ["24", 0],
            "context_audio": ["81", 1],
        }
        if motion_clip_index > 1:
            g["82"] = {
                "class_type": "MiniMaxH3MotionContextLoadLatent",
                "inputs": {
                    "latent_path": motion_latent_prefix.rsplit("/", 1)[0]
                    if "/" in motion_latent_prefix
                    else "h3_context",
                    "clip_index": int(motion_clip_index - 1),
                },
            }
            mc_inputs["context_latent"] = ["82", 0]
        g["83"] = {
            "class_type": "MiniMaxH3MotionContext",
            "inputs": mc_inputs,
        }
        cond_src = ["83", 0]
        trim_frames_src = ["83", 1]

    # --- model + sampler path ---
    if dual_turbo:
        # ANe5s two-stage: rough ckpt850 @1.0 then refine ckpt500 @0.7
        total_steps = int(turbo_steps_rough) + int(turbo_steps_refine)
        g["70a"] = {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["6", 0],
                "lora_name": turbo_lora_rough,
                "strength": float(turbo_strength_rough),
                "low_vram": bool(turbo_low_vram),
            },
        }
        g["70b"] = {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["6", 0],
                "lora_name": turbo_lora_refine,
                "strength": float(turbo_strength_refine),
                "low_vram": bool(turbo_low_vram),
            },
        }
        model_a = _attach_speed_chain(
            g,
            model_in=["70a", 0],
            sage=sage,
            sol_tau=sol_tau,
            spectrum=spectrum,
            fbc_threshold=fbc_threshold,
            fbc_start_step=1,
            fbc_end_dense=1,
            fbc_max_skip=1,
            id_prefix="a_",
        )
        model_b = _attach_speed_chain(
            g,
            model_in=["70b", 0],
            sage=sage,
            sol_tau=sol_tau,
            spectrum=spectrum,
            fbc_threshold=fbc_threshold,
            fbc_start_step=1,
            fbc_end_dense=1,
            fbc_max_skip=1,
            id_prefix="b_",
        )
        # scheduler on rough model; split into high/low sigma halves
        g["9"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": model_a,
                "scheduler": "simple",
                "steps": total_steps,
                "denoise": 1.0,
            },
        }
        g["90"] = {
            "class_type": "SplitSigmas",
            "inputs": {"sigmas": ["9", 0], "step": int(turbo_steps_rough)},
        }
        g["17"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
        g["16a"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": model_a, "conditioning": cond_src},
        }
        g["16b"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": model_b, "conditioning": cond_src},
        }
        # Stage 1 — rough fit (high sigmas)
        g["14a"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16a", 0],
                "sampler": ["17", 0],
                "sigmas": ["90", 0],  # high_sigmas
                "latent_image": ["104", 1],
            },
        }
        # Stage 2 — refine (low sigmas) from stage-1 latent
        g["14"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16b", 0],
                "sampler": ["17", 0],
                "sigmas": ["90", 1],  # low_sigmas
                "latent_image": ["14a", 0],
            },
        }
        final_latent = ["14", 0]
    else:
        # single path
        model_out: list = ["6", 0]
        if turbo:
            g["70"] = {
                "class_type": "MiniMaxH3TurboLoRA",
                "inputs": {
                    "model": model_out,
                    "lora_name": turbo_lora,
                    "strength": float(turbo_strength),
                    "low_vram": bool(turbo_low_vram),
                },
            }
            model_out = ["70", 0]
        model_fbc = _attach_speed_chain(
            g,
            model_in=model_out,
            sage=sage,
            sol_tau=sol_tau,
            spectrum=spectrum,
            fbc_threshold=fbc_threshold,
            fbc_start_step=fbc_start_step if not turbo else 1,
            fbc_end_dense=2 if not turbo else 1,
            fbc_max_skip=2 if not turbo else 1,
            id_prefix="",
        )
        # remap default FBC id used elsewhere: chain uses "fbc" when id_prefix ""
        # _attach with id_prefix "" → sage, sol, fbc nodes "sage","sol","fbc" — wait
        # id_prefix "" gives "sage", "sol", "fbc" 
        g["9"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": model_fbc,
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        }
        if turbo:
            g["17"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
        else:
            g["17"] = {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "res_multistep"},
            }
        g["16"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": model_fbc, "conditioning": cond_src},
        }
        g["14"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16", 0],
                "sampler": ["17", 0],
                "sigmas": ["9", 0],
                "latent_image": ["104", 1],
            },
        }
        final_latent = ["14", 0]

    g["10"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": final_latent, "vae": ["11", 0]},
    }
    g["23"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": final_latent, "vae": ["24", 0]},
    }

    if motion_context and motion_clip_index > 0:
        g["84"] = {
            "class_type": "MiniMaxH3MotionContextSaveLatent",
            "inputs": {
                "latent": final_latent,
                "filename_prefix": motion_latent_prefix,
                "clip_index": int(motion_clip_index),
            },
        }

    image_src: list = ["10", 0]
    audio_src: list = ["23", 0]
    if use_mc and trim_frames_src is not None:
        g["85"] = {
            "class_type": "MiniMaxH3MotionContextTrim",
            "inputs": {
                "images": ["10", 0],
                "trim_frames": trim_frames_src,
                "audio": ["23", 0],
                "fps": float(fps),
                "match_tail": True,
            },
        }
        image_src = ["85", 0]
        audio_src = ["85", 1]

    if upscale:
        g["60"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale},
        }
        g["61"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["60", 0], "image": image_src},
        }
        image_src = ["61", 0]

    g["91"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": image_src, "audio": audio_src, "fps": fps},
    }
    g["92"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["91", 0],
            "filename_prefix": prefix,
            "format": "auto",
            "codec": "auto",
        },
    }
    return g


if __name__ == "__main__":
    import json
    import sys

    prompt = sys.argv[1] if len(sys.argv) > 1 else "test prompt"
    mode = sys.argv[2] if len(sys.argv) > 2 else "stock"
    if mode == "dual":
        print(json.dumps(build_enhanced(prompt, dual_turbo=True, spectrum=False), indent=2))
    elif mode == "turbo":
        print(json.dumps(build_enhanced(prompt, turbo=True, steps=6), indent=2))
    else:
        print(json.dumps(build_enhanced(prompt), indent=2))
