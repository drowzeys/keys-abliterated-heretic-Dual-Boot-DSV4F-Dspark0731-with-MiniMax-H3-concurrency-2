#!/usr/bin/env python3
"""Build the full heretic-enhanced MiniMax H3 API graph.

Patch chain (matches ~/comfy/h3-enhanced-fullstack.json):
  UNET → [optional MiniMaxH3TurboLoRA] → PathchSageAttentionKJ → SolAttnPatch
       → SpectrumApplyMiniMaxH3 → H3FirstBlockCache → BasicGuider / SamplerCustomAdvanced
  TE: heretic NVFP4
  Post: optional RealESRGAN x2/x4 on decoded frames before CreateVideo

Turbo (optional): QrusherZA ComfyUI-fixed MiniMax-H3 Turbo LoRA (orig larryvrh)
  + MiniMaxH3TurboSampler → 4–8 steps instead of ~20 (~3–5× sample speedup).
  Requires custom_nodes/ComfyUI-MiniMax-H3-Turbo and a turbo .safetensors in models/loras/.

Co-tenancy defaults: 864x480, length=124 (~5.17s), steps=20 (or 6 with turbo).
"""
from __future__ import annotations

from typing import Any

# Default pruned ComfyUI-fixed weights (matches pruned_int8 base DiT)
DEFAULT_TURBO_LORA = "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"


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
    sage: str = "auto",  # or "disabled"
    sol_tau: float = 1.3,
    spectrum: bool = True,  # requires ComfyUI >=0.30.1 (time_shift_slope in ldm/minimax)
    fbc_threshold: float = 0.08,
    fbc_start_step: int = 3,
    # turbo LoRA (4–8 step few-step sampling)
    turbo: bool = False,
    turbo_lora: str = DEFAULT_TURBO_LORA,
    turbo_strength: float = 1.0,
    turbo_low_vram: bool = False,  # True = merge (softer, less peak VRAM under co-tenancy)
    # post
    upscale: str | None = "RealESRGAN_x2plus.pth",  # None to skip
    fps: float = 24.0,
) -> dict[str, Any]:
    """Return a ComfyUI API-format prompt graph."""

    # model patch chain — turbo LoRA sits right after UNET (official order)
    g: dict[str, Any] = {
        "6": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
    }
    model_out = ["6", 0]
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

    g["50"] = {
        "class_type": "PathchSageAttentionKJ",
        "inputs": {
            "model": model_out,
            "sage_attention": sage,
            "allow_compile": False,
        },
    }
    g["7"] = {
        "class_type": "SolAttnPatch",
        "inputs": {
            "model": ["50", 0],
            "tau": sol_tau,
            "start_percent": 0.2,
            "end_percent": 0.9,
            "min_tokens": 4096,
            "int8_qk": True,
            "sink_conditioning": "exact_kv_and_rows",
            "morton": False,
            "morton_curve": "2d_frame",
            "verbose": False,
            # TMA: peak VRAM spike; leave off under DS4 co-tenancy
            "use_tma": False,
        },
    }

    # Spectrum needs native MiniMax helpers (time_shift_slope). On stock Comfy
    # 0.30.x that helper is missing → crash. Only wire Spectrum when requested.
    if spectrum:
        g["51"] = {
            "class_type": "SpectrumApplyMiniMaxH3",
            "inputs": {
                "model": ["7", 0],
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
        fbc_in = ["51", 0]
    else:
        fbc_in = ["7", 0]

    # FBC on turbo: denser early steps matter less with 4–8 total; still useful
    g.update({
        "8": {
            "class_type": "H3FirstBlockCache",
            "inputs": {
                "model": fbc_in,
                "threshold": fbc_threshold,
                "start_step": fbc_start_step if not turbo else 1,
                "end_dense_steps": 2 if not turbo else 1,
                "max_consecutive_skips": 2 if not turbo else 1,
            },
        },
        "13": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": te,
                "type": "minimax",
                "device": "default",
            },
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
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["8", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "16": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["8", 0], "conditioning": ["104", 0]},
        },
    })

    # Sampler: turbo dual-schedule vs stock res_multistep
    if turbo:
        g["17"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
    else:
        g["17"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        }

    g.update({
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16", 0],
                "sampler": ["17", 0],
                "sigmas": ["9", 0],
                "latent_image": ["104", 1],
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["14", 0], "vae": ["11", 0]},
        },
        "23": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["14", 0], "vae": ["24", 0]},
        },
    })

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

    # frames for CreateVideo: raw decode or upscaled
    image_src = ["10", 0]
    if upscale:
        g["60"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale},
        }
        g["61"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["60", 0], "image": ["10", 0]},
        }
        image_src = ["61", 0]

    g["91"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": image_src, "audio": ["23", 0], "fps": fps},
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
    print(json.dumps(build_enhanced(prompt), indent=2))
