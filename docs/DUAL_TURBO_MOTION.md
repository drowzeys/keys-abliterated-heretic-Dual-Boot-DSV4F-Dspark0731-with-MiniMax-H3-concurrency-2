# Dual Turbo sampling + Motion Context (high quality multishot)

## 1. Dual-sampler Turbo quality ([@ANe5s discussion #21](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/21))

Two-stage few-step sampling keeps Turbo speed while reducing artifacts:

| Stage | LoRA | Strength | Steps | Role |
|-------|------|----------|------:|------|
| **1 Rough** | `minimax_h3_turbo_4step_ckpt850.safetensors` (**non-EMA**) | **1.0** | **5–7** (default 6) | Layout, physics, motion (high-variance σ) |
| **2 Refine** | `minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors` | **0.7** | **7–8** (default 8) | Detail / de-blur (low-variance σ) |

**Do not** use EMA ckpt850 for stage 1 (ghosting). Strength refine **0.7** is a sharp threshold per ANe5s.

### Graph (per arm, on one H3)

```
UNET ─┬─ TurboLoRA ckpt850@1.0 → Sage→Sol→FBC → GuiderA → Sampler (high σ)
      └─ TurboLoRA ckpt500@0.7 → Sage→Sol→FBC → GuiderB → Sampler (low σ, latent from stage1)
BasicScheduler(total=rough+refine) → SplitSigmas(step=rough)
```

### Dual DGX Spark usage

| Mode | What each box does |
|------|--------------------|
| **Parallel arms** (`H3_DUAL_TURBO=1`, no motion context) | Each arm runs **full dual-stage** on one box; `.2` and `.3` take alternating arms → ~½ wall for motion |
| **Motion-context chain** (`H3_MOTION_CONTEXT=1`) | Arms **sequential** on one box so audio latent continues; dual-stage still runs per arm |

True “stage1 on .2 / stage2 on .3” pipelining is possible later via latent scp; the default dual-stage graph already uses both LoRAs per job and both Sparks for **arm** parallelism.

```bash
# high-quality dual turbo multishot
H3_DUAL_TURBO=1 bash deploy/keyspark/run_quality_parallel.sh

# dual turbo + continuous audio (sequential arms)
H3_DUAL_TURBO=1 H3_MOTION_CONTEXT=1 bash deploy/keyspark/run_quality_parallel.sh
```

Cite **@ANe5s** if you publish results using this recipe.

---

## 2. Motion Context (audio continuity) — proper keyspark adaptation

**Repo:** [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)

Pins previous clip’s last **22** frames + audio onto the next clip’s timeline so the model **continues the same waveform**, not a sound-alike.

### API graph wiring (implemented in `enhanced_graph.py`)

```
arm N-1:  Sampler → SaveLatent(clip_index=N-1) → decode → SaveVideo
arm N:    VHS_LoadVideoPath(abs prev mp4)
            → IMAGE + AUDIO
          MiniMaxH3ImageToVideo (first=K_i, last=K_{i+1})
          LoadLatent(clip_index=N-1)  → context_latent (preferred)
          MotionContext(context_frames, context_audio, context_latent?, audio_mode=timeline)
            → Guider → dual-turbo samplers → decode
          Trim(trim_frames, match_tail=true) → CreateVideo
          SaveLatent(clip_index=N)
```

| Setting | Value | Notes |
|---------|--------|--------|
| context_length | **22** | VAE grid; last 22 frames of prev |
| audio_context_length | **22** | end-aligned with video pin |
| audio_mode | **timeline** | true waveform continue (not ref sound-alike) |
| encode_mode | **video** | one VAE call for motion |
| anchor_mode | **head** | requires Trim |
| Spectrum | **OFF** | required (pinned rows break forecast) |
| prev video path | abs path via `scp` to `ComfyUI/input/mc_prev_arm*.mp4` | VHS_LoadVideoPath |
| latent slots | `output/h3_context/clip_0000N.safetensors` | Save/Load clip_index |

**Arms are sequential on HEAD** when MC is on (latents + prev video stay local). Dual-turbo still runs per arm; dual-box parallelism is for non-MC runs.

Install: `bash deploy/keyspark/setup_h3_turbo.sh` (also installs Motion Context).

---

## 3. Credits

| Piece | Who |
|-------|-----|
| Dual-sampler Turbo recipe | **[@ANe5s](https://huggingface.co/ANe5s)** discussion #21 |
| Turbo LoRA training | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) |
| ComfyUI pruned Turbo weights | [QrusherZA/H3_Turbo_ComfyUI](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI) |
| Motion + audio chain | [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) |
| Dual-H3 co-tenancy factory | [tonyd2wild/ds4-h3-video-gen-factory](https://github.com/tonyd2wild/ds4-h3-video-gen-factory) |
