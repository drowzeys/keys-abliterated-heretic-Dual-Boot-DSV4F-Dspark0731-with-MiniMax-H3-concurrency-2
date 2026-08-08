# Dual-H3 multishot FLF (seamless parallel)

Hermes + keyspark architecture for **two MiniMax-H3 instances** (concurrency=2).

## Why this is “flawless” at seams

Each motion arm is **first/last-frame (FLF)** conditioned:

```
arm_i  first_frame = K_i   last_frame = K_{i+1}
arm_{i+1} first_frame = K_{i+1}  …
```

The **last pixel of segment i is the same PNG as the first pixel of segment i+1**.  
Stitch with a **hard cut** (no xfade). Crossfades only hide mismatches; shared keys remove the mismatch.

## Pipeline

```
Phase 0  SEQUENTIAL on .2
         generate keyframes K0 … KN (full steps, quality)
         optional: re-anchor every key to H3_I0_REF (identity lock)

Phase 1  PARALLEL waves of 2
         .2 arm odd  ‖  .3 arm even
         each arm: MiniMaxH3 first=K_i last=K_{i+1}

Phase 2  ffmpeg hard-cut concat  → master
```

Wall-clock for motion ≈ **ceil(N_arms / 2) × single_arm_time** (≈ half vs pure sequential arms).

## Code

| File | Role |
|------|------|
| `deploy/keyspark/multishot_flf.py` | **Generic engine** + natural skin default |
| `deploy/keyspark/keyframe_dual_flf.py` | HP→dragon 3 keys / 2 arms |
| `deploy/keyspark/connelly_flf_20s.py` | 5 keys / 4 arms (Hermes story) |
| `deploy/keyspark/connelly_flf_30s.py` | 7 keys / 6 arms (Hermes story) |
| `deploy/keyspark/run_quality_parallel.sh` | One-shot runner + preflight |

## Natural skin (default ON)

`multishot_flf.NATURAL_SKIN` is prepended to every prompt:

- visible pores, creases, subsurface scatter  
- no airbrush / plastic CGI / beauty filter  
- prefer film grain + practical light  

If ESRGAN over-sharpens faces: `H3_UPSCALE=0`.

## Env

```bash
H3_I0_REF=/path/to/face.png   # identity for every key (recommended)
H3_KEY_MODE=face|chain|both   # auto: face if ref else chain
H3_UPSCALE=0                  # softer, less processed skin
H3_LEN=124 H3_STEPS=20        # motion arms
H3_LEN_KF=49 H3_STEPS_KF=20   # keyframes
```

## Credit

- Dual-H3 + DS4 co-tenancy factory: **[tonyd2wild/ds4-h3-video-gen-factory](https://github.com/tonyd2wild/ds4-h3-video-gen-factory)**  
- FLF multishot parallel keys→arms→hardcut: Hermes session refinement on keyspark  
