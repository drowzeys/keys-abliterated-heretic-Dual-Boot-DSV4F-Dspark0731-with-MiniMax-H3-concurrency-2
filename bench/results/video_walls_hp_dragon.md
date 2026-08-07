# Video wall-clock — HP→dragon ~10s (keyspark lab 2026-08-06/07)

## Sequential

| Run | part1 | part2 | total | res | stack |
|-----|------:|------:|------:|-----|-------|
| `hp_dragon_dual` run2 | 333s | 347s | **680s** | 832×480 | basic dual |
| `hp_dragon_dual_after` | 411s | (killed ~384s+) | est **~851s** | 1728×960 | heretic+Spectrum+ESRGAN |

## Parallel concurrency=2

| Run | phase0 | phase1 | phase2 dual FLF | total | res |
|-----|-------:|-------:|----------------:|------:|-----|
| `hp_dragon_parallel` (fast scouts) | 88s mid | 92s I0‖I2 | **440s** (A424/B440) | **624s** | 1728×960 |
| `hp_dragon_parallel_q` quality I0‖I1 | **192s** | I2 later | ~440s class | ~14min class | 1728×960 |

## Comparison (enhanced stack)

- Sequential est: **~851s**
- Parallel measured: **624s** → **1.36×** faster, **~227s** saved
- Dual-arm only: 864s sequential vs 440s parallel → **~2×** on heavy phase
