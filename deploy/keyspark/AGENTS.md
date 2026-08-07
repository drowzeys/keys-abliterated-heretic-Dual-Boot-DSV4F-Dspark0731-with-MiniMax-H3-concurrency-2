# Agents — keyspark dual-boot (read first)

**Full one-shot recipe:** [`docs/AGENT_ONESHOT_RECIPE.md`](../../docs/AGENT_ONESHOT_RECIPE.md)  
**Measured speeds:** [`docs/KEYSPARK_RESULTS.md`](../../docs/KEYSPARK_RESULTS.md)

## Non-negotiables

- Nodes: **10.100.10.2** + **10.100.10.3** only (never `.1`/5482)
- Order: **DS4 first → H3 second**; teardown reverse
- `GPU_MEMORY_UTILIZATION` default **0.78**, hard cap **0.85**
- H3 fleet concurrency **2** = one heavy job per Spark
- Default stack: ablit DS4 + heretic H3 (`STACK=ablit`)

## Commands

```bash
bash deploy/keyspark/bringup.sh
bash deploy/keyspark/status.sh
bash deploy/keyspark/run_quality_parallel.sh
bash deploy/keyspark/teardown.sh
```
