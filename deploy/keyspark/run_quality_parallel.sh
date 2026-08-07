#!/usr/bin/env bash
# Quality-first dual-H3 10s pipeline on the ablit+heretic dual-boot stack.
#
# Requires: DS4 ablit serving on .2:8888, H3 heretic on .2/.3:8188 (concurrency=2).
# Does NOT start DS4/H3 — use bringup first:
#   STACK=ablit bash ~/ds4-h3-video-gen-factory/deploy/keyspark/bringup.sh
#
# Usage:
#   bash run_quality_parallel.sh
#   H3_I0_REF=/path/to/face.png bash run_quality_parallel.sh
#   H3_UPSCALE=0 bash run_quality_parallel.sh          # skip ESRGAN
#   H3_QUALITY_ID=0 bash run_quality_parallel.sh       # fast scouts (worse face)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/profile.ablit-heretic-dual.env"

say() { echo -e "\n=== $* ==="; }

say "preflight dual-boot stack"
ok=1
if ! curl -sf -m 5 "${DS4_BASE}/models" >/dev/null 2>&1; then
  echo "ERROR: DS4 not healthy at ${DS4_BASE}" >&2
  echo "  STACK=ablit bash $ROOT/bringup.sh" >&2
  ok=0
fi
for host in "$H3_HEAD" "$H3_WORKER"; do
  if ! curl -sf -m 5 "http://${host}/system_stats" >/dev/null 2>&1; then
    echo "ERROR: H3 down at http://${host}" >&2
    ok=0
  fi
done
if [ "$ok" != "1" ]; then
  exit 1
fi

# show models / H3 free RAM
echo "DS4 model(s):"
curl -sf -m 5 "${DS4_BASE}/models" | python3 -c \
  "import sys,json; print(' ', [m.get('id') for m in json.load(sys.stdin).get('data',[])])" 2>/dev/null || true
for host in "$H3_HEAD" "$H3_WORKER"; do
  curl -sf -m 3 "http://${host}/system_stats" | python3 -c \
    "import sys,json; s=json.load(sys.stdin); print(f'  H3 {\"$host\"} ram_free={s[\"system\"][\"ram_free\"]/1e9:.1f}G')" 2>/dev/null || true
done

echo
echo "profile: $STACK_PROFILE  H3_FLEET_CONCURRENCY=$H3_FLEET_CONCURRENCY"
echo "  quality_id=$H3_QUALITY_ID  spectrum=$H3_SPECTRUM  upscale=$H3_UPSCALE"
echo "  I0 steps=$H3_STEPS_I0 len=$H3_LEN_I0  full steps=$H3_STEPS len=$H3_LEN"
echo "  I0_REF=${H3_I0_REF:-none}"
echo "  OUT_DIR=$OUT_DIR"

# clear stuck queues (safe if idle)
for host in "$H3_HEAD" "$H3_WORKER"; do
  curl -sf -X POST "http://${host}/interrupt" -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1 || true
  curl -sf -X POST "http://${host}/queue" -H 'Content-Type: application/json' -d '{"clear":true}' >/dev/null 2>&1 || true
done

mkdir -p "$OUT_DIR" "$WORK_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

say "quality-first parallel dual-FLF (concurrency=2)"
exec python3 "$ROOT/keyframe_dual_flf.py"
