#!/usr/bin/env bash
# One MiniMax H3 ComfyUI instance per node (.2 and .3), co-tenant with DS4.
# Native process. Tony knobs: --disable-pinned-memory, start AFTER DS4 is serving.
set -euo pipefail

H3_DIR="${H3_DIR:-/home/keyspark/h3-cotenancy}"
PORT="${H3_PORT:-8188}"
HEAD="${HEAD:-10.100.10.2}"
WORKER="${WORKER:-10.100.10.3}"
HEAD_SSH="${HEAD_SSH:-keyspark@$HEAD}"
WORKER_SSH="${WORKER_SSH:-keyspark@$WORKER}"
DS4_HEALTH="${DS4_HEALTH:-http://${HEAD}:8888/v1/models}"

if ! curl -sf -m 5 "$DS4_HEALTH" >/dev/null 2>&1; then
  echo "WARNING: DS4 not healthy at $DS4_HEALTH — start DS4 FIRST."
  [ "${SKIP_CHECK:-0}" = "1" ] || exit 1
fi

launch_one() {
  local ssh_t="$1" label="$2"
  echo "-- launch H3 on $label ($ssh_t) --"
  ssh "$ssh_t" "bash -s" <<REMOTE
set -e
H3_DIR='$H3_DIR'
PORT='$PORT'
mkdir -p "\$H3_DIR/logs"
cat > "\$H3_DIR/h3-comfy-launch.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda-13.0/bin:\${PATH:-}
export CUDA_HOME=\${CUDA_HOME:-/usr/local/cuda-13.0}
cd /home/keyspark/h3-cotenancy/ComfyUI
exec .venv/bin/python main.py --listen 0.0.0.0 --port 8188 --disable-pinned-memory "\$@"
EOF
chmod +x "\$H3_DIR/h3-comfy-launch.sh"
if [ -f "\$H3_DIR/logs/comfyui.pid" ]; then
  kill "\$(cat \$H3_DIR/logs/comfyui.pid)" 2>/dev/null || true
fi
fuser -k \${PORT}/tcp 2>/dev/null || true
sleep 1
cd "\$H3_DIR"
nohup ./h3-comfy-launch.sh > "\$H3_DIR/logs/comfyui.log" 2>&1 &
echo \$! > "\$H3_DIR/logs/comfyui.pid"
echo "  pid=\$(cat \$H3_DIR/logs/comfyui.pid)"
REMOTE
}

launch_one "$HEAD_SSH"   ".2"
launch_one "$WORKER_SSH" ".3"

echo "waiting for Comfy system_stats..."
for ip in "$HEAD" "$WORKER"; do
  ok=0
  for _ in $(seq 1 36); do
    if curl -sf -m 3 "http://${ip}:${PORT}/system_stats" >/dev/null 2>&1; then
      ok=1; break
    fi
    sleep 5
  done
  if [ "$ok" = "1" ]; then
    echo "  H3 ok http://${ip}:${PORT}"
  else
    echo "  H3 NOT ready http://${ip}:${PORT}"
    ssh "keyspark@$ip" "tail -40 $H3_DIR/logs/comfyui.log" || true
  fi
done
