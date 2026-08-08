#!/usr/bin/env bash
# Install MiniMax-H3 Turbo (few-step LoRA) on both co-tenant H3 nodes.
#
# Weights (ComfyUI-fixed for pruned base):
#   https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI
# Original Turbo training / nodes:
#   https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
#   https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
#
# Usage:
#   bash deploy/keyspark/setup_h3_turbo.sh
#   RESTART=1 bash deploy/keyspark/setup_h3_turbo.sh   # restart Comfy after install
set -euo pipefail

HEAD="${HEAD:-10.100.10.2}"
WORKER="${WORKER:-10.100.10.3}"
H3_DIR="${H3_DIR:-/home/keyspark/h3-cotenancy}"
CACHE="${CACHE:-$HOME/.cache/h3-turbo}"
LORA_NAME="${H3_TURBO_LORA:-minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors}"
HF_REPO="QrusherZA/H3_Turbo_ComfyUI"
NODES_URL="https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
RESTART="${RESTART:-1}"

say() { echo -e "\n=== $* ==="; }

say "0/4 download Turbo LoRA ($LORA_NAME)"
mkdir -p "$CACHE"
if [ ! -s "$CACHE/$LORA_NAME" ]; then
  if command -v hf >/dev/null 2>&1; then
    hf download "$HF_REPO" "$LORA_NAME" --local-dir "$CACHE"
  else
    curl -L --fail --retry 3 -o "$CACHE/$LORA_NAME" \
      "https://huggingface.co/${HF_REPO}/resolve/main/${LORA_NAME}?download=true"
  fi
fi
ls -lah "$CACHE/$LORA_NAME"

say "1/4 clone / update custom nodes (local cache)"
NODES_SRC="${NODES_SRC:-/tmp/ComfyUI-MiniMax-H3-Turbo}"
if [ -d "$NODES_SRC/.git" ]; then
  git -C "$NODES_SRC" pull --ff-only || true
else
  git clone --depth 1 "$NODES_URL" "$NODES_SRC"
fi

say "2/4 install nodes + LoRA on .2 and .3"
for ip in "$HEAD" "$WORKER"; do
  echo "-- $ip --"
  ssh "keyspark@$ip" "mkdir -p $H3_DIR/ComfyUI/custom_nodes $H3_DIR/ComfyUI/models/loras"
  rsync -a --delete "$NODES_SRC/" "keyspark@$ip:$H3_DIR/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo/"
  scp -q "$CACHE/$LORA_NAME" "keyspark@$ip:$H3_DIR/ComfyUI/models/loras/$LORA_NAME"
  ssh "keyspark@$ip" "test -f $H3_DIR/ComfyUI/models/loras/$LORA_NAME && \
    test -f $H3_DIR/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py && echo OK"
done

if [ "$RESTART" = "1" ]; then
  say "3/4 restart H3 Comfy on both nodes (loads Turbo nodes)"
  for ip in "$HEAD" "$WORKER"; do
    ssh "keyspark@$ip" "bash -s" <<REMOTE
set -e
H3_DIR='$H3_DIR'
if [ -f "\$H3_DIR/logs/comfyui.pid" ]; then kill "\$(cat \$H3_DIR/logs/comfyui.pid)" 2>/dev/null || true; fi
fuser -k 8188/tcp 2>/dev/null || true
sleep 2
cd "\$H3_DIR"
nohup ./h3-comfy-launch.sh > "\$H3_DIR/logs/comfyui.log" 2>&1 &
echo \$! > "\$H3_DIR/logs/comfyui.pid"
echo restarted pid=\$(cat \$H3_DIR/logs/comfyui.pid)
REMOTE
  done
  echo "waiting for system_stats..."
  for ip in "$HEAD" "$WORKER"; do
    ok=0
    for _ in $(seq 1 36); do
      if curl -sf -m 3 "http://${ip}:8188/system_stats" >/dev/null 2>&1; then ok=1; break; fi
      sleep 5
    done
    echo "  $ip: $([ $ok = 1 ] && echo OK || echo DOWN)"
  done
else
  say "3/4 skip restart (RESTART=0) — restart Comfy manually to load nodes"
fi

say "4/4 verify Turbo node classes"
for ip in "$HEAD" "$WORKER"; do
  curl -sf "http://${ip}:8188/object_info" | python3 -c '
import sys,json
d=json.load(sys.stdin)
need=["MiniMaxH3TurboLoRA","MiniMaxH3TurboSampler"]
print(sys.argv[1], {n:(n in d) for n in need})
' "$ip" || echo "$ip object_info failed (Comfy not ready?)"
done

echo
echo "Enable turbo in jobs:"
echo "  H3_TURBO=1 bash deploy/keyspark/run_quality_parallel.sh"
echo "  # optional: H3_STEPS=6 (default when turbo)  H3_TURBO_LOW_VRAM=1 under tight RAM"
echo "Credits: larryvrh MiniMax-H3-Turbo-Lora + ComfyUI nodes; QrusherZA ComfyUI-fixed pruned weights."
