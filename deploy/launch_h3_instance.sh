#!/usr/bin/env bash
# One ComfyUI + MiniMax H3 instance. Run once per node, alongside DS4.
#
# Two instances on two nodes is the configuration benchmarked in the README.
set -euo pipefail

NODE_SSH="${NODE_SSH:?e.g. user@10.0.0.1}"
H3_DIR="${H3_DIR:?absolute path to the h3 dir on that node, containing ComfyUI/ and models/}"
IMAGE="${IMAGE:-comfy-h3:gb10}"
PORT="${PORT:-8188}"
NAME="${NAME:-comfy-h3}"

ssh "$NODE_SSH" "docker rm -f ${NAME} >/dev/null 2>&1 || true"

# The image's default entrypoint is NOT python — without --entrypoint the
# container runs `sleep` with your args and exits 1.
#
# --disable-pinned-memory is REQUIRED on unified-memory boxes: ComfyUI otherwise
# page-locks most of available RAM and starves everything sharing the pool.
ssh "$NODE_SSH" "docker run -d --name ${NAME} --network host --gpus all \
    --shm-size 8g --entrypoint /opt/env/bin/python3 \
    -v ${H3_DIR}:${H3_DIR} -w ${H3_DIR}/ComfyUI \
    ${IMAGE} main.py --listen 0.0.0.0 --port ${PORT} \
      --disable-pinned-memory --fp16-intermediates"

echo "  [${NAME}] started on :${PORT}"
echo "  check: curl -s http://<node>:${PORT}/system_stats"
echo
echo "NOTE: ComfyUI's own vram_free is unreliable for capacity planning."
echo "It counts torch's cached-but-unallocated blocks as free, and on a"
echo "unified-memory node it reads MemFree (which excludes reclaimable page"
echo "cache). Use nvidia-smi or /proc/meminfo instead."
