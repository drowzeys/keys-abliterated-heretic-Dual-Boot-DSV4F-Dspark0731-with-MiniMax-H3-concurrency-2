#!/usr/bin/env bash
# One ComfyUI + MiniMax H3 instance. Run once per node, alongside DS4.
#
# Two instances on two nodes is the configuration benchmarked in the README.
set -euo pipefail

# ORDER MATTERS. Run launch_ds4_pair.sh FIRST and let it finish loading.
# H3 sizes itself to whatever memory is free when it starts: on an idle 121GiB
# node it keeps every component resident and takes ~50GB, and DS4 (which needs
# ~105GiB) then cannot start at all. Launch DS4 first and H3 loads into the
# remaining 16-18GiB, evicting as it goes, and both run. Reversing the order
# breaks the setup and nothing in either program's output explains why.
if ! curl -s -m 5 "${DS4_HEALTH:-http://127.0.0.1:8888/v1/models}" >/dev/null 2>&1; then
  echo "WARNING: DeepSeek-V4-Flash does not look like it is serving yet."
  echo "         Start it FIRST, or H3 will take the memory it needs."
  echo "         Set DS4_HEALTH to point at your head node, or SKIP_CHECK=1 to override."
  [ "${SKIP_CHECK:-0}" = "1" ] || exit 1
fi

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
