#!/usr/bin/env bash
# DeepSeek-V4-Flash, TP=2 across two DGX Sparks.
#
# Run from anywhere with SSH to both nodes. HEAD serves the OpenAI-compatible
# API on :8888; WORKER runs headless. Start the WORKER FIRST — if the head comes
# up first and then restarts, the worker sits holding ~90GiB of loaded weights
# waiting for a peer that never arrives, and you get "TCPStore ... Broken pipe".
set -euo pipefail

HEAD_SSH="${HEAD_SSH:?e.g. user@10.0.0.1}"
WORKER_SSH="${WORKER_SSH:?e.g. user@10.0.0.2}"
HEAD_FABRIC="${HEAD_FABRIC:?private fabric IP of the head, e.g. 192.168.1.1}"
WORKER_FABRIC="${WORKER_FABRIC:?private fabric IP of the worker}"
MODEL_PATH="${MODEL_PATH:-/models/ds4-0731}"
IMAGE="${IMAGE:?your vllm-dspark runtime image}"
MASTER_PORT="${MASTER_PORT:-29625}"

# 0.78 is deliberate. See README: below ~0.70 the KV pool collapses and vLLM
# refuses to start. Do NOT lower this to make room for video — the video model
# lives in the ~16GiB of headroom that already exists at 0.78.
UTIL="${UTIL:-0.78}"
MAXLEN="${MAXLEN:-1048576}"

mkcmd () {  # $1 = node rank, $2 = extra flags
cat <<INNER
export PATH="/opt/env/bin:\${PATH:-}"
exec /opt/env/bin/vllm serve ${MODEL_PATH} \
 --served-model-name deepseek-v4-flash \
 --host 0.0.0.0 --port 8888 --trust-remote-code \
 --tensor-parallel-size 2 --pipeline-parallel-size 1 \
 --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
 --max-model-len ${MAXLEN} --max-num-seqs 6 --max-num-batched-tokens 8192 \
 --gpu-memory-utilization ${UTIL} --enable-prefix-caching \
 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
 --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
 --reasoning-parser deepseek_v4 \
 --enable-flashinfer-autotune \
 --nnodes 2 --node-rank $1 --master-addr ${HEAD_FABRIC} --master-port ${MASTER_PORT} $2
INNER
}

launch () {  # $1 ssh target, $2 fabric ip, $3 rank, $4 extra, $5 container name
  # Stage in /var/tmp, NEVER /tmp: /tmp is wiped on reboot, Docker then creates
  # a DIRECTORY at the missing bind source, and the container exits 127 while
  # `docker ps` still shows it as Up.
  mkcmd "$3" "$4" | ssh "$1" "cat > /var/tmp/ds4cmd.sh && test -s /var/tmp/ds4cmd.sh"
  ssh "$1" "docker rm -f $5 >/dev/null 2>&1 || true"

  # Unified memory: page cache steals from the GPU allocator and you get
  # NVRM NV_ERR_NO_MEMORY plus a worker death mid-warmup. Always drop first.
  ssh "$1" "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null"

  # --privileged AND /dev/infiniband are REQUIRED. Without the IB devices in the
  # container NCCL cannot load an IB net plugin, and since the env pins
  # NCCL_NET=IB there is no fallback: it dies with "NCCL error: invalid usage"
  # while the fabric itself tests perfectly healthy.
  ssh "$1" "docker run -d --name $5 --network host --ipc host --gpus all \
      --privileged --device /dev/infiniband:/dev/infiniband \
      --shm-size 10g --ulimit memlock=-1:-1 --restart unless-stopped \
      -v /var/tmp/models:/models -v \$HOME/.cache/huggingface:/cache/huggingface \
      -v /var/tmp/ds4cmd.sh:/var/tmp/ds4cmd.sh \
      -e VLLM_HOST_IP=$2 -e NCCL_NET=IB -e NCCL_IB_HCA=rocep1s0f0 \
      -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_GID_INDEX=5 \
      ${IMAGE} -c 'bash /var/tmp/ds4cmd.sh'"
  echo "  [$5] started"
}

echo "== worker first =="
launch "$WORKER_SSH" "$WORKER_FABRIC" 1 "--headless" vllm_ds4_worker
sleep 5
echo "== head =="
launch "$HEAD_SSH" "$HEAD_FABRIC" 0 "" vllm_ds4_head

echo
echo "Weights take ~5.5 min. Watch for: 'Model loading took 79.51 GiB'"
echo "then 'GPU KV cache size: 1,473,052 tokens'. If you instead see"
echo "'No available memory for the cache blocks', your UTIL is too low."
# --- GB10 vLLM spin-wait fix (see GB10_SPIN_WAIT_PATCH.md) --------------------
# If this script runs a stock vLLM image, the served container will busy-spin CPU
# cores at max clock while waiting on shm_broadcast (busy_loop_s=1s default),
# heating the shared GB10 SoC. Prefer an image built with the patch baked in.
# https://nacyot.github.io/artifacts/vllm-spin-wait-gb10/
