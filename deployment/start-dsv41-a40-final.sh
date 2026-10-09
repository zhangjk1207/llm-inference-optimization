#!/usr/bin/env bash
# A40 deployment documented on 2026-09-23, updated to final 8-seq mode from 2026-10-01 report.
# This is reconstructed from the documented production command, not a downloaded final snapshot.
set -euo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL:?Set MODEL to the DeepSeek-V4.1-Flash model directory}"
IMAGE="${IMAGE:-lazymio/vllm-backport:v0.13.1-sm86}"
CONTAINER_NAME="${CONTAINER_NAME:-deepseek-v41-flash-a40}"
PORT="${PORT:-18005}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

case "$MAX_NUM_SEQS" in
  2) GRAPH_CAPTURE_SIZES='[6,12]' ;;
  4) GRAPH_CAPTURE_SIZES='[6,12,18,24]' ;;
  8) GRAPH_CAPTURE_SIZES='[6,12,18,24,30,36,42,48]' ;;
  *) echo 'MAX_NUM_SEQS must be 2, 4, or 8' >&2; exit 1 ;;
esac

COMPILATION_CONFIG="$(python3 -c '
import json,sys
s=json.loads(sys.argv[1])
print(json.dumps({
  "cudagraph_mode":"FULL_AND_PIECEWISE",
  "cudagraph_capture_sizes":s,
  "max_cudagraph_capture_size":max(s),
}))' "$GRAPH_CAPTURE_SIZES")"

[ -d "$MODEL_DIR" ] || { echo "MODEL directory missing" >&2; exit 1; }
# LMCache is started separately; final capacity=128GiB, chunk=1024, MP port=5556.
mkdir -p "${CACHEDIR:-$PWD/compile-cache/dsv41}/triton" "${CACHEDIR:-$PWD/compile-cache/dsv41}/tilelang"

exec docker run -d --name "$CONTAINER_NAME" --init --restart unless-stopped \
  --runtime nvidia --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-opt max-size=100m --log-opt max-file=3 \
  -v "$MODEL_DIR:/model:ro" \
  -v "${CACHEDIR:-$PWD/compile-cache/dsv41}:/root/.cache" \
  -v "${CACHEDIR:-$PWD/compile-cache/dsv41}/triton:/root/.triton" \
  -v "${CACHEDIR:-$PWD/compile-cache/dsv41}/tilelang:/root/.tilelang" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_SOCKET_IFNAME=lo -e "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}" -e NCCL_IB_DISABLE=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NO_PROXY='*' -e no_proxy='*' \
  --entrypoint bash "$IMAGE" -c \
  'set -e; test -e /usr/lib/x86_64-linux-gnu/libcuda.so || \
     ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so; \
   exec vllm serve "$@"' bash /model \
  --host "${API_HOST:-127.0.0.1}" --port "$PORT" \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5556}}' \
  --prefix-cache-retention-interval 1024 \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching --prefix-match-unit 8 \
  --enable-prompt-tokens-details --disable-custom-all-reduce \
  --compilation-config "$COMPILATION_CONFIG" \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}' \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice
