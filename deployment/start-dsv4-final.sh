#!/usr/bin/env bash
# DSv4 final deployed parameters (2026-10-01), adapted for configurable paths.
# Requires an existing LMCache server on loopback:5555, chunk-size=256, L1=256GiB.
set -euo pipefail
IMG="${IMAGE:-lazymio/vllm-backport:v0.13.1-sm80}"
MODEL="${MODEL:?Set MODEL to the DeepSeek-V4-Flash-0731 model directory}"
CC="${CACHEDIR:-$PWD/compile-cache/dsv4}"
[ -d "$MODEL" ] || { echo "MODEL directory missing" >&2; exit 1; }
mkdir -p "$CC/triton" "$CC/tilelang" "$CC/inductor"
docker run -d \
  --name "${CONTAINER_NAME:-deepseek-v4-opt}" \
  --network host --ipc host --shm-size 68719476736 \
  --restart unless-stopped \
  --gpus '"device=0,1,2,3"' \
  -v "$MODEL:/model:ro" \
  -v "$CC/triton:/root/.triton" \
  -v "$CC/tilelang:/root/.tilelang" \
  -v "$CC/inductor:/root/.cache" \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e GLOO_SOCKET_IFNAME=lo \
  -e TOKENIZERS_PARALLELISM=false \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_HOST_IP=127.0.0.1 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --entrypoint vllm \
  "$IMG" \
  serve /model \
  --host "${API_HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --served-model-name dataspace-295b \
  --tensor-parallel-size 4 \
  --max-model-len 393216 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2176 \
  --gpu-memory-utilization 0.92 \
  --kv-offloading-size 256 \
  --kv-offloading-backend lmcache \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 40 48 56 64 80 96 120 144 168 192 \
  --max-cudagraph-capture-size 192 \
  --kv-cache-dtype fp8_ds_mla \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --prefix-cache-retention-interval 256 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --trust-remote-code \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
  --hf-overrides '{"head_dtype":"float32"}'
