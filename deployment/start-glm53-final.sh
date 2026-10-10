#!/usr/bin/env bash
# GLM deployed configuration (2026-10-10): sub-block prefill, MTP k=4, capture max=80.
# Adapted from the production script; paths are configurable. See ../README.md.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# NCCL: 保留传输层约束，算法/协议交给 NCCL 自动选择（已验证）
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export NO_PROXY='*'

export no_proxy='*'

APPTAINER="${APPTAINER:-apptainer}"
IMAGE="${IMAGE:?Set IMAGE to the tested vllm-backport-v0.13.1-sm80-local.sif}"
MODEL="${MODEL:?Set MODEL to your GLM-5.3-Flash-W4A16-vllm directory}"
LOGDIR="${LOGDIR:-$PWD/local-results/glm-logs}"
CACHEDIR="${CACHEDIR:-$PWD/compile-cache/glm}"
[ -d "$MODEL" ] && [ -f "$IMAGE" ] || { echo "MODEL directory or IMAGE file missing" >&2; exit 1; }
command -v "$APPTAINER" >/dev/null
command -v ss >/dev/null
# Both processes must load the same LMCache package. See patches/lmcache-registration/README.md.
LMCACHE_BINDS=()
if [[ -n "${LMCACHE_PACKAGE:-}" ]]; then
  [[ -f "$LMCACHE_PACKAGE/__init__.py" ]] || { echo "LMCACHE_PACKAGE must be a complete package directory" >&2; exit 1; }
  LMCACHE_BINDS=(--bind "$LMCACHE_PACKAGE:/usr/local/lib/python3.12/dist-packages/lmcache:ro")
fi
mkdir -p "$LOGDIR" "$CACHEDIR/triton" "$CACHEDIR/tilelang" "$CACHEDIR/inductor"

"$APPTAINER" exec --nv "${LMCACHE_BINDS[@]}" \
  --bind "$MODEL:/model:ro" \
  --bind "$CACHEDIR/triton:/root/.triton" \
  --bind "$CACHEDIR/tilelang:/root/.tilelang" \
  --bind "$CACHEDIR/inductor:/root/.cache" \
  "$IMAGE" \
  python3 -m lmcache.cli.main server \
  --host 127.0.0.1 \
  --port 5555 \
  --chunk-size 1152 \
  --http-host 127.0.0.1 \
  --http-port 18556 \
  --prometheus-port 18557 \
  --disable-metrics \
  --l1-size-gb 512 \
  --l1-init-size-gb 16 \
  --l1-use-lazy \
  --eviction-policy LRU \
  --eviction-trigger-watermark 0.85 \
  --eviction-ratio 0.20 \
  --max-workers 16 \
  --max-cpu-workers 16 \
  --max-gpu-workers 4 \
  >"$LOGDIR/lmcache.log" 2>&1 &
LM_PID=$!
API_PID=""

cleanup() {
  if [[ -n "$API_PID" ]]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  kill "$LM_PID" 2>/dev/null || true
  wait "$LM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  if ! kill -0 "$LM_PID" 2>/dev/null; then
    tail -100 "$LOGDIR/lmcache.log" >&2 || true
    exit 1
  fi
  if ss -ltn 2>/dev/null | grep -q ':5555 ' \
      && ss -ltn 2>/dev/null | grep -q ':18556 '; then
    break
  fi
  sleep 1
done
if ! kill -0 "$LM_PID" 2>/dev/null \
  || ! ss -ltn 2>/dev/null | grep -q ':5555 ' \
  || ! ss -ltn 2>/dev/null | grep -q ':18556 '; then
  tail -100 "$LOGDIR/lmcache.log" >&2 || true
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCHDIR="$SCRIPT_DIR/../patches"

"$APPTAINER" exec --nv "${LMCACHE_BINDS[@]}" \
  --bind "$MODEL:/model:ro" \
  --bind "$CACHEDIR/triton:/root/.triton" \
  --bind "$CACHEDIR/tilelang:/root/.tilelang" \
  --bind "$CACHEDIR/inductor:/root/.cache" \
  --bind "$PATCHDIR:/opt-patch:ro" \
  --env "VLLM_FORCE_PCIE_CUSTOM_AR=${VLLM_FORCE_PCIE_CUSTOM_AR:-1}" \
  "$IMAGE" \
  python3 /opt-patch/serve_custom_ar.py serve /model \
  --host "${API_HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 1152 \
  --long-prefill-token-threshold 1024 \
  --gpu-memory-utilization 0.92 \
  --kv-offloading-size 512 \
  --kv-offloading-backend lmcache \
  --prefix-cache-retention-interval 1152 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,5,8,10,16,20,25,32,40,48,56,64,72,80],"max_cudagraph_capture_size":80}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --trust-remote-code \
  >"$LOGDIR/vllm.log" 2>&1 &
API_PID=$!

# A supervisor must restart the pair if either child exits, even with status 0.
set +e
wait -n "$LM_PID" "$API_PID"
CHILD_STATUS=$?
set -e
if (( CHILD_STATUS == 0 )); then
  CHILD_STATUS=1
fi
exit "$CHILD_STATUS"
