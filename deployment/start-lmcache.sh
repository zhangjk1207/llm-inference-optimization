#!/usr/bin/env bash
# Companion Docker template matching the recorded cache settings.
# The original DSv4/DSv4.1 LMCache container scripts were not saved locally.
set -euo pipefail
IMAGE="${IMAGE:?Set IMAGE to the tested backport image for this deployment}"
CACHE_PORT="${CACHE_PORT:-5556}"
CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-1024}"
CACHE_SIZE_GB="${CACHE_SIZE_GB:-128}"
CACHE_GPU_WORKERS="${CACHE_GPU_WORKERS:-8}"
exec docker run -d \
  --name "${LMCACHE_CONTAINER_NAME:-deepseek-lmcache}" \
  --gpus all --network host --ipc host --restart unless-stopped \
  --entrypoint python3 "$IMAGE" -m lmcache.cli.main server \
  --host 127.0.0.1 --port "$CACHE_PORT" \
  --http-host 127.0.0.1 --http-port 18556 --prometheus-port 18557 \
  --chunk-size "$CACHE_CHUNK_SIZE" \
  --l1-size-gb "$CACHE_SIZE_GB" --l1-init-size-gb 16 --l1-use-lazy \
  --eviction-policy LRU --eviction-trigger-watermark 0.85 --eviction-ratio 0.20 \
  --max-workers 16 --max-cpu-workers 16 --max-gpu-workers "$CACHE_GPU_WORKERS"
