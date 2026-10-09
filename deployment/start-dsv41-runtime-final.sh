#!/usr/bin/env bash
set -euo pipefail
# Final runtime script downloaded on 2026-10-02; common.env is not available here.
# Supply the pre-existing runtime environment; this script starts only vLLM.
: "${VENV:?Set VENV to the tested virtualenv}"
: "${RUNTIME:?Set RUNTIME to the tested vLLM runtime}"
: "${MODEL:?Set MODEL to the model directory}"
exec "$VENV/bin/python" "$RUNTIME/bin/vllm" serve "$MODEL" \
  --host "${API_HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5556}}' \
  --prefix-cache-retention-interval 1024 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching \
  --prefix-match-unit 8 \
  --enable-prompt-tokens-details \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,42,48],"max_cudagraph_capture_size":48}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}' \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice
