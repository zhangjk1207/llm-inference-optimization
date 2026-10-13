# LLM Inference Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-6e7bf4.svg)](LICENSE)
[![中文文档](https://img.shields.io/badge/README-中文-red.svg)](README.zh-CN.md)

PCIe multi-GPU deployment and inference optimization for **GLM-5.3-Flash**, **DeepSeek-V4-Flash** and **DeepSeek-V4.1-Flash** — deployment scripts, concurrent-load benchmarks, and performance diagnostics. Main levers: CUDA-graph coverage, speculative decoding, multi-GPU communication, and KV cache (CPU offload via LMCache). MIT licensed.

## TL;DR results

| Setup | Headline result |
|---|---|
| GLM-5.3-Flash W4A16 · 4×A800 · TP4 | Long-input admission fix: new 8K-input TTFT while a long generation is running **30.84 s → 4.00 s**; 32K two-pass 36.71 / 37.89 s → 14.86 / 14.89 s; 6/8/10-agent task medians 18.37 / 20.16 / 22.96 s → 14.01 / 14.92 / 18.28 s |
| DeepSeek-V4-Flash · 4×A800 · TP4+EP | After extending CUDA-graph capture range, the ≥6-way throughput cliff disappears: 6-way ~+236%, 8-way ~+211%, 16-way ~+111% aggregate; 108K-token cold-context TTFT 36.8 s |
| DeepSeek-V4.1-Flash · 8×A40 · TP8+EP | Same-config A/B across two nodes (NCCL P2P off vs on); raising max sequences 4→8 on the P2P node: aggregate **340.7 → 583.3 tok/s (+71.2%)**, TTFT **2.3–2.5 s → 0.57–0.73 s** |

Scope notes are stated per section below. In particular, the GLM scheduling change is a queueing/admission improvement — **no total-throughput improvement is claimed for it**.

## Models & deployment environment

| Model | GPUs | Image | Interconnect | Deployment script |
|---|---|---|---|---|
| [GLM-5.3-Flash-W4A16](https://modelscope.cn/models/JANGQ-AI/GLM-5.3-Flash-W4A16) | 4×A800 80GB, 320GB | [lazymio/vllm-backport:v0.13.1-sm80](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm80), Apptainer | PCIe, no NVLink; intra-group PIX, cross-NUMA SYS; P2P available, CustomAllreduce enabled | [GLM](deployment/start-glm53-final.sh) |
| [DeepSeek-V4-Flash-0731](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731) | 4×A800 80GB, 320GB | [lazymio/vllm-backport:v0.13.1-sm80](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm80), Docker | PCIe, PIX/PXB/SYS; NCCL P2P disabled | [DeepSeek V4](deployment/start-dsv4-final.sh) |
| [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash), node A | 8×A40 48GB, 384GB | [lazymio/vllm-backport:v0.13.1-sm86](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm86), Docker | PCIe, no NVLink; PIX/PXB/SYS; NCCL P2P disabled | [A40 node A](deployment/start-dsv41-a40-final.sh) |
| [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash), node B | 8×A40 48GB, 384GB | [lazymio/vllm-backport:v0.13.1-sm86](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm86) | PCIe, no NVLink; P2P enabled | [A40 node B](deployment/start-dsv41-a40-p2p-final.sh) |

The two A40 nodes use identical inference configs, differing only in P2P state. A800 is SM80, A40 is SM86. The P2P-disabled node hit a communication hang during real engine initialization, so `NCCL_P2P_DISABLE=1` is kept.

## GLM-5.3-Flash W4A16

### Configuration

| Parameter | Value |
|---|---|
| Parallelism / kernels | TP4, MARLIN WNA16 MoE, CustomAllreduce |
| Max context / max sequences | 262144 / 16 |
| Speculative decoding / batched tokens | MTP k=4 / 1152 |
| Per-round prefill cap for long inputs | 1024 (2026-10-10, improves admission of new inputs while a long generation is in flight) |
| GPU memory budget | 0.92 |
| KV cache | Prefix cache + LMCache, CPU L1 512GiB, chunk 1152 |
| CUDA graph | FULL_AND_PIECEWISE; `[1,2,4,5,8,10,16,20,25,32,40,48,56,64,72,80]` |

### 2026-10-10: long-input scheduling & agent validation

Deployment adds `--long-prefill-token-threshold 1024`, keeping MTP k=4, 256K total context and the 16-sequence cap. On this backport, total budget and physical cache blocks are both 1152; when existing requests already occupy decode budget, the default alignment can cut the schedulable amount for a new long input to zero. The 1024 cap enables the sub-block prefill supported by this version, improving long-input admission.

| Metric | Before | After |
|---|---:|---:|
| New 8K-input TTFT while a long generation is running | 30.84 s | 4.00 s |
| New 32K-input TTFT while a long generation is running, two passes | 36.71 / 37.89 s | 14.86 / 14.89 s |
| Median completion time of 6 / 8 / 10 full agent tasks | 18.37 / 20.16 / 22.96 s | 14.01 / 14.92 / 18.28 s |
| Single-request 16K cold-input TTFT | 4.70 s | 6.59 s |

Generation requests go through new-api. TTFT includes thinking. Under interleaved load, the median first-token latency of short requests drops from 36.23 s to 1.22 s, and first-prose-token from 38.39 s to 15.44 s. **The gains are mostly reduced waiting; cold long inputs and in-flight requests may get slower — no total-throughput improvement is claimed.** Agent runs are single-round only and cannot serve as P95/SLA evidence.

Cache readback, 12 new-prefix boundary checks, cancellation and near-256K input checks passed. No fixed 8-way admission limit or per-user gateway fair scheduling has been deployed. Full conditions, truncated samples and costs: [measurement report (Chinese)](results/glm-20261010/README.md).

The deployment also uses the [LMCache registration-liveness patch](patches/lmcache-registration/README.md). The startup script can bind the same patched package to both the cache service and the inference process via `LMCACHE_PACKAGE`, exit as a whole when either subprocess dies, and let an external process manager restart it.

### 2026-10-06: historical short-input concurrency

Warm runs at each concurrency, 512-token outputs, p50 over 5 rounds. Throughput in tok/s, TTFT in seconds. This historical set predates the 10-10 scheduling fix, does not represent re-measurement of the current config, and does not cover new-long-input admission while a long generation is in flight.

| Concurrency | Baseline aggregate tok/s | Final aggregate tok/s | Δ | Final TTFT, s | Per-round throughput range |
|---:|---:|---:|---:|---:|---|
| 1 | 201.9 | 225.7 | +11.8% | 0.487 | 223.5–226.6 |
| 2 | 338.4 | 372.8 | +10.2% | 0.590 | 353.9–375.0 |
| 4 | 502.8 | 531.3 | +5.7% | 0.792 | 525.3–540.6 |
| 8 | 719.8 | 703.8 | -2.2% | 1.064 | 648.8–729.6 |
| 16 | 977.7 | 1032.8 | +5.6% | 1.738 | 1024.5–1038.1 |

One 8-way retest measured 715.6 tok/s, TTFT 1.128 s. Gains at 1 / 2 / 16 ways are clear; 8-way is roughly flat.

Raw results: [baseline](results/glm-20261006/baseline-v2-prod-c1-2-4-8-16-o512.json) · [final config](results/glm-20261006/optv10-exactsize-c1-2-4-8-16-o512.json) · [8-way retest](results/glm-20261006/optv10-c8clean-c8-o512.json).

## DeepSeek-V4-Flash

### Configuration

| Parameter | Value |
|---|---|
| Parallelism / speculative decoding | TP4 + EP, DSpark k=5 |
| Max context / max sequences | 393216 / 32 |
| Batched tokens / GPU memory budget | 2176 / 0.92 |
| KV cache | fp8_ds_mla; GPU KV 3,745,707 tokens; CPU L1 256GiB, chunk 256 |
| Communication | NCCL P2P, InfiniBand and CustomAllreduce disabled |
| CUDA graph | `[1,2,4,8,16,24,32,40,48,56,64,80,96,120,144,168,192]` |
| Graph pool | ~1.23–1.25 GiB/GPU |

### Concurrency

Warm tests, 512-token outputs, p50 aggregate throughput.

| Concurrency | Baseline aggregate tok/s | Final aggregate tok/s | Final per-way tok/s, p50 | Final TTFT, s |
|---:|---:|---:|---:|---:|
| 1 | 224.1 | 220.5 | 220 | 0.35 |
| 4 | 535.8 | 537.5 | 137 | 0.57 |
| 6 | 194.2 | 652.6 | 110 | 0.41 |
| 8 | 255.7 | 795.0 | 103 | 0.46 |
| 16 | 481.0 | 1014.6 | 64.8 | 0.52 |
| 24 | — | 1126.3 | 47.3 | — |
| 32 | — | 1187.6 | 32.7 | — |

After extending the CUDA-graph capture range, the ≥6-way throughput cliff disappears: 6-way ~+236%, 8-way ~+211%, 16-way ~+111%. 108K-token cold long-context TTFT is 36.8 s; chat, reasoning and tool-calling regressions pass.

[Deployment script](deployment/start-dsv4-final.sh) · [CUDA-graph tuning notes (Chinese)](docs/cuda-graphs.md)

## DeepSeek-V4.1-Flash: 8×A40

### Configuration

| Parameter | Value |
|---|---|
| Parallelism / speculative decoding | TP8 + EP, DSpark k=5, local argmax reduction |
| Max context / max sequences | 262144 / 8 |
| Batched tokens / GPU memory budget | 2048 / 0.95 |
| KV / Engram | fp8_ds_mla; CPU L1 128GiB, chunk 1024; Engram CPU offload |
| CUDA graph | FULL_AND_PIECEWISE; `[6,12,18,24,30,36,42,48]` |
| Communication | Node A P2P disabled, node B P2P enabled; CustomAllreduce disabled on both |
| GPU KV capacity | Node A 936,141 tokens; node B 937,055 tokens |
| Graph pool | ~0.50 GiB/GPU |

### Concurrency

512-token outputs, real token counts by `usage.completion_tokens`. Both nodes measured:

| Scenario | Node A: P2P disabled | Node B: P2P enabled |
|---|---:|---:|
| c1, short prompt, tok/s | 175 | 212.7 |
| c4, short prompt, tok/s | 268 | 413.9 |
| c8, short prompt, tok/s | 385 | 583.3 |
| c8, 6K shared prefix, tok/s | 436 | 485.8 |
| c8, 30K hot document, tok/s | 248 | 367.9 |
| c8 short-prompt TTFT, s | 0.5–0.6 | 0.57–0.73 |
| 30K second-visit TTFT, s | 0.79 | 0.63–0.73 |
| c8, 30K hot document TTFT, s | 2.1 | 1.66 |

Node A's c8 throughput across three rounds: 329 / 385 / 507 tok/s; node B retest range ~487.7–604.7 tok/s. Other request load was present during testing.

After node B raised max sequences from 4 to 8, aggregate throughput for the same 8-way load went from **340.7 to 583.3 tok/s (+71.2%)**, TTFT from **2.3–2.5 s to 0.57–0.73 s**.

[Node A deployment script](deployment/start-dsv41-a40-final.sh) · [Node B deployment script](deployment/start-dsv41-a40-p2p-final.sh) · [runtime launcher](deployment/start-dsv41-runtime-final.sh)

## Deployment

Download models from the ModelScope pages above and set `MODEL` to the download location. GLM uses the SM80 image's SIF file; DeepSeek uses the Docker image.

```bash
# GLM: start LMCache and the inference service together
MODEL=/models/GLM-5.3-Flash-W4A16 \
IMAGE=/images/vllm-backport-v0.13.1-sm80.sif \
LMCACHE_PACKAGE=/packages/lmcache-registration-fixed \
bash deployment/start-glm53-final.sh

# DeepSeek V4
MODEL=/models/DeepSeek-V4-Flash-0731 bash deployment/start-dsv4-final.sh

# DeepSeek V4.1, P2P-disabled node
MODEL=/models/DeepSeek-V4.1-Flash bash deployment/start-dsv41-a40-final.sh

# DeepSeek V4.1, P2P-enabled node
MODEL=/models/DeepSeek-V4.1-Flash bash deployment/start-dsv41-a40-p2p-final.sh
```

DeepSeek deployments start [LMCache](deployment/start-lmcache.sh) first:

```bash
# V4.1: 128GiB / chunk 1024 / port 5556
IMAGE=lazymio/vllm-backport:v0.13.1-sm86 bash deployment/start-lmcache.sh

# V4: 256GiB / chunk 256 / port 5555
IMAGE=lazymio/vllm-backport:v0.13.1-sm80 CACHE_PORT=5555 \
CACHE_CHUNK_SIZE=256 CACHE_SIZE_GB=256 CACHE_GPU_WORKERS=4 \
bash deployment/start-lmcache.sh
```

The patched package in the GLM example must be prepared per the [patch instructions](patches/lmcache-registration/README.md); if your image already contains an equivalent verified fix, the variable can be omitted. The API listens on `127.0.0.1` by default; set `API_HOST=0.0.0.0` to serve externally. When restarting A40 nodes, wait for old workers to exit and LMCache GPU IPC mappings to be released before starting the new engine.

## Benchmarks & diagnostics

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 benchmarks/bench_decode.py --api http://127.0.0.1:8000 \
  --model YOUR_MODEL --cs 1,2,4,8,16 --out-tokens 512 \
  --warmup-rounds 2 --rounds 5 --cache-mode cold \
  --output local-results/cold.json
```

`--cache-mode shared-prefix` measures shared-prefix reuse; `BENCH_API_KEY` supplies auth. Throughput = output tokens / total request time; TTFT is timed from the first non-empty content. More measurement details in the [benchmarking notes (Chinese)](docs/benchmarking.md).

| Topic | Entry point |
|---|---|
| CUDA graphs & speculative decoding | [Tuning notes (Chinese)](docs/cuda-graphs.md) |
| PCIe, NCCL & CustomAllreduce | [Communication notes (Chinese)](docs/communication.md), [diagnostics scripts](diagnostics/) |
| KV cache & concurrency | [Cache notes (Chinese)](docs/cache-and-concurrency.md) |
| Full experiment results | [results](results/README.md) |

```bash
python3 -m unittest discover -s tests -v
```

## Community

[LinuxDo](https://linux.do/) — 真诚、友善、团结、专业（中文技术社区）.
