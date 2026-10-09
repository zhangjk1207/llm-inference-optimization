#!/usr/bin/env python3
"""Streaming completions benchmark adapted from the October GLM experiments."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import threading
import time
import uuid

import requests


def request_once(api, model, prompt, out_tokens, gate):
    body = {
        "model": model, "prompt": prompt, "max_tokens": out_tokens,
        "temperature": 0, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    headers = {}
    if os.environ.get("BENCH_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["BENCH_API_KEY"]
    gate.wait()
    started = time.perf_counter()
    first = None
    usage = {}
    chunks = 0
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.post(api.rstrip("/") + "/v1/completions", json=body,
                              headers=headers, stream=True, timeout=(30, 900)) as response:
                response.raise_for_status()
                for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    if event.get("error"):
                        raise ValueError("server returned a streaming error")
                    if isinstance(event.get("usage"), dict):
                        usage.update(event["usage"])
                    if any(choice.get("text") for choice in event.get("choices") or []):
                        chunks += 1
                        if first is None:
                            first = time.perf_counter() - started
        latency = time.perf_counter() - started
        tokens = usage.get("completion_tokens")
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("missing positive usage.completion_tokens; token rate unavailable")
        if first is None:
            raise ValueError("no non-empty output content in stream")
        details = usage.get("prompt_tokens_details") or {}
        return {
            "ok": True, "latency_s": latency, "ttft_s": first,
            "completion_tokens": tokens, "content_chunks": chunks,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens", usage.get("cached_tokens")),
        }
    except Exception as exc:
        # Do not echo request headers, URLs, generated content or server error bodies.
        return {"ok": False, "error": type(exc).__name__}


def run_round(args, concurrency, phase, index):
    gate = threading.Event()
    unit = "大模型推理的性能由计算、访存和通信共同决定。"
    base = unit * args.prompt_units
    nonce = uuid.uuid4().hex
    if args.cache_mode == "cold":
        prompts = [f"{nonce}:{i}\n" + base for i in range(concurrency)]
    else:
        prompts = [base + f"\n{nonce}:{i}" for i in range(concurrency)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_once, args.api, args.model, prompt,
                               args.out_tokens, gate) for prompt in prompts]
        # Workers record their clocks after this common release.
        gate.set()
        results = [future.result() for future in futures]
    good = [result for result in results if result["ok"]]
    valid = len(good) == concurrency
    row = {
        "phase": phase, "round": index, "concurrency": concurrency,
        "success": len(good), "errors": concurrency - len(good),
        "valid": valid, "requests": results,
    }
    if valid:
        wall = max(result["latency_s"] for result in good)
        row.update({
            "agg_tok_s": sum(result["completion_tokens"] for result in good) / wall,
            "per_req_tok_s_p50": statistics.median(
                result["completion_tokens"] / result["latency_s"] for result in good),
            "ttft_s_p50": statistics.median(result["ttft_s"] for result in good),
        })
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.environ.get("BENCH_API", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("BENCH_MODEL", "model"))
    parser.add_argument("--cs", default="1,2,4,8,16")
    parser.add_argument("--out-tokens", type=positive, default=512)
    parser.add_argument("--prompt-units", type=positive, default=100)
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--warmup-rounds", type=positive, default=2)
    parser.add_argument("--cache-mode", choices=["cold", "shared-prefix"], default="cold")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        concurrencies = [positive(value) for value in args.cs.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    summaries = []
    failed = False
    for concurrency in concurrencies:
        warmups = [run_round(args, concurrency, "warmup", index)
                   for index in range(args.warmup_rounds)]
        rows = [run_round(args, concurrency, "measure", index)
                for index in range(args.rounds)]
        valid = [row for row in rows if row["valid"]]
        failed |= len(valid) != len(rows) or any(not row["valid"] for row in warmups)
        summary = {
            "concurrency": concurrency, "rounds": args.rounds,
            "valid_rounds": len(valid), "cache_mode": args.cache_mode,
            "out_tokens": args.out_tokens,
        }
        if valid:
            rates = [row["agg_tok_s"] for row in valid]
            median = statistics.median(rates)
            summary.update({
                "agg_tok_s_p50": median, "agg_tok_s_min": min(rates),
                "agg_tok_s_max": max(rates),
                "spread_pct": (max(rates) - min(rates)) / median * 100,
                "ttft_p50_of_p50_s": statistics.median(row["ttft_s_p50"] for row in valid),
            })
        summaries.append(summary)
        print(json.dumps({"summary": summary}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
