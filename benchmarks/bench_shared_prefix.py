#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import threading
import time
from typing import Any

import requests


def request_once(url: str, key: str, common: str, request_id: str, unique_lines: int, gate: threading.Event) -> dict[str, Any]:
    unique = []
    for i in range(unique_lines):
        unique.append(
            f"请求{request_id}的独立历史第{i:04d}条：本段内容只属于这个请求，包含不同的上下文、事实、时间和操作记录。"
        )
    content = common + "\n" + "\n".join(unique) + f"\n请求{request_id}结束，请只输出 OK。"
    body = {
        "model": os.environ.get("BENCH_MODEL", "model"),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    gate.wait()
    started = time.perf_counter()
    result: dict[str, Any] = {"id": request_id, "ok": False}
    try:
        session = requests.Session()
        session.trust_env = False
        with session.post(url, headers=headers, json=body, stream=True, timeout=(20, 240)) as response:
            result["http"] = response.status_code
            response.raise_for_status()
            first_token = None
            usage = None
            error_text = None
            chunks = 0
            for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                chunks += 1
                data = raw[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    error_text = str(chunk["error"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        if first_token is None:
                            first_token = time.perf_counter()
                        break
            finished = time.perf_counter()
            result["ok"] = response.status_code == 200 and error_text is None and first_token is not None
            result["ttft_s"] = first_token - started if first_token is not None else None
            result["latency_s"] = finished - started
            result["usage"] = usage or {}
            if error_text:
                result["error"] = error_text[:500]
            elif not result["ok"]:
                result["error"] = f"HTTP {response.status_code}; chunks={chunks}"
    except Exception as exc:
        result["latency_s"] = time.perf_counter() - started
        result["ttft_s"] = None
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return result


def run_batch(url: str, key: str, common: str, unique_lines: int, concurrency: int, label: str) -> list[dict[str, Any]]:
    gate = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_once, url, key, common, f"{label}-{i:04d}", unique_lines, gate) for i in range(concurrency)]
        time.sleep(0.2)
        gate.set()
        return [f.result() for f in futures]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 3)


def summarize(level: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    good = [r for r in results if r.get("ok")]
    errors: dict[str, int] = {}
    for r in results:
        if not r.get("ok"):
            errors[r.get("error", "unknown")] = errors.get(r.get("error", "unknown"), 0) + 1
    usages = [r.get("usage", {}) for r in good]
    cached = [(u.get("prompt_tokens_details") or {}).get("cached_tokens", u.get("cached_tokens")) for u in usages]
    cached = [n for n in cached if isinstance(n, int)]
    prompts = [u["prompt_tokens"] for u in usages if isinstance(u.get("prompt_tokens"), int)]
    ttfts = [float(r["ttft_s"]) for r in good]
    latencies = [float(r["latency_s"]) for r in good]
    return {
        "concurrency": level,
        "total": len(results),
        "success": len(good),
        "errors": len(results) - len(good),
        "success_rate": round(len(good) / len(results), 3) if results else 0,
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
        "latency_p95_s": percentile(latencies, 0.95),
        "prompt_tokens_min": min(prompts) if prompts else None,
        "prompt_tokens_max": max(prompts) if prompts else None,
        "cached_tokens_min": min(cached) if cached else None,
        "cached_tokens_max": max(cached) if cached else None,
        "error_samples": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--key", default=os.environ.get("NEWAPI_KEY", ""))
    parser.add_argument("--common-repeats", type=int, default=2000)
    parser.add_argument("--unique-lines", type=int, default=1000)
    parser.add_argument("--levels", default="1,2,4,8")
    args = parser.parse_args()
    if not args.key:
        raise SystemExit("set NEWAPI_KEY or pass --key")

    common_unit = "这是同一个 agent 的共享背景信息，用于模拟所有请求都相同的长期上下文。"
    common = common_unit * args.common_repeats
    print(json.dumps({"common_chars": len(common), "unique_lines": args.unique_lines, "levels": args.levels}, ensure_ascii=False), flush=True)
    warmup = run_batch(args.url, args.key, common, args.unique_lines, 2, "warmup")
    print(json.dumps({"warmup": summarize(2, warmup)}, ensure_ascii=False), flush=True)
    for text in args.levels.split(","):
        level = int(text)
        results = run_batch(args.url, args.key, common, args.unique_lines, level, f"c{level}")
        print(json.dumps(summarize(level, results), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
