"""Exercise the real HTTP/SSE parser without model weights or GPUs."""
import importlib.util
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest


spec = importlib.util.spec_from_file_location(
    "bench_decode", Path(__file__).parents[1] / "benchmarks/bench_decode.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
chat_spec = importlib.util.spec_from_file_location(
    "bench_shared_prefix", Path(__file__).parents[1] / "benchmarks/bench_shared_prefix.py")
chat_bench = importlib.util.module_from_spec(chat_spec)
chat_spec.loader.exec_module(chat_bench)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = body["prompt"]
        if mode == "http-error":
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        def send(event):
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
        if mode == "stream-error":
            send({"error": {"message": "failed"}})
        else:
            # Metadata is sent before content to detect false TTFT measurements.
            send({"choices": [], "usage": {"prompt_tokens": 12}})
            time.sleep(0.05)
            send({"choices": [{"text": "many tokens in one chunk"}]})
            if mode != "missing-usage":
                send({"choices": [], "usage": {"completion_tokens": 25}})
        self.wfile.write(b"data: [DONE]\n\n")


class StreamingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.api = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, mode):
        gate = threading.Event()
        gate.set()
        return bench.request_once(self.api, "model", mode, 25, gate)

    def test_tokens_come_from_usage_and_ttft_from_content(self):
        result = self.request("normal")
        self.assertTrue(result["ok"])
        self.assertEqual(result["completion_tokens"], 25)
        self.assertEqual(result["content_chunks"], 1)
        self.assertGreaterEqual(result["ttft_s"], 0.04)
        self.assertIsNone(result["cached_tokens"])

    def test_missing_usage_is_invalid(self):
        self.assertFalse(self.request("missing-usage")["ok"])

    def test_http_error_is_invalid(self):
        self.assertFalse(self.request("http-error")["ok"])

    def test_stream_error_is_invalid(self):
        self.assertFalse(self.request("stream-error")["ok"])

    def test_gate_wait_is_excluded_from_latency(self):
        gate = threading.Event()
        results = []
        worker = threading.Thread(target=lambda: results.append(
            bench.request_once(self.api, "model", "normal", 25, gate)))
        started = time.perf_counter()
        worker.start()
        time.sleep(0.2)
        gate.set()
        worker.join()
        self.assertTrue(results[0]["ok"])
        self.assertGreater(time.perf_counter() - started - results[0]["latency_s"], 0.17)

    def test_chat_missing_cache_count_stays_unknown(self):
        result = chat_bench.summarize(1, [{
            "ok": True, "ttft_s": 0.2, "latency_s": 0.5,
            "usage": {"prompt_tokens": 10, "prompt_tokens_details": None},
        }])
        self.assertEqual(result["prompt_tokens_min"], 10)
        self.assertIsNone(result["cached_tokens_min"])
        self.assertIsNone(result["cached_tokens_max"])

    def test_chat_failed_requests_are_not_latency_samples(self):
        result = chat_bench.summarize(2, [
            {"ok": True, "ttft_s": 0.2, "latency_s": 0.5, "usage": {}},
            {"ok": False, "ttft_s": None, "latency_s": 5.0, "error": "HTTP 500"},
        ])
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["ttft_p95_s"], 0.2)


if __name__ == "__main__":
    unittest.main()
