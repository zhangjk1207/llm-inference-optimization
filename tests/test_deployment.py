"""Validate deployment arguments with command substitutes; never access a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).parents[1]


class DeploymentArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model = self.directory / "model with spaces"
        self.model.mkdir()
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.capture = self.directory / "commands.jsonl"
        mock = '''#!/usr/bin/env python3
import json, os, signal, sys, time
with open(os.environ['DEPLOY_TEST_CAPTURE'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if 'lmcache.cli.main' in sys.argv:
    if os.environ.get('DEPLOY_TEST_CACHE_EXIT'):
        time.sleep(1)
        sys.exit(7)
    signal.pause()
elif os.environ.get('DEPLOY_TEST_CACHE_EXIT'):
    with open(os.environ['DEPLOY_TEST_ENGINE_PID'], 'w') as f:
        f.write(str(os.getpid()))
    signal.pause()
'''
        for name in ("docker", "apptainer"):
            path = self.bin / name
            path.write_text(mock)
            path.chmod(0o755)
        ss = self.bin / "ss"
        ss.write_text("#!/bin/sh\necho 'LISTEN 0 0 127.0.0.1:5555 '\necho 'LISTEN 0 0 127.0.0.1:18556 '\n")
        ss.chmod(0o755)
        self.env = {
            **os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "MODEL": str(self.model), "CACHEDIR": str(self.directory / "cache"),
            "DEPLOY_TEST_CAPTURE": str(self.capture),
        }
        for key in ("IMAGE", "APPTAINER", "PORT", "MAX_NUM_SEQS", "NCCL_P2P_DISABLE", "LMCACHE_PACKAGE", "DEPLOY_TEST_CACHE_EXIT", "DEPLOY_TEST_ENGINE_PID"):
            self.env.pop(key, None)

    def run_script(self, name, expected_returncode=0, **extra):
        result = subprocess.run(
            ["bash", str(ROOT / "deployment" / name)], env={**self.env, **extra},
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, expected_returncode, result.stderr)
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_dsv4_final_graph_and_p2p(self):
        args = self.run_script("start-dsv4-final.sh")[0]
        self.assertIn(str(self.model) + ":/model:ro", args)
        self.assertIn("NCCL_P2P_DISABLE=1", args)
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "32")
        sizes = args[args.index("--cudagraph-capture-sizes") + 1:
                     args.index("--max-cudagraph-capture-size")]
        self.assertEqual([int(size) for size in sizes],
                         [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 120, 144, 168, 192])

    def test_a40_node_a_final_settings(self):
        args = self.run_script("start-dsv41-a40-final.sh")[0]
        self.assertIn("NCCL_P2P_DISABLE=1", args)
        self.assertIn("lazymio/vllm-backport:v0.13.1-sm86", args)
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "8")
        config = json.loads(args[args.index("--compilation-config") + 1])
        self.assertEqual(config["cudagraph_capture_sizes"], [6, 12, 18, 24, 30, 36, 42, 48])

    def test_a40_node_b_keeps_config_and_enables_p2p(self):
        args = self.run_script("start-dsv41-a40-p2p-final.sh")[0]
        self.assertIn("NCCL_P2P_DISABLE=0", args)
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "8")
        self.assertEqual(args[args.index("--port") + 1], "8000")

    def test_glm_final_settings_and_cache_cleanup(self):
        image = self.directory / "tested image.sif"
        image.touch()
        package = self.directory / "lmcache package"
        package.mkdir()
        (package / "__init__.py").touch()
        commands = self.run_script(
            "start-glm53-final.sh", expected_returncode=1, IMAGE=str(image),
            APPTAINER=str(self.bin / "apptainer"),
            LOGDIR=str(self.directory / "logs"), LMCACHE_PACKAGE=str(package),
        )
        engine = next(args for args in commands if "/opt-patch/serve_custom_ar.py" in args)
        for args in commands:
            self.assertIn(str(package) + ":/usr/local/lib/python3.12/dist-packages/lmcache:ro", args)
        self.assertEqual(engine[engine.index("--long-prefill-token-threshold") + 1], "1024")
        self.assertEqual(engine[engine.index("--max-num-batched-tokens") + 1], "1152")
        self.assertEqual(engine[engine.index("--max-model-len") + 1], "262144")
        self.assertIn("VLLM_FORCE_PCIE_CUSTOM_AR=1", engine)
        self.assertEqual(json.loads(engine[engine.index("--speculative-config") + 1])[
            "num_speculative_tokens"], 4)
        config = json.loads(engine[engine.index("--compilation-config") + 1])
        self.assertEqual(config["max_cudagraph_capture_size"], 80)
        self.assertIn(5, config["cudagraph_capture_sizes"])
        self.assertIn(10, config["cudagraph_capture_sizes"])

    def test_glm_cache_failure_stops_engine(self):
        image = self.directory / "image.sif"
        image.touch()
        pidfile = self.directory / "engine.pid"
        self.run_script(
            "start-glm53-final.sh", expected_returncode=7,
            IMAGE=str(image), APPTAINER=str(self.bin / "apptainer"),
            LOGDIR=str(self.directory / "logs"),
            DEPLOY_TEST_CACHE_EXIT="1", DEPLOY_TEST_ENGINE_PID=str(pidfile),
        )
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)


if __name__ == "__main__":
    unittest.main()
