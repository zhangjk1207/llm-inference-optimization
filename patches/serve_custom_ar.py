#!/usr/bin/env python3
"""Opt-in experiment for the tested vLLM backport on four PCIe A800 GPUs.

Derived from opt20261006/patches/serve_custom_ar.py. This uses private APIs;
see docs/communication.md for the topology and validation requirements.
"""
import os
import sys


def patch_fully_connected():
    from vllm.platforms import cuda as cuda_platform

    patched = []
    for name in ("CudaPlatformBase", "CudaPlatform", "NvmlCudaPlatform"):
        candidate = getattr(cuda_platform, name, None)
        if candidate is not None and hasattr(candidate, "is_fully_connected"):
            candidate.is_fully_connected = classmethod(lambda cls, ids: True)
            patched.append(name)
    if not patched:
        raise RuntimeError("tested vLLM platform API not found; refusing to force custom AR")
    print(f"[serve_custom_ar] experimental topology override: {patched}", file=sys.stderr)


def main():
    if os.environ.get("VLLM_FORCE_PCIE_CUSTOM_AR") == "1":
        patch_fully_connected()
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
