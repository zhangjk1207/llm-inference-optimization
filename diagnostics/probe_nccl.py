#!/usr/bin/env python3
"""4-GPU NCCL all_reduce probe with NCCL auto-selection (no ALGO/PROTO pins).

Mirrors the production env (SOCKET_IFNAME=lo, IB disabled, CUMEM off) but
leaves NCCL free to choose algorithm and protocol, which is what the glm53
canary wants to validate before removing the Ring/Simple pins.

Exit 0 = all collectives completed; nonzero/timeout = the auto mode hangs on
this topology and the canary must keep the pinned transport.
"""
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    # torchrun sets both; LOCAL_RANK is the within-node device index.
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    if rank == 0:
        print(f"nccl init ok on {socket.gethostname()}", flush=True)

    # Small sizes dominate decode-time TP all_reduce; large sizes exercise
    # bandwidth paths. 20 reps each to catch intermittent channel hangs.
    sizes = [8 * 1024, 64 * 1024, 1024 * 1024, 64 * 1024 * 1024]
    for nbytes in sizes:
        n = nbytes // 4
        x = torch.full((n,), float(rank + 1), dtype=torch.float32, device="cuda")
        expected = world * (world + 1) / 2
        # Re-fill before each rep: repeated all_reduce on the same buffer
        # compounds (10 -> 40 -> 160 ...), which is correct NCCL behavior,
        # not corruption. Fresh fill per rep is the honest check.
        t0 = time.monotonic()
        for _ in range(20):
            x.fill_(float(rank + 1))
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
        # Check several elements, not just [0]; and print a few raw values
        # on mismatch to distinguish transport corruption from init issues.
        vals = x[:4].tolist()
        value = vals[0]
        ok = all(abs(v - expected) < 1e-3 for v in vals)
        if rank == 0:
            print(
                f"size={nbytes//1024}KiB x20 ok={ok} first4={[round(v,1) for v in vals]} "
                f"total={dt*1000:.1f}ms",
                flush=True,
            )
        if not ok:
            raise SystemExit(f"rank{rank}: wrong all_reduce values {vals}")

    dist.barrier()
    if rank == 0:
        print("NCCL_AUTO_PROBE_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
