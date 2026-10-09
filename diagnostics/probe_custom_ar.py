# Final decisive probe: vLLM CustomAllreduce (forced on) + capture-in-graph behavior
# AND direct kernel-level benchmark at decode sizes 5,10,20,40,80,96 rows bf16.
# Also test warmup effects: custom AR first-call spins.
import os, torch, time, json
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world):
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']='29651'
    dist.init_process_group('gloo', rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    from vllm.distributed.parallel_state import init_distributed_environment
    init_distributed_environment(world_size=world, rank=rank,
        distributed_init_method='tcp://127.0.0.1:29652',
        local_rank=rank, backend='nccl')
    import vllm.distributed.parallel_state as ps
    import vllm.config as _vc
    class NS: pass
    pc = NS(); pc.data_parallel_size=1; pc.enable_elastic_ep=False; pc.use_all2all=False
    pc.enable_eplb=False; pc.enable_elastic_ep_alpha6=False; pc.data_parallel_master_ip=None
    pc.compiled_max_cudagraph_capture_size=None; pc.enable_dp_controller=False
    pc.data_parallel_external_lb=False; pc.pipeline_parallel_size=1
    pc.tensor_parallel_size=world; pc.prefill_context_parallel_size=1
    pc.decode_context_parallel_size=1
    class Wrap: pass
    w = Wrap(); w.parallel_config=pc; w.compilation_config=NS(); w.compilation_config.mode=0
    w.engram_config=None; w.model_config=None; w.scheduler_config=None
    _vc.get_current_vllm_config = lambda: w
    ps.get_current_vllm_config = lambda: w
    ps.initialize_model_parallel(tensor_model_parallel_size=world)
    from vllm.platforms import cuda as pcuda
    pcuda.CudaPlatform.is_fully_connected = classmethod(lambda cls, ids: True)
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    ca = CustomAllreduce(dist.group.WORLD, torch.device(f"cuda:{rank}"))
    assert not ca.disabled
    torch.manual_seed(7)
    out = {}
    for rows in (5, 10, 20, 40, 80, 96):
        x = torch.randn(rows, 4096, dtype=torch.bfloat16, device='cuda')
        for _ in range(50):
            r = ca.custom_all_reduce(x)   # includes spin-wait warmup
        torch.cuda.synchronize(); dist.barrier()
        # correctness
        ref = x.clone(); dist.all_reduce(ref); torch.cuda.synchronize()
        err = (r.float()-ref.float()).abs().max().item()
        assert err == 0.0, f'custom AR incorrect: max_err={err}'
        t0 = time.perf_counter(); N=1000
        for _ in range(N):
            r = ca.custom_all_reduce(x)
        torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/N
        out[f'{rows}'] = {'custom_us': round(dt*1e6,1), 'max_err': err}
    if rank==0: print('CUSTOM_AR_WARM', json.dumps(out, indent=1), flush=True)

    # Capture custom AR into a CUDA graph and replay (the production decode path)
    g = torch.cuda.CUDAGraph()
    x2 = torch.randn(80, 4096, dtype=torch.bfloat16, device='cuda')
    from vllm.distributed.parallel_state import graph_capture
    from vllm.distributed.parallel_state import get_tp_group
    with get_tp_group().graph_capture() as ctx, ca.capture():
        # warmup on the capture stream
        with torch.cuda.stream(ctx.stream):
            for _ in range(3):
                ca.custom_all_reduce(x2)
            torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=ctx.stream, capture_error_mode='thread_local'):
            res = ca.custom_all_reduce(x2)
    # Check the captured output against NCCL before benchmarking replay.
    g.replay()
    torch.cuda.synchronize()
    ref2 = x2.clone()
    dist.all_reduce(ref2)
    torch.cuda.synchronize()
    assert torch.equal(res, ref2), "captured custom AR output differs from NCCL"

    # replay benchmark
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); dist.barrier()
    t0=time.perf_counter(); N=1000
    for _ in range(N): g.replay()
    torch.cuda.synchronize()
    dt=(time.perf_counter()-t0)/N
    if rank==0: print('CUSTOM_AR_IN_GRAPH_80rows_us', round(dt*1e6,1), flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(worker, args=(4,), nprocs=4, join=True)
