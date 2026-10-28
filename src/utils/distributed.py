"""Process group and collectives: setup / teardown, rank helpers, gathers, gradient and LoRA sync."""
import itertools
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    torch.cuda.set_device(lock_rank)
    timeout_seconds = os.getenv("NCCL_TIMEOUT_SECONDS")
    kwargs = {"rank": rank, "world_size": world_size}
    if timeout_seconds:
        kwargs["timeout"] = timedelta(seconds=int(timeout_seconds))
    dist.init_process_group("nccl", **kwargs)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def gather_prompts(prompts, world_size):
    """Every rank's prompts in rank-major order (the order of the gathered rewards)."""
    if world_size == 1:
        return list(prompts)
    nested = [None] * world_size
    dist.all_gather_object(nested, list(prompts))
    return list(itertools.chain.from_iterable(nested))


@torch.no_grad()
def sync_grads(optimizer, world_size):
    """Average the gradients of every parameter in ``optimizer`` over the ranks (one coalesced all-reduce).

    DDP gradient-sync fix (2026-10-25): per-rank grad / weight checksums of a 2-GPU run differed from the
    first optimizer step, so the trainers no longer rely on DDP's hooks alone; on
    ranks DDP already averaged, this averages identical tensors (a no-op up to rounding)."""
    if world_size <= 1:
        return
    grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not grads:
        return
    by_dtype = {}
    for g in grads:
        by_dtype.setdefault((g.device, g.dtype), []).append(g)
    for gs in by_dtype.values():
        flat = torch._utils._flatten_dense_tensors(gs)
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for g, synced in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
            g.copy_(synced)


@torch.no_grad()
def broadcast_lora(module, world_size, device):
    """Copy rank 0's LoRA adapter weights (every parameter whose name contains "lora_") to every rank.
    Parameters paged to CPU (Wan2.2's experts) go through a temporary copy on ``device`` (NCCL)."""
    if world_size <= 1:
        return
    for name, p in module.named_parameters():
        if "lora_" not in name:
            continue
        if p.data.device.type == "cuda":
            dist.broadcast(p.data, src=0)
        else:
            tmp = p.data.to(device)
            dist.broadcast(tmp, src=0)
            p.data.copy_(tmp.to(p.data.device))
