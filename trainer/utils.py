from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {
            key: _move_batch_to_device(value, device) for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]

    return batch


def _gather_tensor_across_ranks(tensor: torch.Tensor, global_rank: int):
    if not dist.is_initialized():
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    if global_rank == 0:
        return torch.cat(gathered, dim=0)
    return None


def _watch_model_with_wandb(wandb_run, watched: bool, model, log_freq: int) -> bool:
    """Register wandb model watching.

    NOTE: ``log="all"`` is intentionally avoided because wandb registers
    gradient hooks on every parameter.  When some parameters are not part of
    the compute graph (e.g. ``depth=[4,0]`` leaves an entire stage unused),
    their gradients are ``None`` and wandb crashes with
    ``AttributeError: 'NoneType' object has no attribute 'data'``.
    Using ``log=None`` still tracks the computational graph without the
    fragile per-parameter gradient hooks.
    """
    if wandb_run is None or watched:
        return watched
    import wandb

    wandb.watch(model, log=None, log_freq=log_freq, log_graph=False)
    return True


def _resolve_autocast_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision_dtype: {dtype_name}")
