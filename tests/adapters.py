from __future__ import annotations

import torch

def get_flashattention_autograd_function_pytorch() -> type:
    from cs336_systems.flashattention.flashattention_pytorch import FlashattentionPytorch
    return FlashattentionPytorch

def get_flashattention_autograd_function_triton() -> type:
    from cs336_systems.flashattention.flashattention_triton_with_mask import FlashattentionTriton
    return FlashattentionTriton

def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    # from cs336_systems.distributed_training.ddp import NaiveDDP
    # naiveDDP = NaiveDDP(module)
    # return naiveDDP

    from cs336_systems.distributed_training.ddp import DDPOverlapIndividualParameters

    ddPOverlapIndividualParameters = DDPOverlapIndividualParameters(module)
    return ddPOverlapIndividualParameters


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    # For example: return FSDP(module, compute_dtype=compute_dtype)
    raise NotImplementedError


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    # For example: fsdp_model.finish_gradient_synchronization()
    raise NotImplementedError


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    from cs336_systems.distributed_training.sharded_optimizer import ShardedOptimizer
    shardedOptimizer = ShardedOptimizer(params, optimizer_cls, **kwargs)
    return shardedOptimizer
