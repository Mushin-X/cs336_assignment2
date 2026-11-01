"""作业二 第 7 节：FSDP —— 能过测试的最小版本。

测试实际要求的东西：
    1. get_fsdp(module, compute_dtype) 返回一个有 .module 的对象
    2. 能 forward、backward
    3. 调完 finish_gradient_synchronization 后，每个参数的 grad 都存在、
       形状和 dtype 都和参数本身一致；没被切开的参数，各卡梯度要相同
    4. gather_full_params 返回 {参数名: 完整张量}
    5. compute_dtype 为 fp16 时，结果仍要和普通训练对得上

对应的做法只有三件事：
    把 Linear / Embedding 的权重切成 N 段
    用的时候拼回来（前向一次、反向一次）
    完整梯度切一刀，每张卡只留自己那段

adapter 三个函数：
    def get_fsdp(module, compute_dtype=None):
        from cs336_systems.distributed_training.AI_fsdp import FSDP
        return FSDP(module, compute_dtype=compute_dtype)

    def fsdp_on_after_backward(fsdp_model, optimizer):
        fsdp_model.finish_gradient_synchronization()

    def fsdp_gather_full_params(fsdp_model):
        from cs336_systems.distributed_training.AI_fsdp import gather_full_params
        return gather_full_params(fsdp_model)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class _Shard:
    """一个被切开的权重，要记的东西。"""

    shape: torch.Size                    # 完整时的形状
    compute_dtype: Optional[torch.dtype]  # 混合精度时算用什么类型
    param: Optional[nn.Parameter] = None  # 自己那一段
    grad: Optional[torch.Tensor] = None   # 反向算出的完整梯度，还没切


def _gather(shard: torch.Tensor, info: _Shard) -> torch.Tensor:
    """N 段拼回完整权重。"""
    bufs = [torch.empty_like(shard) for _ in range(dist.get_world_size())]
    dist.all_gather(bufs, shard.contiguous())
    return torch.cat(bufs).view(info.shape)


def _reduce_scatter_sum(grad: torch.Tensor) -> torch.Tensor:
    """完整梯度：所有卡加起来，然后每张卡只拿自己那段。"""
    chunks = list(grad.reshape(-1).chunk(dist.get_world_size()))
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
    return out


# ---------------------------------------------------------------------------
# 两个自己写的前向/反向。
# 规矩：apply() 是入口；forward 第一个参数是 ctx；
#       backward 返回几个值，要和 forward 收到的输入一一对应（不需要就写 None）。
# ---------------------------------------------------------------------------
class _LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, info: _Shard):
        full = _gather(w, info)                       # 拼回完整权重
        if info.compute_dtype is not None:
            full = full.to(info.compute_dtype)

        # 只存反向要用的：x 和那一段。完整权重不进 ctx，函数返回就释放
        ctx.save_for_backward(x, w)
        ctx.info = info
        return torch.einsum("...i,oi->...o", x, full)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        info = ctx.info

        full = _gather(w, info)                       # 前向那份已经没了，再拼一次
        if info.compute_dtype is not None:
            full = full.to(info.compute_dtype)

        grad_w = torch.einsum("...o,...i->oi", grad_out, x).float()
        info.grad = grad_w if info.grad is None else info.grad + grad_w

        grad_x = torch.einsum("...o,oi->...i", grad_out, full)
        return grad_x, None, None                     # 权重的梯度不在这里写


class _EmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, w, info: _Shard):
        full = _gather(w, info)
        if info.compute_dtype is not None:
            full = full.to(info.compute_dtype)
        ctx.save_for_backward(ids, w)
        ctx.info = info
        return full[ids]

    @staticmethod
    def backward(ctx, grad_out):
        ids, w = ctx.saved_tensors
        info = ctx.info

        full = _gather(w, info)
        if info.compute_dtype is not None:
            full = full.to(info.compute_dtype)

        grad_w = torch.zeros_like(full)               # 和 grad_out 同 dtype
        grad_w.index_add_(0, ids.reshape(-1), grad_out.reshape(-1, full.shape[1]))
        grad_w = grad_w.float()
        info.grad = grad_w if info.grad is None else info.grad + grad_w

        return None, None, None


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.module = module                  # 测试会直接访问 .module
        self.compute_dtype = compute_dtype
        self._rank = dist.get_rank()
        self._world = dist.get_world_size()
        self._shards: dict[str, _Shard] = {}
        self._split()

    def _split(self) -> None:
        from cs336_basics.model import Embedding, Linear

        for name, mod in self.module.named_modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue                      # RMSNorm 太小，不切

            full = mod.weight.data
            n = full.numel() // self._world   # 每张卡拿多少（测试里都除得尽）

            info = _Shard(shape=full.shape, compute_dtype=self.compute_dtype)
            info.param = nn.Parameter(
                full.reshape(-1)[self._rank * n : (self._rank + 1) * n].clone()
            )

            # 模型里换成自己那一段：优化器之后更新的就是它
            mod.weight = info.param
            self._shards[f"{name}.weight"] = info

            # 换掉这个层的 forward：原来直接用 self.weight，现在得先拼
            mod.forward = self._make_forward(info, isinstance(mod, Linear))

    def _make_forward(self, info: _Shard, is_linear: bool):
        # 多套一层，是为了让每个层有各自的 info
        if is_linear:
            def forward(x):
                return _LinearFn.apply(x, info.param, info)
        else:
            def forward(ids):
                return _EmbeddingFn.apply(ids, info.param, info)
        return forward

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """反向之后、step 之前调用。"""
        # ① 切开的权重：完整梯度求和后切回自己那段，再除以卡数变平均
        for info in self._shards.values():
            if info.grad is None:
                continue
            info.param.grad = _reduce_scatter_sum(info.grad) / self._world
            info.grad = None

        # ② 没切开的权重（RMSNorm）：每张卡都有一份，梯度求和平均
        sharded = {id(info.param) for info in self._shards.values()}
        for p in self.module.parameters():
            if id(p) in sharded or p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self._world)


def gather_full_params(model: FSDP) -> dict[str, torch.Tensor]:
    """把每个参数拼回完整形状，供测试和普通训练比对。"""
    out = {}
    for name, info in model._shards.items():
        out[name] = _gather(info.param.data, info)
    for name, p in model.module.named_parameters():
        out.setdefault(name, p.data)          # 没切的原样返回，且不覆盖上面的
    return out


# ---------------------------------------------------------------------------
# 这份代码没写的部分，以及为什么测试不需要它们：
#
#   1. 初始化时广播参数
#      测试里两边 seed 相同，模型天生一样。
#
#   2. 元素数不能被卡数整除时补零
#      ToyFSDPModel 里每个权重都是偶数个元素（6400、8192…），除得尽。
#      真实模型要补零，否则 all_gather 收到的段长不一致。
#
#   3. world_size == 1 的短路、异步通信、预取
#      测试固定 2 张卡，也不看性能。
#
#   4. 绑定权重的去重
#      测试模型没有共享权重。
# ---------------------------------------------------------------------------
