import torch
import torch.nn as nn
import torch.distributed as dist

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        self.broadcast_parameters()

    def broadcast_parameters(self):
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False) #async_op为 false 保证所有进程的计算顺序，使得在全规约操作执行完后，才会进行后续的平均操作

        for param in self.module.parameters():
            if param.grad is None:
                continue
            param.grad /= self.world_size

class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.broadcast_parameters()

        self.handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.hook)

    def hook(self, param):
        if param.grad is None:
            return 
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def broadcast_parameters(self):
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0) 

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for param in self.module.parameters():
            if param.grad is None:
                continue
            param.grad /= self.world_size