import torch
import torch.distributed as dist
from typing import Any, Type

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.my_optimizer = None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.total_params = []

        #该 super().__init__ 函数会调用下面的add_param_group，而其中又会用到上面变量，因此这里必须要先创建变量，再调用父类初始化
        super().__init__(params, {})

    def add_param_group(self, param_group: dict[str, Any]):
        #复制一份，使后续操作不影响父类传过来的原参数组
        params = list(param_group["params"])

        start = len(self.total_params)
        self.total_params.extend(params)

        my_params = [p for i, p in enumerate(params) if (start + i) % self.world_size == self.rank]
        super().add_param_group({**param_group, "params": my_params}) #第二个参数会覆盖第一个参数中的"params"键值

        if my_params is None:
            return 
        
        my_param_group = {**param_group, "params": my_params}
        if self.my_optimizer is None:
            self.my_optimizer = self.optimizer_cls([my_param_group], **self.kwargs)
        else :
            self.my_optimizer.add_param_group(my_param_group)

    def step(self, closure=None, **kwargs):
        if self.my_optimizer is not None:
            self.my_optimizer.step(**kwargs)

        for i, param in enumerate(self.total_params):
            dist.broadcast(param.data, src=i % self.world_size)