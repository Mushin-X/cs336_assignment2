import torch
from contextlib import nullcontext
from timeit import default_timer
from cs336_basics.model import CausalMultiHeadSelfAttention

def get_time(run):
    for _ in range(5):
        run()

    torch.cuda.synchronize()

    total_time = 0.0
    for _ in range(5):
        start = default_timer()
        run()
        torch.cuda.synchronize()
        end = default_timer()
        total_time += end - start

    return total_time / 5

device = 'cuda'

atten = CausalMultiHeadSelfAttention(
    d_model=16,
    num_heads=1
)

inputs = torch.randint(
    low=0,
    high=100,
    size=(8, 256, 16),
    device=device
)

def run_attention():
    out = CausalMultiHeadSelfAttention(inputs)
    return out

print(f"time: {get_time(run_attention)}")