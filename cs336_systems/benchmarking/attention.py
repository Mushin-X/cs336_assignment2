import torch
from contextlib import nullcontext
from timeit import default_timer
from cs336_basics.model import scaled_dot_product_attention

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

q = torch.randn((8, 256, 16), device=device)
k = torch.randn((8, 256, 16), device=device)
v = torch.randn((8, 256, 16), device=device)

def run_attention():
    out = scaled_dot_product_attention(q, k, v)
    return out

print(f"time: {get_time(run_attention)}")