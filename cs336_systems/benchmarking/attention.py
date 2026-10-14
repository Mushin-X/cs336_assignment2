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

def run_attention(q_, k_, v_):
    out = scaled_dot_product_attention(q_, k_, v_)
    return out

seq, d_model = 256, 16
q = torch.randn((8, seq, d_model), device=device)
k = torch.randn((8, seq, d_model), device=device)
v = torch.randn((8, seq, d_model), device=device)
print(f"seq, d_model = 256, 16 time: {get_time(run_attention(q, k, v))}")

seq, d_model = 1024, 16
q = torch.randn((8, seq, d_model), device=device)
k = torch.randn((8, seq, d_model), device=device)
v = torch.randn((8, seq, d_model), device=device)
print(f"seq, d_model = 1024, 16 time: {get_time(run_attention())}")

seq, d_model = 4096, 16
q = torch.randn((8, seq, d_model), device=device)
k = torch.randn((8, seq, d_model), device=device)
v = torch.randn((8, seq, d_model), device=device)
print(f"seq, d_model = 4096, 16 time: {get_time(run_attention())}")