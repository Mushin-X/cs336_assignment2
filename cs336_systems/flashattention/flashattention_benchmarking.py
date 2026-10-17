import torch
from timeit import default_timer
from cs336_systems.flashattention.flashattention_pytorch import FlashattentionPytorch
from cs336_systems.flashattention.flashattention_triton import FlashattentionTriton

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

def flashattention_pytorch():
    FlashattentionPytorch.apply(q, k, v)

def flashattention_triton():
    FlashattentionTriton.apply(q, k, v)

device = 'cuda'
seq, d_model = 512, 32
q = torch.randn((8, seq, d_model), device=device, requires_grad=True)
k = torch.randn((8, seq, d_model), device=device, requires_grad=True)
v = torch.randn((8, seq, d_model), device=device, requires_grad=True)

print(f"flashattention_pytorch: {get_time(flashattention_pytorch)}")
print(f"flashattention_triton: {get_time(flashattention_triton)}")
