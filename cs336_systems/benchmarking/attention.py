import torch
from torch.utils.checkpoint import checkpoint
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

def run_attention():
    out = scaled_dot_product_attention(q, k, v)
    return out

# seq, d_model = 256, 16
# q = torch.randn((8, seq, d_model), device=device)
# k = torch.randn((8, seq, d_model), device=device)
# v = torch.randn((8, seq, d_model), device=device)
# print(f"seq, d_model = 256, 16 time: {get_time(run_attention)}")

# seq, d_model = 1024, 16
# q = torch.randn((8, seq, d_model), device=device)
# k = torch.randn((8, seq, d_model), device=device)
# v = torch.randn((8, seq, d_model), device=device)
# print(f"seq, d_model = 1024, 16 time: {get_time(run_attention)}")

# seq, d_model = 4096, 16
# q = torch.randn((8, seq, d_model), device=device)
# k = torch.randn((8, seq, d_model), device=device)
# v = torch.randn((8, seq, d_model), device=device)
# print(f"seq, d_model = 4096, 16 time: {get_time(run_attention)}")

# seq, d_model = 4096, 32
# q = torch.randn((8, seq, d_model), device=device)
# k = torch.randn((8, seq, d_model), device=device)
# v = torch.randn((8, seq, d_model), device=device)
# print(f"seq, d_model = 4096, 32 time: {get_time(run_attention)}")

# seq, d_model = 4096, 64
# q = torch.randn((8, seq, d_model), device=device)
# k = torch.randn((8, seq, d_model), device=device)
# v = torch.randn((8, seq, d_model), device=device)
# print(f"seq, d_model = 4096, 64 time: {get_time(run_attention)}")

def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):  # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    return t
    
seq, d_model = 4096, 32
q = torch.randn((8, seq, d_model), device=device, requires_grad=True)
k = torch.randn((8, seq, d_model), device=device, requires_grad=True)
v = torch.randn((8, seq, d_model), device=device, requires_grad=True)
total_size_bytes = 0
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = run_attention()
print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")

compile_atten = torch.compile(run_attention)
total_size_bytes = 0
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = compile_atten()
print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")
