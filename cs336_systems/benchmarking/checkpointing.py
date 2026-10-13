import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 768, 3072, 12, 12
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
    positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)

# Now log the number of bytes saved
total_size_bytes = 0
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

def run(x, t):
    for _ in range(t):
        x = block(x)
    return x 

def blocks_checkpoint(x, n, t):
    if n % t != 0:
        raise ValueError("n必须为t的整数倍")
    for _ in range(n / t):
        x = checkpoint(run, x, use_reentrant=False)
    return x    

# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = blocks_checkpoint(x, 8, 1)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")