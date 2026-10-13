import torch
from contextlib import nullcontext
from timeit import default_timer
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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

model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=512,
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=128,
).to(device=device)

inputs = torch.randint(
    low=0,
    high=100,
    size=(2, 32),
    device=device
)
targets = torch.randint(
    low=0,
    high=100,
    size=(2, 32),
    device=device
)

optimizer = AdamW(model.parameters())

def only_forward():
    model(inputs)

def forward_and_backward(use_bf16=False):
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with context:
        out = model(inputs)
        loss = cross_entropy(out, targets)

    optimizer.zero_grad()
    loss.backward()

def forward_and_backward_and_optimizer():
    forward_and_backward()
    optimizer.step()


print(f"only_forward: {get_time(only_forward)}")
print(f"forward_and_backward (FP32): {get_time(forward_and_backward)}")
print(f"forward_and_backward (BF16): {get_time(lambda: forward_and_backward(use_bf16=True))}")
print(f"forward_and_backward_and_optimizer: {get_time(forward_and_backward_and_optimizer)}")
