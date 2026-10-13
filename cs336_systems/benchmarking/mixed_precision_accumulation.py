import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"fc1前: {x.dtype}")
        x = self.fc1(x)
        print(f"fc1后 | relu前: {x.dtype}")
        x = self.relu(x)
        print(f"relu后 | ln前: {x.dtype}")
        x = self.ln(x)
        print(f"ln后 | fc2前: {x.dtype}")
        x = self.fc2(x)
        print(f"fc2后: {x.dtype}")
        return x

model = ToyModel(4, 3).cuda()
x = torch.randn(2, 4, device="cuda")
target = torch.randint(0, 3, (2,), device="cuda")

with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)

print("参数:", model.fc1.weight.dtype)
print("logits:", logits.dtype)
print("损失:", loss.dtype)

loss.backward()  # 放在 autocast 外
print("梯度:", model.fc1.weight.grad.dtype)