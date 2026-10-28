import torch

# 1. Tensor 생성
x = torch.tensor(2.0)

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 2. 순전파
y = w * x + b

print("y:", y)

# 3. 역전파
y.backward()

# 4. Gradient 확인
print("w.grad:", w.grad)
print("b.grad:", b.grad)
print("x.grad:", x.grad)