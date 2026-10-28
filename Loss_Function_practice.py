import torch

# 1. 입력과 정답
x = torch.tensor(2.0)
target = torch.tensor(10.0)

# 2. 학습할 파라미터
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 3. 순전파
y = w * x + b

# 4. Loss 계산
loss = (y - target) ** 2

print("Prediction:", y)
print("Target:", target)
print("Loss:", loss)

# 5. 역전파
loss.backward()

# 6. Gradient 확인
print("w.grad:", w.grad)
print("b.grad:", b.grad)
