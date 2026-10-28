import torch

# 1. 입력 데이터와 정답
x = torch.tensor(2.0)
target = torch.tensor(10.0)

# 2. 학습 파라미터
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 3. Optimizer 설정
optimizer = torch.optim.SGD([w, b], lr=0.1)

# 4. 순전파
y = w * x + b

# 5. Loss 계산
loss = (y - target) ** 2

print("Before Update")
print("Prediction:", y.item())
print("Loss:", loss.item())

# 6. Gradient 초기화
optimizer.zero_grad()

# 7. 역전파
loss.backward()

print("w.grad:", w.grad.item())
print("b.grad:", b.grad.item())

# 8. 파라미터 업데이트
optimizer.step()

print("\nAfter Update")
print("w:", w.item())
print("b:", b.item())

# 9. 업데이트된 파라미터로 재예측
new_y = w * x + b
new_loss = (new_y - target) ** 2

print("New Prediction:", new_y.item())
print("New Loss:", new_loss.item())
