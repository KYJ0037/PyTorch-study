import torch

# 1. Tensor 생성
x = torch.tensor(3.0, requires_grad=True)

# 2. Optimizer 생성
optimizer = torch.optim.SGD([x], lr=0.1)

# 3. 첫 번째 역전파
optimizer.zero_grad() # 이전 Gradient 초기화

y1 = x ** 2
y1.backward() # 현재 Gradient 계산
              #그러나 optimizer.step()을 사용하지 않아 파라미터 업데이트 안됨

print("First backward:", x.grad)

# 4. 두 번째 역전파
optimizer.zero_grad()

y2 = x ** 2
y2.backward()

print("Second backward:", x.grad)

# 5. 세 번째 역전파
optimizer.zero_grad()

y3 = x ** 2
y3.backward()

print("Third backward:", x.grad)