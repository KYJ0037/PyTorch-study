import torch

# 1. Tensor 생성
x = torch.tensor(3.0, requires_grad=True)



# 2. 첫 번째 역전파
y1 = x ** 2
y1.backward()

print("First backward:", x.grad)

# 3. 두 번째 역전파
y2 = x ** 2
y2.backward()

print("Second backward:", x.grad)

# 4. 세 번째 역전파
y3 = x ** 2
y3.backward()

print("Third backward:", x.grad)
# 결과처럼 기울기는 누적됨. 따라서 zero_grad()로 누적 방지 설정