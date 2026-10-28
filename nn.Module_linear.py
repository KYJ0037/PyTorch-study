import torch
import torch.nn as nn

# 1. 신경망 계층 생성
model = nn.Linear(1,1)

# 2. 입력 Tensor 생성
x = torch.tensor([[2.0]])

# 3. 순전파
y = model(x)

# 4. 결과 출력
print("Model:", model)
print("Input:", x)
print("Output:", y)

# 5. Weight와 Bias 확인
print("Weight:", model.weight)
print("Bias:", model.bias)

# 6. Gradient 추적 여부 확인
print("Weight requires_grad:", model.weight.requires_grad)
print("Bias requires_grad:", model.bias.requires_grad)