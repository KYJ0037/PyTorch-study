import torch
import torch.nn as nn

# 1. 신경망 클래스 정의
class MyModel(nn.Module):

    # 2. 신경망 구조 정의
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    # 3. 순전파 정의
    def forward(self, x):
        y = self.linear(x)
        return y


# 4. 모델 객체 생성
model = MyModel()

# 5. 입력 데이터 생성
x = torch.tensor([[2.0]])

# 6. 순전파
y = model(x)

# 7. 결과 확인
print("Model:", model)
print("Input:", x)
print("Output:", y)

print("Weight:", model.linear.weight)
print("Bias:", model.linear.bias)