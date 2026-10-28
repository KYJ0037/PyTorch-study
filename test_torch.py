import torch
import torch.nn as nn

# 간단한 신경망 생성
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# 임의의 입력 데이터
x = torch.randn(1, 10)

# 신경망 실행
y = model(x)

print("PyTorch version:", torch.__version__)
print("Output:", y)
print("Device:", x.device)
