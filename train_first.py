import torch
import torch.nn as nn
import torch.optim as optim

# 1. 재현 가능한 실험을 위한 난수 시드
torch.manual_seed(42)

# 2. 가상의 학습 데이터
# 입력 10개, 정답 2개
X = torch.randn(100, 10)
Y = torch.randn(100, 2)

# 3. 신경망 설계
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# 4. 손실 함수와 최적화 알고리즘
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. 학습 루프
for epoch in range(101):
    # 예측
    prediction = model(X)

    # 예측과 정답의 차이
    loss = criterion(prediction, Y)

    # 이전 기울기 초기화
    optimizer.zero_grad()

    # 역전파로 기울기 계산
    loss.backward()

    # 가중치 업데이트
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
