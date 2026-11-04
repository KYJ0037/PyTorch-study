import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 모델 정의
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 2. 학습 데이터 준비
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 3. Dataset / DataLoader 생성
dataset = TensorDataset(x, y)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# 4. Model / Loss / Optimizer
model = MyModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Traning Loop
for epoch in range(100):

    for batch_x, batch_y in dataloader:

        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

# 6. 학습 결과 확인
print("Weight:", model.linear.weight)
print("Bias:", model.linear.bias)