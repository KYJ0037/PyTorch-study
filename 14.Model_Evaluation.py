
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 1. Model 정의
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 2. Dataset / DataLoader
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

dataset = TensorDataset(x, y)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# 3. Model / Loss / Optimizer
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 4. Training
model.train()

for epoch in range(500):
    for batch_x, batch_y in dataloader:

        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 5. Evaluation Mode
model.eval()


# 6. Inference
new_x = torch.tensor([[5.0]])

with torch.no_grad():
    prediction = model(new_x)


# 7. 결과 확인
print("Input:", new_x.item())
print("Prediction:", prediction.item())
print("Expected:", 10.0)

print("Weight:", model.linear.weight.item())
print("Bias:", model.linear.bias.item())
