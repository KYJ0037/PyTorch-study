import torch
import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 1. Training Data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. Model
model = MyModel()

# 3. Loss Function
criterion = nn.MSELoss()

# 4. Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
# lr = learning rate

# 5. Traning
for epoch in range(100):
    prediction = model(x) # Forward pass, 현재의 weight과 bias값으로 계산

    loss = criterion(prediction, y)
    # Loss 계산, predict값과 정답의 차이를 숫자로 표현

    optimizer.zero_grad()
    # 이전 loop에 남아 있는 gradient 초기화

    loss.backward()
    # backpropagation, 각 parameter의 gradient 계산
    # weight.grad, bias.grad 등에 저장

    optimizer.step()
    # 계산된 gradient를 이용해서 실제 weight, bias값 없데이트

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():4f}")

# 6. Result
print("Weight:", model.linear.weight)
print("Bias:", model.linear.bias)