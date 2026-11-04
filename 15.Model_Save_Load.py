
import torch
import torch.nn as nn
import torch.optim as optim


# 1. Model 정의
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 2. Training Dataset
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])


# 3. Model / Loss / Optimizer
model = MyModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 4. Training
model.train()

for epoch in range(1000):
    prediction = model(x)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 5. Model 저장
torch.save(model.state_dict(), "model.pth")

print("Model saved!")


# 6. 새로운 Model 생성
loaded_model = MyModel()


# 7. 저장된 Weight / Bias 불러오기
state = torch.load("model.pth", weights_only=True)
loaded_model.load_state_dict(state)


# 8. Evaluation Mode
loaded_model.eval()


# 9. Inference
new_x = torch.tensor([[5.0]])

with torch.no_grad():
    original_prediction = model(new_x)
    loaded_prediction = loaded_model(new_x)


# 10. 결과 확인
print("Original Prediction:", original_prediction.item())
print("Loaded Prediction:", loaded_prediction.item())

print("Loaded Weight:", loaded_model.linear.weight.item())
print("Loaded Bias:", loaded_model.linear.bias.item())
