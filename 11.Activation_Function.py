import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4) # Input:2, Output:4
        self.relu = nn.ReLU() # 비선형성으로 복잡한 패턴 학습시키기 위함
        self.fc2 = nn.Linear(4, 1) # Input:4, Output:1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = MyModel()

x = torch.tensor([[1.0, 2.0]])

output = model(x)

print("Model:")
print(model)

print("Input:", x)
print("Output:", output)

print("Input shape:", x.shape)
print("Output shape:", output.shape)