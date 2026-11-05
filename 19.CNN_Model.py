import torch
import torch.nn as nn

# 1. CNN Model 정의
class CNNMOdel(nn.Module):
    def __init__(self):
        super().__init__()

        # feature Extraction
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Flatten
        self.flatten = nn.Flatten()

        # Classification
        self.fc = nn.Linear(8 * 13 * 13, 10)
    
    def forward(self, x):

        print("Input:", x.shape)

        x = self.conv(x)
        print("After Conv2d:", x.shape)

        x = self.relu(x)
        print("After ReLU:", x.shape)

        x = self.pool(x)
        print("After Pooling:", x.shape)

        x = self.flatten(x)
        print("After Flatten:", x.shape)

        x = self.fc(x)
        print("After Linear:", x.shape)

        return x

# 2. Model 생성

model = CNNMOdel()

# 3. 가상의 흑백 이미지 생성
x = torch.randn(1, 1, 28, 28)

# 4. Inference
model.eval()

with torch.no_grad():
    output = model(x)

# 5. Result
print("\nFianl Output Shape:", output.shape)
print("Final Output:", output)