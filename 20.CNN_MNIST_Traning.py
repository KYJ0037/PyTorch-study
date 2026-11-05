import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 2. MNIST Dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# CPU 실습을 위해 5000개만 사용
train_dataset = Subset(train_dataset, range(5000))

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# 3.CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 8, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return(x)

# 4. Model / Loss / Optimizer
model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 5. Training
epochs = 2

model.train()

for epoch in range(epochs):

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss 누적
        total_loss += loss.item() * images.size(0)

        # Accuracy 계산
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}]"
        f"Loss: {avg_loss:.4f}"
        f"Train Accuracy: {accuracy:.2f}%"
    )

# 6. 학습 종료
print("Training Completed!")
