import torch
import torch.nn as nn


# 1. Input Feature Map
x = torch.tensor([
    [1., 3., 2., 4.],
    [5., 6., 1., 2.],
    [7., 2., 8., 3.],
    [4., 1., 5., 9.]
])

# 2. Conv2d / Pool2d 입력 형태로 변환
x = x.unsqueeze(0).unsqueeze(0)


# 3. Pooling Layers
max_pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)


# 4. Forward Pass
max_output = max_pool(x)
avg_output = avg_pool(x)


# 5. 결과 확인
print("Input shape:", x.shape)

print("\nMax Pooling:")
print(max_output)
print("Max Output shape:", max_output.shape)

print("\nAverage Pooling:")
print(avg_output)
print("Avg Output shape:", avg_output.shape)