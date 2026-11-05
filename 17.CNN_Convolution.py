import torch
import torch.nn as nn

# 1. 입력 이미지 생성
x = torch.tensor([
    [1., 2., 0., 1., 3.],
    [0., 1., 2., 2., 3.],
    [3., 1., 1., 0., 2.],
    [2., 0., 1., 3., 1.],
    [1., 2., 2., 1., 0.]
])

# 2. CNN 입력 형태로 변환
x = x.unsqueeze(0).unsqueeze(0)

# 3. Convolution Layer
conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

# 4. Kernel 값 직접 설정
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[
        [[1., 0., -1.],
         [1., 0., -1.],
         [1., 0., -1.]]
    ]]))

# 5. Forward Pass
output = conv(x)

# 6. result
print("Input shape:", x.shape)
print("Kernel shape:", conv.weight.shape)
print("Output shape:", output.shape)
print("Output:", output)