import torch

# 1. Tensor 생성
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("Tensor:",x)

# 2. Tensor의 형태와 차원
print("Shape:", x.shape)
print("차원 수:", x.ndim)
print("원소 수:", x.numel())
print("자료형:", x.dtype)
print("저장위치:", x.device)
