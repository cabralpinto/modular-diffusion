import torch
from torch import Tensor


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)