from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Conv2d, Module, Parameter
from torch.nn import Sequential as _Sequential
from torch.nn.functional import conv2d

__all__ = [
    "FastGELU",
    "Lambda",
    "WeightStdConv2d",
    "Swish",
    "SinusoidalPositionalEmbedding",
]


class Sequential(_Sequential):

    def forward(self, *input: Any) -> Any:
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


class Lambda(Module):

    def __init__(self, function: Callable[..., Any]):
        super().__init__()
        self.function = function

    def forward(self, *input: Any):  # type: ignore
        return self.function(*input)


class WeightStdConv2d(Conv2d):

    def forward(self, input: Tensor) -> Tensor:
        mean = self.weight.mean((1, 2, 3), keepdim=True)
        var = self.weight.var((1, 2, 3), unbiased=False, keepdim=True)
        return conv2d(
            input,
            (self.weight - mean) * var.rsqrt(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SinusoidalPositionalEmbedding(Module):

    def __init__(self, size: int = 32, base: float = 1e4) -> None:
        super().__init__()
        self.w = Parameter(base**torch.arange(0, -1, -2 / size)[None])

    def forward(self, t: Tensor) -> Tensor:
        wt = self.w * t[:, None]
        return torch.stack((wt.sin(), wt.cos()), 2).flatten(1)


class FastGELU(Module):

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class Swish(Module):

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)