from dataclasses import dataclass

import torch
from torch import Tensor

from .base import Schedule

__all__ = ["Constant", "Linear", "Cosine", "Sqrt"]


@dataclass
class Constant(Schedule):
    value: float

    def compute(self) -> Tensor:
        return torch.full((self.steps + 1,), self.value)


@dataclass
class Linear(Schedule):
    start: float
    end: float

    def compute(self) -> Tensor:
        return torch.linspace(self.start, self.end, self.steps + 1)


@dataclass
class Cosine(Schedule):
    offset: float = 8e-3
    exponent: float = 2

    def compute(self) -> Tensor:
        t = torch.arange(self.steps + 2)
        delta = ((t / (self.steps + 2) + self.offset) / (1 + self.offset) * torch.pi /
                   2).cos()**self.exponent
        alpha = torch.clip(delta[1:] / delta[:-1], 1e-3, 1)
        # TODO check if this is correct
        return alpha


@dataclass
class Sqrt(Schedule):
    offset: float = 8e-3

    def compute(self) -> Tensor:
        t = torch.arange(self.steps + 2)
        delta = 1 - torch.sqrt(t / self.steps + self.offset)
        alpha = torch.clip(delta[1:] / delta[:-1], 0, 0.999)
        return alpha
