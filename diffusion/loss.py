from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

from .base import Batch, Distribution, Loss

__all__ = ["Lambda", "Simple", "VLB"]


@dataclass
class Lambda(Loss[Distribution]):
    function: Callable[[Batch[Distribution]], Tensor]

    def compute(self, batch: Batch[Distribution]) -> Tensor:
        return self.function(batch)


@dataclass
class Simple(Loss[Distribution]):
    parameter: Literal["x", "epsilon"] = "epsilon"
    index = 0

    def compute(self, batch: Batch[Distribution]) -> Tensor:
        return torch.mean((getattr(batch, self.parameter) - batch.hat[self.index])**2)


class VLB(Loss[Distribution]):

    def compute(self, batch: Batch[Distribution]) -> Tensor:
        t = batch.t.view(-1, *(1,) * (batch.x.ndim - 2))
        return batch.q.dkl(batch.p).where(t > 1, batch.p.nll(batch.x)).sum()
