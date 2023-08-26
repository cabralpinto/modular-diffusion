from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

import torch
from torch import Tensor

from .base import Batch, Distribution, Loss

__all__ = ["Lambda", "Simple", "VLB"]

D = TypeVar("D", bound=Distribution)


@dataclass
class Lambda(Loss[D]):
    function: Callable[[Batch[D]], Tensor]

    def compute(self, batch: Batch[D]) -> Tensor:
        return self.function(batch)


@dataclass
class Simple(Loss[Distribution]):
    parameter: Literal["x", "epsilon"] = "x"
    index = 0

    def compute(self, batch: Batch[Distribution]) -> Tensor:
        return torch.mean((getattr(batch, self.parameter) - batch.hat[self.index])**2)


class VLB(Loss[Distribution]):

    def compute(self, batch: Batch[Distribution]) -> Tensor:
        t = batch.t.view(-1, *(1,) * (batch.x.ndim - 1))
        return batch.q.dkl(batch.p).where(t > 1, batch.p.nll(batch.x)).mean()
