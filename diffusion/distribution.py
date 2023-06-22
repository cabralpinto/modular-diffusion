from dataclasses import dataclass

import torch
from torch import Tensor
from typing_extensions import Self

from .base import Distribution


@dataclass
class Normal(Distribution):
    mu: Tensor
    sigma: Tensor

    def sample(self) -> tuple[Tensor, Tensor]:
        epsilon = torch.randn(self.mu.shape, device=self.mu.device)
        return self.mu + self.sigma * epsilon, epsilon

    def nll(self, x: Tensor) -> Tensor:
        return (0.5 * ((x - self.mu) / self.sigma)**2 +
                (self.sigma * 2.5066282746310002).log())

    def dkl(self, other: Self) -> Tensor:
        return (torch.log(other.sigma / self.sigma) +
                (self.sigma**2 + (self.mu - other.mu)**2) / (2 * other.sigma**2) - 0.5)


@dataclass
class Categorical(Distribution):
    p: Tensor

    def __post_init__(self) -> None:
        self.k = self.p.shape[-1]
        self.i = torch.eye(self.k, device=self.p.device)

    def sample(self) -> tuple[Tensor, None]:
        index = torch.multinomial(self.p.view(-1, self.k), 1, True)
        return self.i[index.view(*self.p.shape[:-1])], None

    def nll(self, x: Tensor) -> Tensor:
        return -(self.p * x).sum(-1).log()

    def dkl(self, other: Self) -> Tensor:
        p1, p2 = self.p + 1e-6, other.p + 1e-6
        return (p1 * (p1.log() - p2.log())).sum(-1)
