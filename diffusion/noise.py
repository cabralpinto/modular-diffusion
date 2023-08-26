from abc import abstractmethod
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal

import torch
from torch import Tensor

from .base import Noise
from .distribution import Categorical as Cat
from .distribution import Normal as N

__all__ = ["Gaussian", "Categorical"]


@dataclass
class Gaussian(Noise[N]):
    parameter: Literal["x", "epsilon", "mu"] = "x"
    variance: Literal["fixed", "range", "learned"] = "fixed"

    # TODO add lambda_ parameter to allow DDIM

    def schedule(self, alpha: Tensor) -> None:
        delta = alpha.cumprod(0)
        self.q1 = delta.sqrt()
        self.q2 = (1 - delta).sqrt()
        self.q3 = alpha.sqrt() * (1 - delta.roll(1)) / (1 - delta)
        self.q4 = delta.roll(1).sqrt() * (1 - alpha) / (1 - delta)
        self.q5 = ((1 - alpha) * (1 - delta.roll(1)) / (1 - delta)).sqrt()
        if self.parameter == "x":
            self.p1, self.p2 = self.q3, self.q4
        elif self.parameter == "epsilon":
            self.p1 = 1 / alpha.sqrt()
            self.p2 = (alpha - 1) / ((1 - delta).sqrt() * alpha.sqrt())
        else:
            self.p1, self.p2 = torch.zeros(alpha.shape), torch.ones(alpha.shape)
        self.p3 = (1 - alpha).log()
        self.p4 = self.q5.log()

    def isotropic(self, shape: tuple[int, ...]) -> N:
        return N(torch.zeros(shape), torch.ones(shape))

    def prior(self, x: Tensor, t: Tensor) -> N:
        t = t.view(-1, *(1,) * (x.dim() - 1))
        return N(self.q1[t] * x, self.q2[t])

    def posterior(self, x: Tensor, z: Tensor, t: Tensor) -> N:
        t = t.view(-1, *(1,) * (x.dim() - 1))
        return N(self.q3[t] * z + self.q4[t] * x, self.q5[t])

    def approximate(self, z: Tensor, t: Tensor, hat: Tensor) -> N:
        t = t.view(-1, *(1,) * (z.dim() - 1))
        # mu =  + self.p2[t] * hat[0]
        # print(z.min().item(), z.max().item(), flush=True)
        # print(self.p1[t].min().item(), self.p1[t].max().item(), flush=True)

        # x = (z - (1 - self.delta[t]) * hat[0]) / self.delta[t].sqrt()
        # print(x.min().item(), x.max().item(), flush=True)
        # if self.parameter == "epsilon":
        #     x = (z - (1 - self.delta[t]) * hat[0]) / self.delta[t].sqrt()
        # else:
        #     x = hat[0]
        # print(x.min().item(), x.max().item(), flush=True)

        return N(
            self.p1[t] * z + self.p2[t] * hat[0],
            self.q5[t]
            if self.variance == "fixed"
            else torch.exp(hat[1] * self.p3[t] + (1 - hat[1]) * self.p4[t])
            if self.variance == "range"
            else hat[1],
        )


class Categorical(Noise[Cat]):
    @abstractmethod
    def q(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def r(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def prior(self, x: Tensor, t: Tensor) -> Cat:
        return Cat(x @ self.r(t))

    def posterior(self, x: Tensor, z: Tensor, t: Tensor) -> Cat:
        return Cat(
            (z @ self.q(t).transpose(1, 2))
            * (x @ self.r(t - 1))
            / (x @ self.r(t) * z).sum(2, keepdim=True)
        )

    def approximate(self, z: Tensor, t: Tensor, hat: Tensor) -> Cat:
        return self.posterior(hat[0], z, t)


class MemoryInefficientCategorical(Categorical):
    def schedule(self, alpha: Tensor) -> None:
        self._q = self.transition(alpha)
        self._r = torch.stack([*accumulate(self._q, torch.mm)])

    @abstractmethod
    def transition(self, alpha: Tensor) -> Tensor:
        raise NotImplementedError

    def q(self, t: Tensor) -> Tensor:
        return self._q[t]

    def r(self, t: Tensor) -> Tensor:
        return self._r[t]


@dataclass
class MemoryEfficientCategorical(Categorical):
    k: int

    def schedule(self, alpha: Tensor) -> None:
        self.alpha = alpha.view(-1, 1, 1)
        self.delta = self.alpha.cumprod(0)
        self.i = torch.eye(self.k, device=alpha.device)[None]

    @property
    @abstractmethod
    def a(self) -> Tensor:
        raise NotImplementedError

    def q(self, t: Tensor) -> Tensor:
        return self.alpha[t] * self.i + (1 - self.alpha[t]) * self.a

    def r(self, t: Tensor) -> Tensor:
        return self.delta[t] * self.i + (1 - self.delta[t]) * self.a


class Uniform(MemoryEfficientCategorical):
    @property
    def a(self) -> Tensor:
        return torch.ones(self.k, self.k, device=self.i.device) / self.k

    def isotropic(self, shape: tuple[int, ...]) -> Cat:
        return Cat(torch.full(shape, 1 / self.k))


@dataclass
class Absorbing(MemoryEfficientCategorical):
    m: int = -1

    @property
    def a(self) -> Tensor:
        return self.i[:, self.m].repeat(1, self.k, 1)

    def isotropic(self, shape: tuple[int, ...]) -> Cat:
        return Cat(torch.eye(self.k)[self.m].repeat(*shape[:-1], 1))
