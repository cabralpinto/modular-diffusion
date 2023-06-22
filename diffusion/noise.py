from dataclasses import dataclass
# from itertools import accumulate
from typing import Literal, Optional

import torch
from torch import Tensor

from .base import Noise
from .distribution import Normal as N

__all__ = ["Normal"]


@dataclass
class Normal(Noise[N]):
    parameter: Optional[Literal["x", "epsilon"]] = None
    variance: Literal["fixed", "range", "learned"] = "fixed"

    def schedule(self, alpha: Tensor) -> None:
        cumprod = alpha.cumprod(0)
        self.q1 = cumprod.sqrt()
        self.q2 = (1 - cumprod).sqrt()
        self.q3 = alpha.sqrt() * (1 - cumprod.roll(1)) / (1 - cumprod)
        self.q4 = cumprod.roll(1).sqrt() * (1 - alpha) / (1 - cumprod)
        self.q5 = ((1 - alpha) * (1 - cumprod.roll(1)) / (1 - cumprod)).sqrt()
        if self.parameter == "x":
            self.p1, self.p2 = self.q3, self.q4
        elif self.parameter == "epsilon":
            self.p1 = 1 / alpha.sqrt()
            self.p2 = (alpha - 1) / ((1 - cumprod).sqrt() * alpha.sqrt())
        else:
            self.p1, self.p2 = torch.zeros(alpha.shape), torch.ones(alpha.shape)
        self.p3 = (1 - alpha).log()
        self.p4 = self.q5.log()

    def isotropic(self, *shape: int) -> N:
        return N(torch.zeros(shape), torch.ones(shape))

    def prior(self, x: Tensor, t: Tensor) -> N:
        t = t.view(-1, *(1,) * (x.dim() - 1))
        return N(self.q1[t] * x, self.q2[t])

    def posterior(self, x: Tensor, z: Tensor, t: Tensor) -> N:
        t = t.view(-1, *(1,) * (x.dim() - 1))
        return N(self.q3[t] * z + self.q4[t] * x, self.q5[t])

    def approximate(self, z: Tensor, t: Tensor, hat: Tensor) -> N:
        t = t.view(-1, *(1,) * (z.dim() - 1))
        return N(
            self.p1[t] * z + self.p2[t] * hat[0],
            self.q5[t] if self.variance == "fixed" else
            torch.exp(hat[1] * self.p3[t] + (1 - hat[1]) * self.p4[t])
            if self.variance == "range" else hat[1],
        )
