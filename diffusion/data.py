from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .base import Data


class Identity(Data):

    def encode(self, w: Tensor) -> Tensor:
        return w

    def decode(self, x: Tensor) -> Tensor:
        return x


@dataclass
class OneHot(Data):
    k: Optional[int] = None

    def __post_init__(self):
        assert self.k is not None
        self.i = torch.eye(self.k)

    def encode(self, w: Tensor) -> Tensor:
        self.i = self.i.to(w.device)
        return self.i[w]

    def decode(self, x: Tensor) -> Tensor:
        return x.argmax(-1)


@dataclass
class Embedding(Data):
    k: Optional[int] = None
    d: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.k is not None and self.d is not None
        self.embedding = nn.Embedding(self.k, self.d)

    def encode(self, w: Tensor) -> Tensor:
        return self.embedding(w)

    def decode(self, x: Tensor) -> Tensor:
        return torch.cdist(x, self.embedding.weight).argmin(-1)