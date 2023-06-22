from dataclasses import dataclass

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
    k: int = 2

    def __post_init__(self):
        self.i = torch.eye(self.k)

    def encode(self, w: Tensor) -> Tensor:
        return self.i[w]

    def decode(self, x: Tensor) -> Tensor:
        return x.argmax(-1)


@dataclass
class Embedding(Data):
    k: int = 2
    d: int = 256

    def __post_init__(self) -> None:
        self.embedding = nn.Embedding(self.k, self.d)

    @property
    def shape(self) -> tuple[int]:
        return (*self.x.shape[1:], self.d)

    def encode(self, w: Tensor) -> Tensor:
        return self.embedding(w)

    def decode(self, x: Tensor) -> Tensor:
        return torch.cdist(x, self.embedding.weight).argmin(-1)