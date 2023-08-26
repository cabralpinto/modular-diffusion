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
    dimension: Optional[int] = None

    def __post_init__(self):
        self.i = torch.eye(self.dimension)

    def encode(self, w: Tensor) -> Tensor:
        self.i = self.i.to(w.device) # TODO change
        return self.i[w]

    def decode(self, x: Tensor) -> Tensor:
        return x.argmax(-1)


@dataclass
class Embedding(Data):
    count: Optional[int] = None
    dimension: Optional[int] = None

    def __post_init__(self) -> None:
        self.embedding = nn.Embedding(self.count, self.dimension)

    def encode(self, w: Tensor) -> Tensor:
        return self.embedding(w)

    def decode(self, x: Tensor) -> Tensor:
        return torch.cdist(x, self.embedding.weight).argmin(-1)