from dataclasses import dataclass

import torch
from torch import Tensor

from .base import Time


@dataclass
class Discrete(Time):

    def sample(self, steps: int, size: int) -> Tensor:
        return torch.randint(1, steps + 1, (size,))
