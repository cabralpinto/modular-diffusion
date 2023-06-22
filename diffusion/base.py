from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import Self

__all__ = ["Batch", "Data", "Distribution", "Loss", "Net", "Noise", "Schedule", "Time"]


class Distribution(ABC):

    @abstractmethod
    def sample(self) -> tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def nll(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def dkl(self, other: Self) -> Tensor:
        raise NotImplementedError


D = TypeVar("D", bound=Distribution, covariant=True)


@dataclass
class Batch(Generic[D]):
    device: torch.device
    w: Tensor = field(init=False)
    x: Tensor = field(init=False)
    y: Tensor = field(init=False)
    t: Tensor = field(init=False)
    epsilon: Optional[Tensor] = field(init=False)
    z: Tensor = field(init=False)
    hat: Tensor = field(init=False)
    q: D = field(init=False)
    p: D = field(init=False)
    l: Tensor = field(init=False)

    def __setattr__(self, prop: str, val: Any):
        # TODO test difference in speed if only doing necessary conversions
        if isinstance(val, Tensor):
            if hasattr(self, prop):
                del self.__dict__[prop]
            val = val.to(self.device)
        super().__setattr__(prop, val)


@dataclass
class Data(ABC):
    x: Tensor
    y: Optional[Tensor] = None
    batch: int = 1
    shuffle: bool = False

    @property
    def shape(self) -> tuple[int]:
        return self.x.shape[1:]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        if self.y is None:
            self.y = torch.zeros(self.x.shape[0], dtype=torch.int)
        if self.shuffle:
            index = torch.randperm(self.x.shape[0])
            self.x, self.y = self.x[index], self.y[index]
        self.data = zip(self.x.split(self.batch), self.y.split(self.batch))
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        return next(self.data)

    def __len__(self) -> int:
        return self.x.shape[0] // self.batch

    @abstractmethod
    def encode(self, w: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Time(ABC):

    @abstractmethod
    def sample(self, steps: int, size: int) -> Tensor:
        raise NotImplementedError

    def __call__(self, steps: int, size: int) -> Tensor:
        return self.sample(steps, size)


@dataclass
class Schedule(ABC):
    steps: int

    @abstractmethod
    def compute(self) -> Tensor:
        raise NotImplementedError


@dataclass
class Noise(ABC, Generic[D]):

    @abstractmethod
    def schedule(self, alpha: Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def isotropic(self, *shape: int) -> D:
        raise NotImplementedError

    @abstractmethod
    def prior(self, x: Tensor, t: Tensor) -> D:
        raise NotImplementedError

    @abstractmethod
    def posterior(self, x: Tensor, z: Tensor, t: Tensor) -> D:
        raise NotImplementedError

    @abstractmethod
    def approximate(self, z: Tensor, t: Tensor, hat: Tensor) -> D:
        raise NotImplementedError


class Net(ABC, nn.Module):
    __call__: Callable[[Tensor, Tensor, Tensor], Tensor]

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError


class Guidance(ABC):
    pass


class Loss(ABC, Generic[D]):

    @abstractmethod
    def compute(self, batch: Batch[D]) -> Tensor:
        raise NotImplementedError

    def __call__(self, batch: Batch[D]) -> Tensor:
        return self.compute(batch)

    def __mul__(self, factor: float) -> "Mul[D]":
        return Mul(factor, self)

    def __rmul__(self, factor: float) -> "Mul[D]":
        return Mul(factor, self)

    def __truediv__(self, divisor: float) -> "Mul[D]":
        return Mul(1 / divisor, self)

    def __add__(self, other: "Loss[D]") -> "Add[D]":
        return Add(self, other)

    def __sub__(self, other: "Loss[D]") -> "Add[D]":
        return Add(self, Mul(-1, other))


@dataclass
class Mul(Loss[D]):
    factor: float
    loss: Loss[D]

    def compute(self, batch: Batch[D]) -> Tensor:
        return self.factor * self.loss.compute(batch)


class Add(Loss[D]):

    def __init__(self, *losses: Loss[D]) -> None:
        self.losses = losses

    def compute(self, batch: Batch[D]) -> Tensor:
        sum = self.losses[0].compute(batch)
        for loss in self.losses[1:]:
            sum += loss.compute(batch)
        return sum
