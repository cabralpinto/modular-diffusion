import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Generic, Iterator, Optional, TypeVar

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import AdamW, Optimizer
from tqdm import tqdm

from . import data, guidance, loss, net, noise, schedule, time
from .base import Batch, Data, Distribution, Guidance, Loss, Net, Noise, Schedule, Time
from .time import Uniform

__all__ = ["data", "loss", "net", "noise", "schedule", "time", "Model"]

D = TypeVar("D", bound=Distribution, covariant=True)


@dataclass
class Model(Generic[D]):
    data: Data
    schedule: Schedule
    noise: Noise[D]
    loss: Loss[D]
    net: Net
    time: Time = field(default_factory=Uniform)
    guidance: Optional[Guidance] = None  # TODO remove hardcoding
    optimizer: Optional[Optimizer | Callable[..., Optimizer]] = None
    device: torch.device = torch.device("cuda")

    @property
    def parameters(self) -> Iterator[Parameter]:
        return self.net.parameters()  # TODO add all parameters

    @torch.no_grad()
    def __post_init__(self):
        self.noise.schedule(self.schedule.compute().to(self.device))
        if self.optimizer is None:
            self.optimizer = partial(AdamW, lr=1e-4)
        if callable(self.optimizer):
            self.optimizer = self.optimizer(self.parameters)
        self.net = self.net.to(self.device)  # TODO: add all tensors to device
        if sys.version_info <= (3, 10):
            self.net = torch.compile(self.net)  # type: ignore[union-attr]

    @torch.enable_grad()
    def train(self, epochs: int = 1) -> Iterator[float]:
        self.net.train()
        batch = Batch[D](self.device)
        for _ in range(epochs):
            for batch.w, batch.y in (bar := tqdm(self.data)):
                if isinstance(self.guidance, guidance.ClassifierFree):
                    i = torch.randperm(batch.y.shape[0])
                    batch.y[i[:int(batch.y.shape[0] * self.guidance.dropout)]] = 0
                batch.x = self.data.encode(batch.w)
                batch.t = self.time(self.schedule.steps, batch.x.shape[0])
                batch.z, batch.epsilon = self.noise.prior(batch.x, batch.t).sample()
                batch.hat = self.net(batch.z, batch.y, batch.t)
                batch.q = self.noise.posterior(batch.x, batch.z, batch.t)
                batch.p = self.noise.approximate(batch.z, batch.t, batch.hat)
                batch.l = self.loss(batch)
                bar.set_postfix(loss=batch.l.item())
                self.optimizer.zero_grad()  # type: ignore[union-attr]
                batch.l.backward()
                self.optimizer.step()  # type: ignore[union-attr]
            yield batch.l.item()

    @torch.no_grad()
    def sample(self, y: Optional[Tensor] = None, batch: int = 1) -> Tensor:
        if y is None:
            y = torch.zeros(batch, dtype=torch.int, device=self.device)
        self.net.eval()
        y = y.to(self.device)
        pi = self.noise.isotropic(y.shape[0], *self.data.shape)
        z = pi.sample()[0].to(self.device)
        l = torch.zeros(0, y.shape[0], *self.data.shape, device=self.device)
        for t in tqdm(range(self.schedule.steps, 0, -1)):
            t = torch.full_like(y, t)
            hat = self.net(z, y, t)
            if isinstance(self.guidance, guidance.ClassifierFree):
                s = self.guidance.weight
                hat = s * hat + (1 - s) * self.net(z, torch.zeros_like(y), t)
            z, _ = self.noise.approximate(z, t, hat).sample()
            w = self.data.decode(z)
            l = torch.cat((l, w[None]), 0)
        return l
