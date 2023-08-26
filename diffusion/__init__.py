import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, Generic, Iterator, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from . import data, guidance, loss, net, noise, schedule, time
from .base import Batch, Data, Distribution, Guidance, Loss, Net, Noise, Schedule, Time
from .time import Discrete

__all__ = ["data", "loss", "net", "noise", "schedule", "time", "Model"]

D = TypeVar("D", bound=Distribution, covariant=True)


@dataclass
class Model(Generic[D]):
    data: Data
    schedule: Schedule
    noise: Noise[D]
    loss: Loss[D]
    net: Net
    time: Time = field(default_factory=Discrete)
    guidance: Optional[Guidance] = None  # TODO remove hardcoding
    optimizer: Optional[Optimizer | Callable[..., Optimizer]] = None
    device: torch.device = torch.device("cpu")
    compile: bool = True

    @torch.no_grad()
    def __post_init__(self):
        self.noise.schedule(self.schedule.compute().to(self.device))
        parameters = chain(self.data.parameters(), self.net.parameters())
        if self.optimizer is None:
            self.optimizer = Adam(parameters, lr=1e-4)
        elif callable(self.optimizer):
            self.optimizer = self.optimizer(parameters)
        self.net = self.net.to(self.device)
        for name, value in vars(self.data).items():
            if isinstance(value, nn.Module):
                setattr(self.data, name, value.to(self.device)) 
        if self.compile and sys.version_info < (3, 11):
            self.net = torch.compile(self.net)  # type: ignore[union-attr]

    @torch.no_grad()
    def load(self, path: Path | str):
        state = torch.load(path)
        self.net.load_state_dict(state["net"])
        for name, dict in state["data"].items():
            getattr(self.data, name).load_state_dict(dict)

    @torch.no_grad()
    def save(self, path: Path | str):
        state = {
            "net": self.net.state_dict(),
            "data": {
                name: value.state_dict()
                for name, value in vars(self.data).items()
                if isinstance(value, nn.Module)
            }
        }
        torch.save(state, path)

    @torch.enable_grad()
    def train(self, epochs: int = 1, progress: bool = True) -> Iterator[float]:
        self.net.train()
        batch = Batch[D](self.device)
        for _ in range(epochs):
            bar = tqdm(self.data, disable=not progress)
            for batch.w, batch.y in self.data:
                if isinstance(self.guidance, guidance.ClassifierFree):
                    i = torch.randperm(batch.y.shape[0])
                    batch.y[i[:int(batch.y.shape[0] * self.guidance.dropout)]] = 0
                batch.x = self.data.encode(batch.w)
                batch.t = self.time.sample(self.schedule.steps, batch.x.shape[0])
                batch.z, batch.epsilon = self.noise.prior(batch.x, batch.t).sample()
                batch.hat = self.net(batch.z, batch.y, batch.t)
                batch.q = self.noise.posterior(batch.x, batch.z, batch.t)
                batch.p = self.noise.approximate(batch.z, batch.t, batch.hat)
                batch.l = self.loss.compute(batch)
                self.optimizer.zero_grad()  # type: ignore[union-attr]
                batch.l.backward()
                self.optimizer.step()  # type: ignore[union-attr]
                bar.set_postfix(loss=f"{batch.l.item():.2e}")
                bar.update()
            bar.close()
            yield batch.l.item()

    @torch.no_grad()
    def sample(
        self,
        y: Optional[Tensor] = None,
        batch: int = 1,
        progress: bool = True,
    ) -> Tensor:
        self.net.eval()
        if y is None:
            shape = 1, *(() if self.data.y is None else self.data.y.shape[1:])
            y = torch.zeros(shape, dtype=torch.int, device=self.device)
        y = y.repeat_interleave(batch, 0).to(self.device)
        pi = self.noise.isotropic((y.shape[0], *self.data.shape))
        z = pi.sample()[0].to(self.device)
        l = self.data.decode(z)[None]
        bar = tqdm(total=self.schedule.steps, disable=not progress)
        for t in range(self.schedule.steps, 0, -1):
            t = torch.full((batch,), t, device=self.device)
            hat = self.net(z, y, t)
            if isinstance(self.guidance, guidance.ClassifierFree):
                s = self.guidance.strength
                hat = (1 + s) * hat - s * self.net(z, torch.zeros_like(y), t)
            z, _ = self.noise.approximate(z, t, hat).sample()
            w = self.data.decode(z)
            l = torch.cat((l, w[None]), 0)
            bar.update()
        bar.close()
        return l
