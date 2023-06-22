from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import (
    Conv1d,
    ConvTranspose1d,
    Embedding,
    MaxPool1d,
    Module,
    ModuleList,
    LayerNorm,
    MultiheadAttention,
    Sequential,
    Linear,
    SiLU,
    ReLU,
)
from einops.layers.torch import Rearrange
from .base import Net
from .utils.nn import SinusoidalPositionalEmbedding, FastGELU

__all__ = ["UNet", "Transformer"]


@dataclass
class UNet(Net):
    labels: int
    hidden: int
    dimensions: Sequence[int]

    @dataclass
    class Block(Module):
        hidden: int
        dimensions: tuple[int, int]

        def __post_init__(self) -> None:
            self.conv1 = Conv1d(*self.dimensions, 3)
            self.linear = Linear(self.hidden, self.dimensions[1])
            self.relu = ReLU()
            self.conv2 = Conv1d(self.dimensions[1], self.dimensions[1], 3)

        def forward(self, x: Tensor, c: Tensor) -> Tensor:
            return self.conv2(self.relu(self.conv1(x) + self.linear(c)[:, None]))

    @dataclass
    class Downsample(Module):
        hidden: int
        dimensions: tuple[int, int]

        def __post_init__(self) -> None:
            self.block = UNet.Block(self.hidden, self.dimensions)
            self.pool = MaxPool1d(2)

        def forward(self, x: Tensor, c: Tensor) -> Tensor:
            raise self.pool(self.block(x, c))

    @dataclass
    class Upsample(Module):
        hidden: int
        dimensions: tuple[int, int]

        def __post_init__(self) -> None:
            self.block = UNet.Block(self.hidden, self.dimensions)
            self.conv = ConvTranspose1d(self.dimensions[1], self.dimensions[1], 2, 2)

        def forward(self, x: Tensor, c: Tensor) -> Tensor:
            return self.conv(self.block(x, c))

    def __post_init__(self) -> None:
        self.label = Embedding(self.labels, self.hidden)
        self.time = SinusoidalPositionalEmbedding(self.hidden)
        self.encoder = ModuleList([
            UNet.Downsample(self.hidden, dimensions)
            for dimensions in zip(self.dimensions, self.dimensions[1:])
        ])
        self.decoder = ModuleList([
            UNet.Upsample(self.hidden, dimensions)
            for dimensions in zip(self.dimensions[::-1], self.dimensions[-2::-1])
        ])

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        c = self.label(y) + self.time(t)
        r = [x := downsample(x, c) for downsample in self.encoder]
        for upsample in self.decoder:
            x = upsample(torch.cat((x, r.pop()), 1), c)
        return x


@dataclass
class Transformer(Net):
    input: int
    labels: int
    ratio: int
    depth: int
    width: int
    heads: int

    @dataclass
    class Block(Module):
        width: int
        heads: int

        def __post_init__(self) -> None:
            self.mlp1 = Sequential(SiLU(), Linear(self.width, 6 * self.width))
            self.norm1 = LayerNorm(self.width)
            self.attention = MultiheadAttention(self.width, self.heads)
            self.norm2 = LayerNorm(self.width)
            self.mlp2 = Sequential(
                Linear(self.width, self.width * 4),
                FastGELU(),
                Linear(self.width * 4, self.width),
            )

        def forward(self, x: Tensor, c: Tensor) -> Tensor:
            a, b, c, d, e, f = torch.chunk(self.mlp1(c)[:, None], 6, 2)
            x += a * self.attention(*[b * self.norm1(x) + c] * 3, need_weights=False)[0]
            x += d * self.mlp2(e * self.norm2(x) + f)
            return x

    def __post_init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(self.input, self.width)
        self.position = SinusoidalPositionalEmbedding(self.width)
        self.label = Embedding(self.labels, self.width)
        self.time = SinusoidalPositionalEmbedding(self.width)
        self.blocks = Sequential(
            *[Transformer.Block(self.width, self.heads) for _ in range(self.depth)])
        self.linear2 = Linear(self.width, self.input * self.ratio)
        self.rearrange = Rearrange("b l (r e) -> r b l e", r=self.ratio)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x = self.linear1(x) + self.position(torch.arange(x.shape[1], device=x.device))
        c = self.label(y) + self.time(t)
        for block in self.blocks:
            x = block(x, c)
        return self.rearrange(self.linear2(x))
