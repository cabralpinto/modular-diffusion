from typing import Sequence, Optional

import torch
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import (
    Conv2d,
    Embedding,
    GroupNorm,
    Identity,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    MultiheadAttention,
    Sequential,
    SiLU,
    Upsample,
)
from torch.nn.functional import pad, silu
from torchvision.transforms.functional import crop

from .base import Net
from .utils.nn import FastGELU, SinusoidalPositionalEmbedding, WeightStdConv2d, Swish

__all__ = ["UNet", "Transformer"]


class UNet(Net):

    class Block(Module):

        def __init__(self, hidden: int, channels: tuple[int, int], groups: int) -> None:
            super().__init__()
            self.linear = Linear(hidden, 2 * channels[1])
            self.conv1 = WeightStdConv2d(*channels, 3, 1, 1)
            self.norm1 = GroupNorm(groups, channels[1])
            self.conv2 = WeightStdConv2d(channels[1], channels[1], 3, 1, 1)
            self.norm2 = GroupNorm(groups, channels[1])

        def forward(self, x: Tensor, c: Tensor) -> Tensor:
            a, b = torch.chunk(silu(self.linear(c))[..., None, None], 2, 1)
            x = silu(self.norm1(self.conv1(x)) * (a + 1) + b)
            x = silu(self.norm2(self.conv2(x)))
            return x

    class Attention(Module):

        def __init__(self, hidden: int, heads: int) -> None:
            super().__init__()
            self.attention = MultiheadAttention(hidden, heads)

        def forward(self, x: Tensor) -> Tensor:
            shape = x.shape
            x = x.flatten(2).permute(2, 0, 1)
            x, _ = self.attention(x, x, x, need_weights=False)
            x = x.permute(1, 2, 0).reshape(shape)
            return x

    def __init__(
        self,
        channels: Sequence[int],
        labels: int = 0,
        ratio: int = 1,
        hidden: int = 256,
        heads: int = 8,
        groups: int = 16,
    ) -> None:
        super().__init__()
        self.label = Embedding(labels + 1, hidden)
        self.time = SinusoidalPositionalEmbedding(hidden)
        self.input = Conv2d(channels[0], channels[1], 3, 1, 1)
        self.encoder = ModuleList([
            ModuleList([
                UNet.Block(hidden, 2 * (channels_[0],), groups),
                UNet.Block(hidden, 2 * (channels_[0],), groups),
                UNet.Attention(channels_[0], heads),
                GroupNorm(groups, channels_[0]),
                Sequential(
                    Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2) if
                    (last := channels_[1] < channels[-1]) else Identity(),
                    Conv2d(channels_[0] * (1 + 3 * last), channels_[1], 1),
                )
            ]) for channels_ in zip(channels[1:], channels[2:])
        ])
        self.bottleneck = ModuleList([
            UNet.Block(hidden, (channels[-1], channels[-1]), groups),
            UNet.Attention(channels[-1], heads),
            UNet.Block(hidden, (channels[-1], channels[-1]), groups),
        ])
        self.decoder = ModuleList([
            ModuleList([
                UNet.Block(hidden, (sum(channels_), channels_[0]), groups),
                UNet.Block(hidden, (sum(channels_), channels_[0]), groups),
                UNet.Attention(channels_[0], heads),
                GroupNorm(groups, channels_[0]),
                Sequential(
                    Upsample(None, 2, "nearest")
                    if channels_[1] > channels[1] else Identity(),
                    Conv2d(channels_[0], channels_[1], 3, 1, 1),
                )
            ]) for channels_ in zip(channels[:1:-1], channels[-2::-1])
        ])
        self.output = Sequential(
            Conv2d(channels[1], ratio * channels[0], 3, 1, 1),
            Rearrange("b (r c) h w -> r b c h w", r=ratio),
        )

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x = self.input(x)
        c = self.label(y) + self.time(t)
        h = list[Tensor]()
        for block1, block2, attention, norm, transform in self.encoder:  # type: ignore[assignment]
            x = block1(x, c)
            h.append(x)
            x = block2(x, c)
            x = attention(norm(x)) + x
            h.append(x)
            x = pad(x, (0, x.shape[2] % 2, 0, x.shape[3] % 2))
            x = transform(x)
        x = self.bottleneck[0](x, c)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, c)
        for block1, block2, attention, norm, transform in self.decoder:  # type: ignore[assignment]
            x = crop(x, 0, 0, *h[-1].shape[2:])
            x = block1(torch.cat((x, h.pop()), 1), c)
            x = block2(torch.cat((x, h.pop()), 1), c)
            x = attention(norm(x)) + x
            x = transform(x)
        x = self.output(x)
        return x


class EfficientUNet(Net):
    """Implementation of Efficient U-Net from https://arxiv.org/abs/2205.11487v1"""

    class Block(Module):

        def __init__(self, channels: int, groups: int) -> None:
            super().__init__()
            self.conv = Conv2d(channels, channels, 1)
            self.blocks = Sequential(
                GroupNorm(groups, channels),
                Swish(),
                Conv2d(channels, channels, 3, 1, 1),
                GroupNorm(groups, channels),
                Swish(),
                Conv2d(channels, channels, 3, 1, 1),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x) + self.blocks(x)

    class Condition(Module):

        def __init__(self, labels: int, channels: int, hidden: int) -> None:
            super().__init__()
            self.label = Embedding(labels + 1, channels)
            self.time = SinusoidalPositionalEmbedding(channels)
            self.linear = Linear(hidden, 2 * channels)

    class DBlock(Module):

        def __init__(
            self,
            channels: tuple[int, int],
            groups: int,
            blocks: int,
            heads: int,
            stride: int = 1,
        ) -> None:
            super().__init__()
            self.conv = Conv2d(channels[0], channels[1], 3, stride, 1)
            self.condition = EfficientUNet.Condition(0, channels[1], 256)
            self.blocks = ModuleList(
                [EfficientUNet.Block(channels[1], groups) for _ in range(blocks)])
            self.attention = (MultiheadAttention(2 * channels, heads, batch_first=True)
                              if heads > 0 else None)

        def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
            x = self.conv(x)
            x = self.condition(x, y, t)
            for block in self.blocks:
                x = block(x)
            if self.attention is not None:
                x, _ = self.attention(x, x, x, need_weights=False)
            return x

    class UBlock(Module):

        def __init__(
            self,
            channels: tuple[int, int],
            groups: int,
            blocks: int,
            heads: int,
            stride: int = 1,
        ) -> None:
            super().__init__()
            self.condition = EfficientUNet.Condition(0, channels[1], 256)
            self.blocks = ModuleList(
                [EfficientUNet.Block(channels[1], groups) for _ in range(blocks)])
            self.attention = MultiheadAttention(2 * channels, heads, batch_first=True)
            self.conv = Conv2d(channels[0], channels[1], 3, stride, 1)

        def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
            x = self.condition(x, y, t)
            for block in self.blocks:
                x = block(x)
            x, _ = self.attention(x, x, x, need_weights=False)
            x = self.conv(x)
            return x

    def __init__(
        self,
        channels: Sequence[int],
        strides: int | Sequence[int] = 2,
        blocks: int | Sequence[int] = 2,
        groups: int | Sequence[int] = 16,
        heads: int | Sequence[int] = 8,
        labels: int = 0,
        ratio: int = 1,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        if isinstance(strides, int):
            strides = [strides] * (len(channels) - 1)
        if isinstance(blocks, int):
            blocks = [blocks] * (len(channels) - 1)
        if isinstance(groups, int):
            groups = [groups] * (len(channels) - 1)
        if isinstance(heads, int):
            heads = [heads] * (len(channels) - 1)

        self.conv = Conv2d(channels[0], channels[1], 3, 1, 1)
        self.encoder = ModuleList([
            EfficientUNet.DBlock(channels, groups, blocks, heads, stride)
            for channels, stride, blocks, groups, heads
            in zip(zip(channels[:-1], channels[1:]), strides, blocks, groups, heads)
        ]) # yapf: disable
        self.bottleneck = ModuleList([
            EfficientUNet.DBlock(channels[0], groups, 1, heads, (2, 2)),
            EfficientUNet.UBlock(channels[0], groups, 1, heads, (2, 2))
        ])
        self.encoder = ModuleList([
            EfficientUNet.DBlock(channels, groups, blocks, heads, stride)
            for channels, stride, blocks, groups, heads
            in zip(zip(channels[::-1], channels[-2::-1]), strides, blocks, groups, heads)
        ]) # yapf: disable
        self.linear = Linear(hidden, 2 * channels[1])
        self.rearrange = Rearrange("b (r c) h w -> r b c h w", r=ratio)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x = self.conv(x)
        h = [x := block(x, y, t) for block in self.encoder]
        for block in self.bottleneck:
            x = self.bottleneck(x, y, t)
        for block in self.decoder:
            x = x + h.pop()
            x = block(x, y, t)
        return x


class Transformer(Net):

    class Block(Module):

        def __init__(self, width: int, heads: int) -> None:
            super().__init__()
            self.mlp1 = Sequential(SiLU(), Linear(width, 6 * width))
            self.norm1 = LayerNorm(width)
            self.attn = MultiheadAttention(width, heads, batch_first=True)
            self.norm2 = LayerNorm(width)
            self.mlp2 = Sequential(
                Linear(width, width * 4),
                FastGELU(),
                Linear(width * 4, width),
            )

        def forward(self, x: Tensor, c: Tensor) -> Tensor:
            a, b, c, d, e, f = torch.chunk(self.mlp1(c)[:, None], 6, 2)
            x = x + a * self.attn(*[b * self.norm1(x) + c] * 3, need_weights=False)[0]
            x = x + d * self.mlp2(e * self.norm2(x) + f)
            return x

    def __init__(
        self,
        input: int,
        labels: int = 0,
        ratio: int = 1,
        depth: int = 6,
        width: int = 256,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.linear1 = Linear(input, width)
        self.position = SinusoidalPositionalEmbedding(width)
        self.label = Embedding(labels + 1, width)
        self.time = SinusoidalPositionalEmbedding(width)
        self.blocks = Sequential(
            *[Transformer.Block(width, heads) for _ in range(depth)])
        self.linear2 = Linear(width, input * ratio)
        self.rearrange = Rearrange("b l (r e) -> r b l e", r=ratio)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x = self.linear1(x) + self.position(torch.arange(x.shape[1], device=x.device))
        c = self.label(y) + self.time(t)
        for block in self.blocks:
            x = block(x, c)
        return self.rearrange(self.linear2(x))
