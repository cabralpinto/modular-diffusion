import warnings
from pathlib import Path

import torch
from einops import rearrange
from torchvision.utils import save_image

import diffusion
from diffusion.data import Identity
from diffusion.loss import Simple
from diffusion.net import Transformer
from diffusion.noise import Normal
from diffusion.schedule import Cosine

dataset = "mnist"
input, output = Path("data/in") / dataset, Path("data/out") / dataset
output.mkdir(exist_ok=True, parents=True)
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)

c, h, w, p, q = 1, 28, 28, 2, 2
x = (input / "images").read_bytes()[16:]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x = torch.frombuffer(x, dtype=torch.uint8).view(-1, c, h, w)
x = rearrange(x / 127.5 - 1, "b c (h p) (w q) -> b (h w) (c p q)", p=p, q=q)
y = torch.frombuffer((input / "labels").read_bytes()[8:], dtype=torch.uint8)

model = diffusion.Model(
    data=Identity(x, y, batch=128, shuffle=True),
    schedule=Cosine(1000),
    loss=Simple(),
    noise=Normal(parameter="x", variance="fixed"),
    net=Transformer(
        input=x.shape[2],
        labels=10,
        hidden=768,
        heads=12,
        depth=12,
    ),
    device=torch.device("cuda"),
)

for epoch, loss in enumerate(model.train(epochs=100)):
    l = model.sample(torch.arange(10))
    l = (l[torch.linspace(0, l.shape[0] - 1, 10).long()] + 1) / 2
    l = rearrange(l, "t b (h w) (c p q) -> c (b h p) (t w q)", h=h // p, p=p, q=q)
    save_image(l, output / f"{epoch} ({loss:.2e}).png")