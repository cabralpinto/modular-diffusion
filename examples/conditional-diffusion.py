import sys
from pathlib import Path

import torch
from einops import rearrange
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

sys.path.append(".")

import diffusion
from diffusion.data import Identity
from diffusion.guidance import ClassifierFree
from diffusion.loss import Simple
from diffusion.net import UNet
from diffusion.noise import Gaussian
from diffusion.schedule import Cosine

file = Path(__file__)
input = file.parent / "data/in"
output = file.parent / "data/out" / file.stem
output.mkdir(parents=True, exist_ok=True)
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

x, y = zip(*MNIST(str(input), transform=ToTensor(), download=True))
x, y = torch.stack(x) * 2 - 1, torch.tensor(y) + 1

model = diffusion.Model(
    data=Identity(x, y, batch=128, shuffle=True),
    schedule=Cosine(steps=1000),
    noise=Gaussian(parameter="epsilon", variance="fixed"),
    net=UNet(channels=(1, 64, 128, 256), labels=10),
    guidance=ClassifierFree(dropout=0.1, strength=2),
    loss=Simple(parameter="epsilon"),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

if (output / "model.pt").exists():
    model.load(output / "model.pt")
epoch = sum(1 for _ in output.glob("[0-9]*"))

for epoch, loss in enumerate(model.train(epochs=100), 1):
    z = model.sample(torch.arange(1, 11))
    z = z[torch.linspace(0, z.shape[0] - 1, 10).int()]
    z = rearrange(z, "t b c h w -> c (b h) (t w)")
    z = (z + 1) / 2
    save_image(z, output / f"{epoch}-{loss:.2e}.png")
    model.save(output / "model.pt")