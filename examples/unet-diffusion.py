import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from torch.optim import Adam
from functools import partial

sys.path.append(".")
sys.path.append("examples")

from utils import download

import diffusion
from diffusion.base import Noise, Loss, Batch
from diffusion.data import Identity
from diffusion.distribution import Normal as N
from diffusion.loss import Simple
from diffusion.net import UNet
from diffusion.schedule import Cosine, Linear

from diffusion.noise import Gaussian

@dataclass
class Gaussian(Noise[N]):

    def schedule(self, alpha: Tensor) -> None:
        self.alpha = alpha
        self.delta = alpha.cumprod(0)

    def isotropic(self, shape: tuple[int, ...]) -> N:
        return N(torch.zeros(shape), torch.ones(shape))

    def prior(self, x: Tensor, t: Tensor) -> N:
        t = t.view(-1, *(1,) * (x.dim() - 1))
        return N(self.alpha[t].sqrt() * x, 1 - self.delta[t])

    def posterior(self, x: Tensor, z: Tensor, t: Tensor) -> N:
        t = t.view(-1, *(1,) * (x.dim() - 1))
        mu = self.alpha[t].sqrt() * (1 - self.delta[t - 1]) * z
        mu += self.delta[t - 1].sqrt() * (1 - self.alpha[t]) * x
        mu /= (1 - self.delta[t])
        sigma = (1 - self.alpha[t]) * (1 - self.delta[t - 1]) / (1 - self.delta[t])
        return N(mu, sigma)

    def approximate(self, z: Tensor, t: Tensor, hat: Tensor) -> N:
        # print(z.min().item(), z.max().item(), flush=True)
        t = t.view(-1, *(1,) * (z.dim() - 1))
        mu = (z - (1 - self.alpha[t]) / (1 - self.delta[t]).sqrt() * hat[0])
        mu /= self.alpha[t].sqrt()
        sigma = (1 - self.alpha[t]) * (1 - self.delta[t - 1]) / (1 - self.delta[t])
        return N(mu, sigma)


class CustomLoss(Loss[N]):

    def compute(self, batch: Batch[N]) -> Tensor:
        # t = batch.t.view(-1, *(1,) * (batch.x.ndim - 1))
        # x = (batch.z - (1 - model.noise.delta[t]).sqrt() * batch.epsilon) / model.noise.delta[t].sqrt()
        # xh = (batch.z - (1 - model.noise.delta[t]).sqrt() * batch.hat[0]) / model.noise.delta[t].sqrt()
        # print()
        # print(batch.x.min().item(), batch.x.max().item(), flush=True)
        # print(x.min().item(), x.max().item(), flush=True)
        # print(xh.min().item(), xh.max().item(), flush=True)
        # print(batch.epsilon.min().item(), batch.epsilon.max().item(), flush=True)
        # print(batch.hat[0].min().item(), batch.hat[0].max().item(), flush=True)
        # print(model.noise.delta[t].min().item(), model.noise.delta[t].max().item(), flush=True)
        # print(batch.z.min().item(), batch.z.max().item(), flush=True)
        # print(((batch.epsilon - batch.hat[0])**2).mean().item(), flush=True)

        # print(x, flush=True)
        # print(xh, flush=True)
        return (torch.norm(batch.epsilon - batch.hat[0])**2).mean()


# TODO simple loss should use norm

file = Path(__file__)
input = file.parent / "data/in/afhq"
input.parent.mkdir(parents=True, exist_ok=True)
output = file.parent / "data/out" / file.stem
output.mkdir(parents=True, exist_ok=True)
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

if not input.exists():
    download("https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1", "afhq.zip")
    shutil.unpack_archive("afhq.zip", input.parent)
    Path("afhq.zip").unlink()
    (input / "dog").mkdir()
    for path in input.glob("*/dog/*"):
        path.rename(input / "dog" / path.name)
    for path in input / "train", input / "val":
        shutil.rmtree(path)

x, _ = zip(*ImageFolder(str(input), ToTensor()))
x = torch.stack(x) * 2 - 1
x = resize(x, [32, 32], antialias=True)
print(x.shape)

model = diffusion.Model(
    data=Identity(x, batch=128, shuffle=True),
    schedule=Cosine(1000),
    noise=Gaussian(parameter="epsilon"),
    net=UNet(channels=(3, 64, 128, 256)),
    loss=CustomLoss(),  # Simple(parameter="epsilon"), 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    optimizer=partial(Adam, lr=3e-4),
)

if (output / "model.pt").exists():
    model.load(output / "model.pt")
epoch = sum(1 for _ in output.glob("[0-9]*"))

# model.sample(batch=10)

for epoch, loss in enumerate(model.train(epochs=1000), epoch + 1):
    z = model.sample(batch=10)
    print(z[-1].min(), z[-1].max(), flush=True)
    z = z[torch.linspace(0, z.shape[0] - 1, 10).int()]
    z = (z + 1) / 2
    # print(z[-1].min(), z[-1].max(), flush=True)
    z = rearrange(z, "t b c h w -> c (b h) (t w)")
    save_image(z, output / f"{epoch}-{loss:.2e}.png")
    model.save(output / "model.pt")