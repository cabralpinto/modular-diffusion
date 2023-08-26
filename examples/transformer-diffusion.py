import shutil
import sys
from pathlib import Path

import torch
from einops import rearrange
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

sys.path.append(".")
sys.path.append("examples")

from utils import download

import diffusion
from diffusion.data import Identity
from diffusion.loss import Simple
from diffusion.net import Transformer
from diffusion.noise import Gaussian
from diffusion.schedule import Cosine

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

c, h, w, p, q = 3, 64, 64, 2, 2
x, _ = zip(*ImageFolder(str(input), ToTensor()))
x = torch.stack(x) * 2 - 1
x = resize(x, [h, w], antialias=False)
x = rearrange(x, "b c (h p) (w q) -> b (h w) (c p q)", p=p, q=q)

model = diffusion.Model(
    data=Identity(x, batch=16, shuffle=True),
    schedule=Cosine(1000),
    noise=Gaussian(parameter="epsilon", variance="fixed"),
    net=Transformer(input=x.shape[2], width=768, depth=12, heads=12),
    loss=Simple(parameter="epsilon"),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

if (output / "model.pt").exists():
    model.load(output / "model.pt")
epoch = sum(1 for _ in output.glob("[0-9]*"))

for epoch, loss in enumerate(model.train(epochs=10000), epoch + 1):
    z = model.sample(batch=10)
    print(z[-1].min().item(), z[-1].max().item(), flush=True)
    z = z[torch.linspace(0, z.shape[0] - 1, 10).int()]
    z = rearrange(z, "t b (h w) (c p q) -> c (b h p) (t w q)", h=h // p, p=p, q=q)
    z = (z + 1) / 2
    save_image(z, output / f"{epoch}-{loss:.2e}.png")
    model.save(output / "model.pt")
