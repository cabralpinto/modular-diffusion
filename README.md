# Modular Diffusion

[![PyPI version](https://badge.fury.io/py/modular-diffusion.svg)](https://badge.fury.io/py/modular-diffusion)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://cabralpinto.github.io/modular-diffusion/)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](https://lbesson.mit-license.org/)

Modular Diffusion provides an easy-to-use modular API to design and train custom Diffusion Models with PyTorch. Whether you're an enthusiast exploring Diffusion Models or a hardcore ML researcher, **this framework is for you**.

## Features

- ⚙️ **Highly Modular Design**: Effortlessly swap different components of the diffusion process, including noise type, schedule type, denoising network, and loss function.
- 📚 **Growing Library of Pre-built Modules**: Get started right away with our comprehensive selection of pre-built modules.
- 🔨 **Custom Module Creation Made Easy**: Craft your own original modules by inheriting from a base class and implementing the required methods.
- 🤝 **Integration with PyTorch**: Built on top of PyTorch, Modular Diffusion enables you to develop custom modules using a familiar syntax.
- 🌈 **Broad Range of Applications**: From generating high-quality images to implementing non-autoregressive text synthesis pipelines, the possiblities are endless.

## Installation

Modular Diffusion officially supports Python 3.10+ and is available on PyPI:

```bash
pip install modular-diffusion
```

> **Note**: Although Modular Diffusion works with later Python versions, we currently recommend using Python 3.10. This is because `torch.compile`, which significantly improves the speed of the models, is not currently available for versions above Python 3.10.

## Usage

With Modular Diffusion, you can build and train a custom Diffusion Model in just a few lines. First, load and normalize your dataset. We are using the dog pictures from [AFHQ](https://paperswithcode.com/dataset/afhq).

```python
x, _ = zip(*ImageFolder("afhq", ToTensor()))
x = resize(x, [h, w], antialias=False)
x = torch.stack(x) * 2 - 1
```

Next, build your custom model using either Modular Diffusion's prebuilt modules or [your custom modules](https://cabralpinto.github.io/modular-diffusion/guides/custom-modules/).

```python
model = diffusion.Model(
   data=Identity(x, batch=128, shuffle=True),
   schedule=Cosine(steps=1000),
   noise=Gaussian(parameter="epsilon", variance="fixed"),
   net=UNet(channels=(1, 64, 128, 256)),
   loss=Simple(parameter="epsilon"),
)
```

Now, train and sample from the model.

```python
losses = [*model.train(epochs=400)]
z = model.sample(batch=10)
z = z[torch.linspace(0, z.shape[0] - 1, 10).long()]
z = rearrange(z, "t b c h w -> c (b h) (t w)")
save_image((z + 1) / 2, "output.png")
```

Finally, marvel at the results.

<img width="360" alt="Modular Diffusion teaser" src="https://github.com/cabralpinto/modular-diffusion/assets/47889626/2756f798-8037-460e-b827-255812f203b6">

## Contributing

We appreciate your support and welcome your contributions! Please fell free to submit pull requests if you found a bug or typo you want to fix. If you want to contribute a new prebuilt module or feature, please start by opening an issue and discussing it with us. If you don't know where to begin, take a look at the [open issues](https://github.com/cabralpinto/modular-diffusion/issues).

## License

This project is licensed under the [MIT License](LICENSE).
