# Modular Diffusion

Welcome to Modular Diffusion! This package provides an easy-to-use and extendable API to train generative diffusion models from scratch. It is designed with flexibility and modularity in mind, allowing you to quickly build your own diffusion models by plugging in different components.

> **Warning**: The code for Modular Diffusion is not yet available on PyPI, as it is still undergoing significant development. A stable release will be published there once the code base, unit tests, and documentation reach a satisfactory state of completion. Stay tuned for more updates, and thank you for your patience!

## Features
- ⚙️ **Highly Modular Design**: Tailor and build your own diffusion models by leveraging our flexible `Model` class. This class allows you to choose and implement various components of the diffusion process such as noise type, schedule type, denoising network, and loss function.
- 📚 **Growing Library of Pre-built Modules**: Get started right away with our comprehensive selection of pre-built modules.
- 🔨 **Custom Module Creation Made Easy**: Craft your own unique modules with ease. Just inherit from the base class of your chosen module type and get creative!
- 🤝 **Seamless Integration with PyTorch**: This project harmoniously integrates with PyTorch, allowing users to leverage the plethora of features and community support the framework has to offer.
- 🌈 **Broad Range of Applications**: From generating high-quality images to implementing non-autoregressive audio and text synthesis pipelines, the potential uses of Modular Diffusion are vast.

## Installation

Modular Diffusion officially supports Python 3.10+ and is available on PyPI:

```bash
pip install modular-diffusion
```

> **Note**: Although Modular Diffusion works with later Python versions, we highly recommend using Python 3.10. This is because `torch.compile`, which significantly improves the speed of the models, is not currently available for versions above Python 3.10.

## Usage

Using Modular Diffusion is simple and straightforward. Here is an example to get you started:

```python
import diffusion
from diffusion.data import Identity
from diffusion.loss import Simple
from diffusion.net import UNet
from diffusion.noise import Normal
from diffusion.schedule import Linear

model = diffusion.Model(
    data=Identity(x, y, batch=128, shuffle=True),
    schedule=Linear(1000, 0.9999, 0.98),
    loss=Simple(),
    noise=Normal(parameter="epsilon", variance="fixed"),
    net=UNet(x.shape[2], labels=10),
)
losses = [*model.train(epochs=1000)]
z = model.sample(torch.arange(10))
```

In this example, we train and sample from a generative diffusion model with the parameters used in the seminal work of [Ho et al. (2020)](https://arxiv.org/abs/2006.11239). Remember that the model can be customized by swapping modules or creating your own to fit your specific needs. The above example is only one of the numerous possibilities with Modular Diffusion. For more detailed examples, you can check out the [examples](examples) folder in our repository, and to explore all possibilities offered by this package, please refer to our [documentation](https://modular-diffusion.readthedocs.io/).

## Contributing

Contributions to this project are very much welcome! Feel free to raise issues or submit pull requests here on GitHub.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions, suggestions, or just want to say hello, feel free to email me at [jmcabralpinto@gmail.com](mailto:jmcabralpinto@gmail.com).
