import torch
from torch import nn
from diffusion import Model, Batch, Data, Net, Discrete
from diffusion.loss import Simple
from diffusion.noise import Gaussian
from diffusion.schedule import Linear
from typing import Optional

# Mock classes to implement abstract base classes
class MockDistribution(Gaussian):
    def sample(self):
        return torch.zeros(10, 2), torch.ones(10, 2)

    def nll(self, x: torch.Tensor):
        return torch.sum(x ** 2)

    def dkl(self, other: 'MockDistribution'):
        return torch.sum(self.sample()[0] - other.sample()[0])

class MockTime(Discrete):
    def sample(self, steps: int, size: int):
        return torch.randint(0, steps, (size,))

class MockSchedule(Linear):
    def __init__(self, steps):
        super().__init__(steps=steps, start=0.1, end=1.0)
        self.steps = steps

    def compute(self):
        return torch.linspace(0.1, 1.0, self.steps)

class MockNoise(Gaussian):
    def schedule(self, alpha: torch.Tensor):
        pass

    def stationary(self, shape: tuple[int, ...]):
        return MockDistribution()

    def prior(self, x: torch.Tensor, t: torch.Tensor):
        return MockDistribution()

    def posterior(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        return MockDistribution()

    def approximate(self, z: torch.Tensor, t: torch.Tensor, hat: torch.Tensor):
        return MockDistribution()

class MockNet(Net):
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        # Ensure batch sizes match
        y_expanded = y.unsqueeze(1).expand(x.size(0), x.size(1))
        t_expanded = t.unsqueeze(1).expand(x.size(0), x.size(1))
        return x + y_expanded + t_expanded

class MockLoss(Simple):
    def compute(self, batch: Batch[Gaussian]):
        return torch.tensor(0.0, requires_grad=True)

class MockData(Data):
    def __init__(self, w: torch.Tensor, y: Optional[torch.Tensor] = None, batch: int = 1, shuffle: bool = False):
        super().__init__(w, y, batch, shuffle)
        self.model = nn.Linear(w.shape[1], w.shape[1])
        self.batch = batch

    def encode(self, w: torch.Tensor):
        return self.model(w)

    def decode(self, x: torch.Tensor):
        return x

    def parameters(self):
        return self.model.parameters()

    def __iter__(self):
        if self.y is None:
            self.y = torch.zeros(self.w.shape[0], dtype=torch.int)
        self.iter_data = iter([(self.w[i:i+self.batch], self.y[i:i+self.batch]) for i in range(0, self.w.shape[0], self.batch)])
        return self

    def __next__(self):
        return next(self.iter_data)

# Tests
def test_model_initialization():
    data = MockData(torch.randn(100, 3))
    schedule = MockSchedule(steps=10)
    noise = MockNoise()
    net = MockNet()
    loss = MockLoss()
    model = Model(data=data, schedule=schedule, noise=noise, net=net, loss=loss)
    assert model.data == data
    assert model.schedule == schedule
    assert model.noise == noise
    assert model.net == net
    assert model.loss == loss

def test_model_train():
    data = MockData(torch.randn(100, 3), batch=10)
    schedule = MockSchedule(steps=10)
    noise = MockNoise()
    net = MockNet()
    loss = MockLoss()
    model = Model(data=data, schedule=schedule, noise=noise, net=net, loss=loss, device='cpu')
    losses = list(model.train(epochs=1, progress=False))
    assert len(losses) > 0

def test_model_save_load(tmp_path):
    data = MockData(torch.randn(100, 3), batch=10)
    schedule = MockSchedule(steps=10)
    noise = MockNoise()
    net = MockNet()
    loss = MockLoss()
    model = Model(data=data, schedule=schedule, noise=noise, net=net, loss=loss, device='cpu')
    path = tmp_path / "model.pth"
    model.save(path)
    model.load(path)
    assert model.net is not None
