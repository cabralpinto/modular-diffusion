import pytest
import torch
from diffusion.time import Discrete

def test_discrete_sample():
    discrete = Discrete()
    steps = 10
    size = 5
    samples = discrete.sample(steps, size)
    assert samples.shape == (size,)
    assert torch.all(samples >= 1) and torch.all(samples <= steps)

if __name__ == "__main__":
    pytest.main()
