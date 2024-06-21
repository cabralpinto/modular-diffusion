import pytest
import torch
from diffusion.schedule import Constant, Linear, Cosine, Sqrt

def test_constant_schedule():
    constant = Constant(steps=5, value=0.5)
    alpha = constant.compute()
    assert alpha.shape == (6,)
    assert torch.all(alpha == 0.5)

def test_linear_schedule():
    linear = Linear(steps=5, start=0.1, end=0.9)
    alpha = linear.compute()
    assert alpha.shape == (6,)
    assert torch.allclose(alpha, torch.tensor([0.1, 0.26, 0.42, 0.58, 0.74, 0.9]), atol=0.01)

def test_cosine_schedule():
    cosine = Cosine(steps=5)
    alpha = cosine.compute()
    assert alpha.shape == (6,)
    assert torch.all(alpha >= 0) and torch.all(alpha <= 1)

def test_sqrt_schedule():
    sqrt = Sqrt(steps=5)
    alpha = sqrt.compute()
    assert alpha.shape == (6,)
    assert torch.all(alpha >= 0) and torch.all(alpha <= 1)

if __name__ == "__main__":
    pytest.main()
