import pytest
import torch
from diffusion.noise import Gaussian, Uniform, Absorbing

def test_gaussian_schedule():
    gaussian = Gaussian()
    alpha = torch.tensor([0.9, 0.8, 0.7])
    gaussian.schedule(alpha)
    assert hasattr(gaussian, 'q1')
    assert hasattr(gaussian, 'q2')
    assert hasattr(gaussian, 'q3')
    assert hasattr(gaussian, 'q4')
    assert hasattr(gaussian, 'q5')

def test_gaussian_stationary():
    gaussian = Gaussian()
    shape = (2, 2)
    dist = gaussian.stationary(shape)
    assert dist.mu.shape == shape
    assert dist.sigma.shape == shape

def test_uniform_schedule():
    uniform = Uniform(k=3)
    alpha = torch.tensor([0.9, 0.8, 0.7])
    uniform.schedule(alpha)
    assert hasattr(uniform, 'alpha')
    assert hasattr(uniform, 'delta')
    assert hasattr(uniform, 'i')

def test_uniform_stationary():
    uniform = Uniform(k=3)
    shape = (2, 3)
    dist = uniform.stationary(shape)
    assert hasattr(dist, 'p')
    assert dist.p.shape == shape

def test_absorbing_schedule():
    absorbing = Absorbing(k=3, m=1)
    alpha = torch.tensor([0.9, 0.8, 0.7])
    absorbing.schedule(alpha)
    assert hasattr(absorbing, 'alpha')
    assert hasattr(absorbing, 'delta')
    assert hasattr(absorbing, 'i')

def test_absorbing_stationary():
    absorbing = Absorbing(k=3, m=1)
    shape = (2, 3)
    dist = absorbing.stationary(shape)
    assert hasattr(dist, 'p')
    assert dist.p.shape == shape

if __name__ == "__main__":
    pytest.main()
