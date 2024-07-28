import pytest
import torch
from diffusion.distribution import Normal

class TestNormal:

    @pytest.fixture(params=[(2, 2), (3, 3, 3), (4, 4, 4, 4)])
    def distribution(self, request) -> Normal:
        size = request.param
        mu, sigma = torch.zeros(size), torch.ones(size)
        return Normal(mu, sigma)

    def test_sample(self, distribution: Normal) -> None:
        z, epsilon = distribution.sample()
        assert z.shape == distribution.mu.shape
        assert epsilon.shape == distribution.mu.shape
        assert torch.allclose(z, distribution.mu + distribution.sigma * epsilon)

    @pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 1e6])
    def test_nll(self, distribution: Normal, x: float) -> None:
        nll = distribution.nll(torch.full(distribution.mu.shape, x))
        assert nll.shape == distribution.mu.shape
        assert nll.shape == distribution.sigma.shape
        assert torch.allclose(
            nll, 0.5 * ((x - distribution.mu) / distribution.sigma)**2 +
            (distribution.sigma * 2.5066282746310002).log())

    @pytest.mark.parametrize("mu, sigma", [(0.0, 1.0), (-1.0, 3.0), (1e6, 2e6)])
    def test_dkl(self, distribution: Normal, mu: float, sigma: float) -> None:
        other = Normal(
            torch.full(distribution.mu.shape, mu),
            torch.full(distribution.sigma.shape, sigma),
        )
        dkl = distribution.dkl(other)
        assert dkl.shape == distribution.mu.shape
        assert torch.allclose(
            dkl,
            torch.log(other.sigma / distribution.sigma) +
            (distribution.sigma**2 +
             (distribution.mu - other.mu)**2) / (2 * other.sigma**2) - 0.5,
        )
