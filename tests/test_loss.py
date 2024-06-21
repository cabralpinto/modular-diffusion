import pytest
import torch
from torch import Tensor
from diffusion.base import Batch, Distribution
from diffusion.loss import Lambda, Simple, VLB

# Mock Distribution for testing
class MockDistribution(Distribution):
    def sample(self):
        return torch.zeros(2, 2), torch.ones(2, 2)

    def nll(self, x: Tensor):
        return torch.sum(x ** 2)

    def dkl(self, other: Distribution):
        return torch.sum(self.sample()[0] - other.sample()[0])

# Test Lambda class
def test_lambda_loss():
    def mock_function(batch: Batch[MockDistribution]) -> Tensor:
        return torch.tensor(1.0)
    
    lambda_loss = Lambda(mock_function)
    batch = Batch(device=torch.device('cpu'))
    batch.q = MockDistribution()
    batch.p = MockDistribution()
    assert lambda_loss.compute(batch) == torch.tensor(1.0)

# Test Simple class
def test_simple_loss():
    simple_loss = Simple(parameter="x")
    batch = Batch(device=torch.device('cpu'))
    batch.x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch.hat = [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
    assert simple_loss.compute(batch) == 0.0

# Test VLB class
def test_vlb_loss():
    vlb_loss = VLB()
    batch = Batch(device=torch.device('cpu'))
    batch.t = torch.tensor([0, 1, 2, 3])
    batch.x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch.q = MockDistribution()
    batch.p = MockDistribution()
    assert vlb_loss.compute(batch) >= 0.0  # VLB should be non-negative

if __name__ == "__main__":
    pytest.main()
