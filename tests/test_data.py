import pytest
import torch
from diffusion.data import Identity, OneHot, Embedding

# Tests for Identity class
def test_identity_encode_decode():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    identity = Identity(data)
    encoded = identity.encode(data)
    decoded = identity.decode(encoded)
    assert torch.equal(data, encoded)
    assert torch.equal(data, decoded)

# Tests for OneHot class
def test_onehot_encode_decode():
    data = torch.tensor([0, 1, 2])
    onehot = OneHot(data, k=3)
    encoded = onehot.encode(data)
    decoded = onehot.decode(encoded)
    expected_encoded = torch.eye(3)[data]
    assert torch.equal(encoded, expected_encoded)
    assert torch.equal(decoded, data)

# Tests for Embedding class
def test_embedding_encode_decode():
    data = torch.tensor([0, 1, 2])
    embedding = Embedding(data, k=3, d=2)
    encoded = embedding.encode(data)
    decoded = embedding.decode(encoded)
    assert encoded.shape == (3, 2)
    assert decoded.shape == data.shape

if __name__ == "__main__":
    pytest.main()
