import pytest
import torch
from diffusion.net import UNet, Transformer

# Tests for UNet class
def test_unet_forward():
    channels = (3, 64, 128, 256)
    unet = UNet(channels)
    x = torch.randn(1, 3, 64, 64)
    y = torch.tensor([0])
    t = torch.tensor([1.0])
    output = unet(x, y, t)
    assert output.shape == (1, 1, 3, 64, 64) 

def test_unet_initialization():
    channels = (3, 64, 128, 256)
    unet = UNet(channels)
    assert isinstance(unet.input, torch.nn.Conv2d)
    assert isinstance(unet.output[0], torch.nn.Conv2d)

def test_unet_block():
    block = UNet.Block(hidden=256, channels=(64, 128), groups=16)
    x = torch.randn(1, 64, 32, 32)
    c = torch.randn(1, 256)
    output = block(x, c)
    assert output.shape == (1, 128, 32, 32)

def test_unet_attention():
    attention = UNet.Attention(hidden=256, heads=8)
    x = torch.randn(1, 256, 16, 16)
    output = attention(x)
    assert output.shape == (1, 256, 16, 16)

# Tests for Transformer class
def test_transformer_forward():
    transformer = Transformer(input=10, labels=10, parameters=1, depth=2, width=32, heads=2)
    x = torch.randn(1, 10, 10)
    y = torch.tensor([0])
    t = torch.tensor([1.0])
    output = transformer(x, y, t)
    assert output.shape == (1, 1, 10, 10) 

def test_transformer_initialization():
    transformer = Transformer(input=10, labels=10, parameters=1, depth=2, width=32, heads=2)
    assert isinstance(transformer.linear1, torch.nn.Linear)
    assert isinstance(transformer.linear2, torch.nn.Linear)
    assert isinstance(transformer.blocks[0], Transformer.Block)

def test_transformer_block():
    block = Transformer.Block(width=256, heads=8)
    x = torch.randn(1, 10, 256)
    c = torch.randn(1, 256)
    output = block(x, c)
    assert output.shape == (1, 10, 256)

if __name__ == "__main__":
    pytest.main()
