import pytest
from diffusion.guidance import ClassifierFree

def test_classifier_free_initialization():
    dropout = 0.5
    strength = 1.0
    guidance = ClassifierFree(dropout=dropout, strength=strength)

    assert guidance.dropout == dropout
    assert guidance.strength == strength

if __name__ == "__main__":
    pytest.main()
