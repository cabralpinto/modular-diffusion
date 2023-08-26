from dataclasses import dataclass

from .base import Guidance

__all__ = ["ClassifierFree"]


@dataclass
class ClassifierFree(Guidance):
    dropout: float
    strength: float
