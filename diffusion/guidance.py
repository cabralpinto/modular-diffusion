from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["Base", "ClassifierFree"]


class Base(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError


@dataclass
class ClassifierFree(Base):
    dropout: float
    weight: float
