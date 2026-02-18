from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class ValueObject(ABC):
    pass
