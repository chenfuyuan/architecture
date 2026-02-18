from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

Id = TypeVar("Id", bound=Any)


@dataclass(eq=False)
class Entity(ABC, Generic[Id]):
    id: Id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return type(self) is type(other) and self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self), self.id))
