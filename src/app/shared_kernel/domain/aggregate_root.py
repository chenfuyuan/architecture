from abc import ABC
from .entity import Entity, Id


class AggregateRoot(Entity[Id], ABC):
    pass
