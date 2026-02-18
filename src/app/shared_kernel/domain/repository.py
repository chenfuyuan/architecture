from abc import ABC, abstractmethod
from typing import Generic, List, Optional
from .entity import Id
from .aggregate_root import AggregateRoot


class Repository(ABC, Generic[AggregateRoot, Id]):
    @abstractmethod
    async def save(self, aggregate: AggregateRoot) -> None:
        pass

    @abstractmethod
    async def find_by_id(self, id: Id) -> Optional[AggregateRoot]:
        pass

    @abstractmethod
    async def find_all(self) -> List[AggregateRoot]:
        pass

    @abstractmethod
    async def delete(self, aggregate: AggregateRoot) -> None:
        pass
