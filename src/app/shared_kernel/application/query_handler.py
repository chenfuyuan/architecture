from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .query import Query

Result = TypeVar("Result")


class QueryHandler(ABC, Generic[Query, Result]):
    @abstractmethod
    async def handle(self, query: Query) -> Result:
        pass
