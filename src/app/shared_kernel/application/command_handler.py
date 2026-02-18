from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .command import Command

Result = TypeVar("Result")


class CommandHandler(ABC, Generic[Command, Result]):
    @abstractmethod
    async def handle(self, command: Command) -> Result:
        pass
