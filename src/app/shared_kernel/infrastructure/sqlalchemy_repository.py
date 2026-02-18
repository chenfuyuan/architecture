from typing import List, Optional, TypeVar
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.app.shared_kernel.domain.aggregate_root import AggregateRoot
from src.app.shared_kernel.domain.entity import Id
from src.app.shared_kernel.domain.repository import Repository

Model = TypeVar("Model")


class SqlAlchemyRepository(Repository[AggregateRoot, Id]):
    def __init__(self, session: AsyncSession, model_class: type[Model]) -> None:
        self._session = session
        self._model_class = model_class

    def _to_entity(self, model: Model) -> AggregateRoot:
        raise NotImplementedError

    def _to_model(self, entity: AggregateRoot) -> Model:
        raise NotImplementedError

    async def save(self, aggregate: AggregateRoot) -> None:
        model = self._to_model(aggregate)
        self._session.add(model)
        await self._session.commit()

    async def find_by_id(self, id: Id) -> Optional[AggregateRoot]:
        result = await self._session.execute(
            select(self._model_class).where(self._model_class.id == id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_all(self) -> List[AggregateRoot]:
        result = await self._session.execute(select(self._model_class))
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]

    async def delete(self, aggregate: AggregateRoot) -> None:
        model = await self._session.get(self._model_class, aggregate.id)
        if model:
            await self._session.delete(model)
            await self._session.commit()
