from collections.abc import AsyncGenerator

from fastapi import Request

from app.shared_kernel.application.mediator import Mediator
from app.shared_kernel.infrastructure.database import Database
from app.shared_kernel.infrastructure.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_mediator(request: Request) -> Mediator:
    return request.app.state.mediator


async def get_uow(request: Request) -> AsyncGenerator[SqlAlchemyUnitOfWork, None]:
    db: Database = request.app.state.db
    async with SqlAlchemyUnitOfWork(db.session_factory) as uow:
        yield uow
