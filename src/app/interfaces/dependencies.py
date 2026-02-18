from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.shared_kernel.infrastructure.database import get_session
from src.app.shared_kernel.infrastructure.logging import get_logger

logger = get_logger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session
