import pytest
from unittest.mock import AsyncMock

from app.shared_kernel.application.unit_of_work import UnitOfWork


@pytest.fixture
def mock_uow() -> AsyncMock:
    uow = AsyncMock(spec=UnitOfWork)
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    return uow
