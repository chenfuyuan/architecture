# DDD + FastAPI 骨架全面重构 - 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有骨架从"有 DDD 目录名的普通分层项目"重构为真正可用的 DDD + 整洁架构项目模板，修正所有类型错误、补齐 DDD 核心概念、增加示例模块和生产级工程化配套。

**Architecture:** 四层架构（Domain → Application → Infrastructure ← Interfaces），依赖倒置。Domain 层定义实体/值对象/聚合根/仓储接口，Application 层通过 CQRS（Command/Query + Handler）编排用例并用 UnitOfWork 管理事务，Infrastructure 层提供 SQLAlchemy 实现，Interfaces 层是 FastAPI 入口。Mediator 串联 CQRS 分发。

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy (async), PostgreSQL, asyncpg, Alembic, pydantic-settings, structlog, pytest, ruff, mypy

**Design Doc:** `docs/plans/2026-02-19-ddd-skeleton-refactor-design.md`

---

## Task 1: Entity 基类 — 修正泛型

**Files:**
- Modify: `src/app/shared_kernel/domain/entity.py`
- Test: `tests/unit/shared_kernel/test_entity.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_entity.py
from dataclasses import dataclass
from app.shared_kernel.domain.entity import Entity, ID


@dataclass(eq=False)
class FakeEntity(Entity[int]):
    name: str = ""


class TestEntity:
    def test_equal_by_id(self) -> None:
        a = FakeEntity(id=1, name="Alice")
        b = FakeEntity(id=1, name="Bob")
        assert a == b

    def test_not_equal_different_id(self) -> None:
        a = FakeEntity(id=1)
        b = FakeEntity(id=2)
        assert a != b

    def test_not_equal_different_type(self) -> None:
        @dataclass(eq=False)
        class OtherEntity(Entity[int]):
            pass

        a = FakeEntity(id=1)
        b = OtherEntity(id=1)
        assert a != b

    def test_hash_by_type_and_id(self) -> None:
        a = FakeEntity(id=1, name="Alice")
        b = FakeEntity(id=1, name="Bob")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_not_equal_to_non_entity(self) -> None:
        a = FakeEntity(id=1)
        assert a != "not an entity"
        assert a != 1
        assert a != None  # noqa: E711
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_entity.py -v`
Expected: FAIL — `ID` is not exported from entity.py (currently `Id`)

**Step 3: Write minimal implementation**

Replace `src/app/shared_kernel/domain/entity.py` with:

```python
from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ID = TypeVar("ID", bound=Any)


@dataclass(eq=False)
class Entity(ABC, Generic[ID]):
    id: ID

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self), self.id))
```

Changes from current:
- `Id` → `ID` (conventional TypeVar naming)
- `isinstance(other, Entity)` → `isinstance(other, type(self))` (stricter type check)

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_entity.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/domain/entity.py tests/unit/shared_kernel/test_entity.py
git commit -m "refactor: fix Entity generics and add unit tests"
```

---

## Task 2: DomainEvent 基类 — 新增

**Files:**
- Create: `src/app/shared_kernel/domain/domain_event.py`
- Test: `tests/unit/shared_kernel/test_domain_event.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_domain_event.py
from dataclasses import dataclass
from datetime import datetime, timezone

from app.shared_kernel.domain.domain_event import DomainEvent


@dataclass(frozen=True)
class FakeEvent(DomainEvent):
    entity_id: int = 0


class TestDomainEvent:
    def test_has_occurred_at(self) -> None:
        event = FakeEvent(entity_id=42)
        assert isinstance(event.occurred_at, datetime)

    def test_is_immutable(self) -> None:
        event = FakeEvent(entity_id=42)
        try:
            event.entity_id = 99  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass

    def test_equality_by_value(self) -> None:
        ts = datetime.now(timezone.utc)
        a = FakeEvent(entity_id=1, occurred_at=ts)
        b = FakeEvent(entity_id=1, occurred_at=ts)
        assert a == b
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_domain_event.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.shared_kernel.domain.domain_event'`

**Step 3: Write minimal implementation**

```python
# src/app/shared_kernel/domain/domain_event.py
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class DomainEvent:
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_domain_event.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/domain/domain_event.py tests/unit/shared_kernel/test_domain_event.py
git commit -m "feat: add DomainEvent base class with tests"
```

---

## Task 3: ValueObject 基类 — 增加验证钩子

**Files:**
- Modify: `src/app/shared_kernel/domain/value_object.py`
- Test: `tests/unit/shared_kernel/test_value_object.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_value_object.py
import pytest
from dataclasses import dataclass

from app.shared_kernel.domain.value_object import ValueObject
from app.shared_kernel.domain.exception import ValidationException


@dataclass(frozen=True)
class Email(ValueObject):
    address: str = ""

    def _validate(self) -> None:
        if self.address and "@" not in self.address:
            raise ValidationException(f"Invalid email: {self.address}")


class TestValueObject:
    def test_is_immutable(self) -> None:
        vo = Email(address="a@b.com")
        with pytest.raises(AttributeError):
            vo.address = "x@y.com"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = Email(address="a@b.com")
        b = Email(address="a@b.com")
        assert a == b

    def test_inequality_by_value(self) -> None:
        a = Email(address="a@b.com")
        b = Email(address="x@y.com")
        assert a != b

    def test_validation_hook_called(self) -> None:
        with pytest.raises(ValidationException, match="Invalid email"):
            Email(address="not-an-email")

    def test_validation_hook_passes(self) -> None:
        email = Email(address="valid@test.com")
        assert email.address == "valid@test.com"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_value_object.py -v`
Expected: FAIL — `test_validation_hook_called` fails because `_validate()` is not called

**Step 3: Write minimal implementation**

Replace `src/app/shared_kernel/domain/value_object.py` with:

```python
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class ValueObject(ABC):
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        pass
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_value_object.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/domain/value_object.py tests/unit/shared_kernel/test_value_object.py
git commit -m "refactor: add validation hook to ValueObject base class"
```

---

## Task 4: AggregateRoot 基类 — 增加事件收集

**Files:**
- Modify: `src/app/shared_kernel/domain/aggregate_root.py`
- Test: `tests/unit/shared_kernel/test_aggregate_root.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_aggregate_root.py
from dataclasses import dataclass, field

from app.shared_kernel.domain.aggregate_root import AggregateRoot
from app.shared_kernel.domain.domain_event import DomainEvent


@dataclass(frozen=True)
class ThingCreated(DomainEvent):
    thing_id: int = 0


@dataclass(eq=False)
class Thing(AggregateRoot[int]):
    name: str = ""

    @classmethod
    def create(cls, id: int, name: str) -> "Thing":
        thing = cls(id=id, name=name)
        thing.add_event(ThingCreated(thing_id=id))
        return thing


class TestAggregateRoot:
    def test_is_entity(self) -> None:
        thing = Thing(id=1, name="test")
        assert thing.id == 1

    def test_add_and_collect_events(self) -> None:
        thing = Thing.create(id=1, name="test")
        events = thing.collect_events()
        assert len(events) == 1
        assert isinstance(events[0], ThingCreated)
        assert events[0].thing_id == 1

    def test_collect_events_clears_list(self) -> None:
        thing = Thing.create(id=1, name="test")
        thing.collect_events()
        assert thing.collect_events() == []

    def test_no_events_initially(self) -> None:
        thing = Thing(id=1, name="test")
        assert thing.collect_events() == []

    def test_multiple_events(self) -> None:
        thing = Thing(id=1, name="test")
        thing.add_event(ThingCreated(thing_id=1))
        thing.add_event(ThingCreated(thing_id=1))
        events = thing.collect_events()
        assert len(events) == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_aggregate_root.py -v`
Expected: FAIL — `AggregateRoot` has no `add_event` method

**Step 3: Write minimal implementation**

Replace `src/app/shared_kernel/domain/aggregate_root.py` with:

```python
from abc import ABC
from dataclasses import dataclass, field

from .domain_event import DomainEvent
from .entity import Entity, ID


@dataclass(eq=False)
class AggregateRoot(Entity[ID], ABC):
    _events: list[DomainEvent] = field(default_factory=list, init=False, repr=False)

    def add_event(self, event: DomainEvent) -> None:
        self._events.append(event)

    def collect_events(self) -> list[DomainEvent]:
        events = self._events.copy()
        self._events.clear()
        return events
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_aggregate_root.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/domain/aggregate_root.py tests/unit/shared_kernel/test_aggregate_root.py
git commit -m "refactor: add domain event collection to AggregateRoot"
```

---

## Task 5: Repository 基类 — 修正泛型

**Files:**
- Modify: `src/app/shared_kernel/domain/repository.py`

No test file needed — Repository is a pure abstract interface. Its correctness is validated by mypy and by integration tests with concrete implementations.

**Step 1: Replace implementation**

Replace `src/app/shared_kernel/domain/repository.py` with:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .aggregate_root import AggregateRoot

AR = TypeVar("AR", bound=AggregateRoot)
ID = TypeVar("ID")


class Repository(ABC, Generic[AR, ID]):
    @abstractmethod
    async def find_by_id(self, id: ID) -> AR | None:
        pass

    @abstractmethod
    async def save(self, aggregate: AR) -> None:
        pass

    @abstractmethod
    async def delete(self, aggregate: AR) -> None:
        pass
```

Changes from current:
- `Generic[AggregateRoot, Id]` → `Generic[AR, ID]` with proper TypeVar bounds
- Removed `find_all()` (ISP: define per concrete repository as needed)
- `Optional[AR]` → `AR | None` (Python 3.11 style)

**Step 2: Run all domain tests to verify nothing broke**

Run: `python -m pytest tests/unit/shared_kernel/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/app/shared_kernel/domain/repository.py
git commit -m "refactor: fix Repository generic types, remove find_all"
```

---

## Task 6: Command/Query Handler 基类 — 修正泛型

**Files:**
- Modify: `src/app/shared_kernel/application/command_handler.py`
- Modify: `src/app/shared_kernel/application/query_handler.py`

**Step 1: Replace CommandHandler**

Replace `src/app/shared_kernel/application/command_handler.py` with:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .command import Command

C = TypeVar("C", bound=Command)
R = TypeVar("R")


class CommandHandler(ABC, Generic[C, R]):
    @abstractmethod
    async def handle(self, command: C) -> R:
        pass
```

**Step 2: Replace QueryHandler**

Replace `src/app/shared_kernel/application/query_handler.py` with:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .query import Query

Q = TypeVar("Q", bound=Query)
R = TypeVar("R")


class QueryHandler(ABC, Generic[Q, R]):
    @abstractmethod
    async def handle(self, query: Q) -> R:
        pass
```

**Step 3: Run all existing tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/app/shared_kernel/application/command_handler.py src/app/shared_kernel/application/query_handler.py
git commit -m "refactor: fix CommandHandler and QueryHandler generic types"
```

---

## Task 7: UnitOfWork 抽象 — 新增

**Files:**
- Create: `src/app/shared_kernel/application/unit_of_work.py`
- Test: `tests/unit/shared_kernel/test_unit_of_work.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_unit_of_work.py
import pytest
from app.shared_kernel.application.unit_of_work import UnitOfWork


class FakeUnitOfWork(UnitOfWork):
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> "FakeUnitOfWork":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if exc_type:
            await self.rollback()

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class TestUnitOfWork:
    async def test_commit(self) -> None:
        async with FakeUnitOfWork() as uow:
            await uow.commit()
        assert uow.committed is True

    async def test_rollback_on_exception(self) -> None:
        uow = FakeUnitOfWork()
        with pytest.raises(ValueError):
            async with uow:
                raise ValueError("boom")
        assert uow.rolled_back is True

    async def test_context_manager_returns_self(self) -> None:
        async with FakeUnitOfWork() as uow:
            assert isinstance(uow, FakeUnitOfWork)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_unit_of_work.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/app/shared_kernel/application/unit_of_work.py
from abc import ABC, abstractmethod
from typing import Self


class UnitOfWork(ABC):
    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        pass

    @abstractmethod
    async def commit(self) -> None:
        pass

    @abstractmethod
    async def rollback(self) -> None:
        pass
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_unit_of_work.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/application/unit_of_work.py tests/unit/shared_kernel/test_unit_of_work.py
git commit -m "feat: add UnitOfWork abstract base class with tests"
```

---

## Task 8: Mediator — 新增

**Files:**
- Create: `src/app/shared_kernel/application/mediator.py`
- Test: `tests/unit/shared_kernel/test_mediator.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_mediator.py
import pytest
from dataclasses import dataclass

from app.shared_kernel.application.command import Command
from app.shared_kernel.application.command_handler import CommandHandler
from app.shared_kernel.application.query import Query
from app.shared_kernel.application.query_handler import QueryHandler
from app.shared_kernel.application.mediator import Mediator


@dataclass(frozen=True)
class AddNumbers(Command):
    a: int = 0
    b: int = 0


class AddNumbersHandler(CommandHandler[AddNumbers, int]):
    async def handle(self, command: AddNumbers) -> int:
        return command.a + command.b


@dataclass(frozen=True)
class GetGreeting(Query):
    name: str = ""


class GetGreetingHandler(QueryHandler[GetGreeting, str]):
    async def handle(self, query: GetGreeting) -> str:
        return f"Hello, {query.name}!"


class TestMediator:
    def _make_mediator(self) -> Mediator:
        mediator = Mediator()
        mediator.register_command_handler(AddNumbers, lambda: AddNumbersHandler())
        mediator.register_query_handler(GetGreeting, lambda: GetGreetingHandler())
        return mediator

    async def test_send_command(self) -> None:
        mediator = self._make_mediator()
        result = await mediator.send(AddNumbers(a=2, b=3))
        assert result == 5

    async def test_send_query(self) -> None:
        mediator = self._make_mediator()
        result = await mediator.query(GetGreeting(name="World"))
        assert result == "Hello, World!"

    async def test_unregistered_command_raises(self) -> None:
        mediator = Mediator()
        with pytest.raises(KeyError):
            await mediator.send(AddNumbers(a=1, b=2))

    async def test_unregistered_query_raises(self) -> None:
        mediator = Mediator()
        with pytest.raises(KeyError):
            await mediator.query(GetGreeting(name="Test"))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_mediator.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/app/shared_kernel/application/mediator.py
from typing import Any, Callable

from .command import Command
from .command_handler import CommandHandler
from .query import Query
from .query_handler import QueryHandler


class Mediator:
    def __init__(self) -> None:
        self._command_handlers: dict[type[Command], Callable[[], CommandHandler[Any, Any]]] = {}
        self._query_handlers: dict[type[Query], Callable[[], QueryHandler[Any, Any]]] = {}

    def register_command_handler(
        self,
        command_type: type[Command],
        factory: Callable[[], CommandHandler[Any, Any]],
    ) -> None:
        self._command_handlers[command_type] = factory

    def register_query_handler(
        self,
        query_type: type[Query],
        factory: Callable[[], QueryHandler[Any, Any]],
    ) -> None:
        self._query_handlers[query_type] = factory

    async def send(self, command: Command) -> Any:
        handler_factory = self._command_handlers[type(command)]
        handler = handler_factory()
        return await handler.handle(command)

    async def query(self, query: Query) -> Any:
        handler_factory = self._query_handlers[type(query)]
        handler = handler_factory()
        return await handler.handle(query)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_mediator.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/application/mediator.py tests/unit/shared_kernel/test_mediator.py
git commit -m "feat: add Mediator for CQRS command/query dispatching"
```

---

## Task 9: Database 类 — 重构为延迟初始化

**Files:**
- Modify: `src/app/shared_kernel/infrastructure/database.py`
- Test: `tests/unit/shared_kernel/test_database.py`

**Step 1: Write the failing test**

```python
# tests/unit/shared_kernel/test_database.py
from app.shared_kernel.infrastructure.database import Database


class TestDatabase:
    def test_creates_engine(self) -> None:
        db = Database(url="sqlite+aiosqlite:///:memory:")
        assert db.session_factory is not None

    async def test_dispose(self) -> None:
        db = Database(url="sqlite+aiosqlite:///:memory:")
        await db.dispose()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/shared_kernel/test_database.py -v`
Expected: FAIL — `Database` class does not exist (current code has module-level globals)

**Step 3: Write minimal implementation**

Replace `src/app/shared_kernel/infrastructure/database.py` with:

```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        self._engine = create_async_engine(url, echo=echo)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def check_connection(self) -> None:
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def dispose(self) -> None:
        await self._engine.dispose()
```

Changes from current:
- Module-level globals → `Database` class with explicit init
- `declarative_base()` (legacy) → `DeclarativeBase` (modern SQLAlchemy 2.0)
- Added `check_connection()` for health check
- Removed `get_session()` generator (sessions now created via UoW)

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/shared_kernel/test_database.py -v`
Expected: All 2 tests PASS

Note: Test requires `aiosqlite` package. Add to dev dependencies in pyproject.toml (done in Task 13).

**Step 5: Commit**

```bash
git add src/app/shared_kernel/infrastructure/database.py tests/unit/shared_kernel/test_database.py
git commit -m "refactor: Database as class with lazy init, add health check"
```

---

## Task 10: SqlAlchemyUnitOfWork — 新增

**Files:**
- Create: `src/app/shared_kernel/infrastructure/sqlalchemy_unit_of_work.py`
- Test: `tests/integration/test_sqlalchemy_uow.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_sqlalchemy_uow.py
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession

from app.shared_kernel.infrastructure.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


class TestSqlAlchemyUnitOfWork:
    async def test_provides_session(self, session_factory) -> None:
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            assert uow.session is not None

    async def test_commit(self, session_factory) -> None:
        async with SqlAlchemyUnitOfWork(session_factory) as uow:
            await uow.commit()

    async def test_rollback_on_exception(self, session_factory) -> None:
        with pytest.raises(ValueError):
            async with SqlAlchemyUnitOfWork(session_factory) as uow:
                raise ValueError("boom")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_sqlalchemy_uow.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/app/shared_kernel/infrastructure/sqlalchemy_unit_of_work.py
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.app.shared_kernel.application.unit_of_work import UnitOfWork


class SqlAlchemyUnitOfWork(UnitOfWork):
    session: AsyncSession

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def __aenter__(self) -> Self:
        self.session = self._session_factory()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type:
            await self.rollback()
        await self.session.close()

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_sqlalchemy_uow.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/app/shared_kernel/infrastructure/sqlalchemy_unit_of_work.py tests/integration/test_sqlalchemy_uow.py
git commit -m "feat: add SqlAlchemyUnitOfWork implementation"
```

---

## Task 11: SqlAlchemyRepository — 重构

**Files:**
- Modify: `src/app/shared_kernel/infrastructure/sqlalchemy_repository.py`

**Step 1: Replace implementation**

Replace `src/app/shared_kernel/infrastructure/sqlalchemy_repository.py` with:

```python
from abc import abstractmethod
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.shared_kernel.domain.aggregate_root import AggregateRoot
from src.app.shared_kernel.domain.repository import Repository

AR = TypeVar("AR", bound=AggregateRoot)
ID = TypeVar("ID")


class SqlAlchemyRepository(Repository[AR, ID]):
    def __init__(self, session: AsyncSession, model_class: type) -> None:
        self._session = session
        self._model_class = model_class

    @abstractmethod
    def _to_entity(self, model: Any) -> AR:
        pass

    @abstractmethod
    def _to_model(self, entity: AR) -> Any:
        pass

    async def save(self, aggregate: AR) -> None:
        model = self._to_model(aggregate)
        self._session.add(model)

    async def find_by_id(self, id: ID) -> AR | None:
        result = await self._session.execute(
            select(self._model_class).where(self._model_class.id == id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def delete(self, aggregate: AR) -> None:
        model = await self._session.get(self._model_class, aggregate.id)
        if model:
            await self._session.delete(model)
```

Changes from current:
- Fixed generic types (`AR`, `ID` properly bound TypeVars)
- `save()` no longer calls `commit()` — UoW manages transactions
- `delete()` no longer calls `commit()`
- Removed `find_all()` (removed from Repository interface)
- `_to_entity()` / `_to_model()` now `@abstractmethod` (not `raise NotImplementedError`)
- Import `select` from `sqlalchemy` (not `sqlalchemy.future`)

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/app/shared_kernel/infrastructure/sqlalchemy_repository.py
git commit -m "refactor: fix SqlAlchemyRepository generics, remove auto-commit"
```

---

## Task 12: Config & Dependencies — 修正

**Files:**
- Modify: `src/app/config.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `alembic.ini`

**Step 1: Update config.py**

Replace `src/app/config.py` with:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    APP_NAME: str = "myapp"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/myapp"

    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    LOG_LEVEL: str = "DEBUG"


settings = Settings()
```

**Step 2: Update pyproject.toml**

Replace `pyproject.toml` with:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "app"
version = "0.1.0"
description = "DDD-based FastAPI project skeleton"
requires-python = ">=3.11"
authors = [
  { name = "Developer" }
]
dependencies = [
  "fastapi>=0.109.0",
  "uvicorn[standard]>=0.27.0",
  "sqlalchemy>=2.0.25",
  "alembic>=1.13.0",
  "asyncpg>=0.29.0",
  "pydantic-settings>=2.1.0",
  "structlog>=24.1.0",
  "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4.0",
  "pytest-asyncio>=0.23.0",
  "httpx>=0.26.0",
  "aiosqlite>=0.20.0",
  "faker>=22.0.0",
  "ruff>=0.1.0",
  "mypy>=1.8.0",
  "pre-commit>=3.6.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]
```

Changes: `psycopg[binary,pool]` → `asyncpg`, added `aiosqlite` for tests, added `pre-commit`, added ruff lint rules, added pydantic mypy plugin.

**Step 3: Update .env.example**

Replace `.env.example` with:

```
# Application
APP_NAME=myapp
APP_ENV=development
APP_DEBUG=true
APP_HOST=0.0.0.0
APP_PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/myapp

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Logging
LOG_LEVEL=DEBUG
```

**Step 4: Update alembic.ini**

In `alembic.ini`, replace line 4:
```
sqlalchemy.url = postgresql+asyncpg://user:password@localhost:5432/ecommerce
```
with:
```
sqlalchemy.url = driver://user:pass@localhost/dbname
```

This is just a placeholder — `migrations/env.py` already overrides it with `settings.DATABASE_URL`.

**Step 5: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/app/config.py pyproject.toml .env.example alembic.ini
git commit -m "fix: replace ecommerce with myapp, fix dependencies (asyncpg), add CORS config"
```

---

## Task 13: Interfaces — middleware + exception_handler + dependencies

**Files:**
- Modify: `src/app/interfaces/middleware.py`
- Modify: `src/app/interfaces/exception_handler.py`
- Modify: `src/app/interfaces/dependencies.py`
- Delete: `src/app/interfaces/scheduler.py`

**Step 1: Update middleware.py**

Replace `src/app/interfaces/middleware.py` with:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.config import settings


def setup_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

**Step 2: Update exception_handler.py**

Replace `src/app/interfaces/exception_handler.py` with:

```python
import traceback

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.app.interfaces.response import ApiResponse
from src.app.shared_kernel.domain.exception import (
    DomainException,
    NotFoundException,
    ValidationException,
)
from src.app.shared_kernel.infrastructure.logging import get_logger

logger = get_logger(__name__)


async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
    if isinstance(exc, NotFoundException):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, ValidationException):
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        status_code = status.HTTP_400_BAD_REQUEST

    response = ApiResponse.error(code=status_code, message=exc.message)
    return JSONResponse(content=response.model_dump(), status_code=status_code)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    error_messages = []
    for err in errors:
        loc = " -> ".join([str(x) for x in err["loc"] if x != "body"])
        error_messages.append(f"{loc}: {err['msg']}")

    message = "; ".join(error_messages) if error_messages else "Validation error"
    response = ApiResponse.error(code=status.HTTP_422_UNPROCESSABLE_ENTITY, message=message)
    return JSONResponse(content=response.model_dump(), status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", exc_info=True, traceback=traceback.format_exc())
    response = ApiResponse.error(
        code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Internal server error"
    )
    return JSONResponse(content=response.model_dump(), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

**Step 3: Update dependencies.py**

Replace `src/app/interfaces/dependencies.py` with:

```python
from typing import AsyncGenerator

from fastapi import Request

from src.app.shared_kernel.application.mediator import Mediator
from src.app.shared_kernel.infrastructure.database import Database
from src.app.shared_kernel.infrastructure.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_mediator(request: Request) -> Mediator:
    return request.app.state.mediator


async def get_uow(request: Request) -> AsyncGenerator[SqlAlchemyUnitOfWork, None]:
    db: Database = request.app.state.db
    async with SqlAlchemyUnitOfWork(db.session_factory) as uow:
        yield uow
```

**Step 4: Delete scheduler.py**

Delete `src/app/interfaces/scheduler.py` (empty file, not needed).

**Step 5: Commit**

```bash
git add src/app/interfaces/middleware.py src/app/interfaces/exception_handler.py src/app/interfaces/dependencies.py
git rm src/app/interfaces/scheduler.py
git commit -m "refactor: CORS from config, add logging to errors, add UoW/Mediator deps"
```

---

## Task 14: main.py — 重构 lifespan 和路由注册

**Files:**
- Modify: `src/app/interfaces/main.py`

**Step 1: Replace implementation**

Replace `src/app/interfaces/main.py` with:

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from src.app.config import settings
from src.app.interfaces.exception_handler import (
    domain_exception_handler,
    general_exception_handler,
    validation_exception_handler,
)
from src.app.interfaces.middleware import setup_middleware
from src.app.interfaces.response import ApiResponse
from src.app.shared_kernel.application.mediator import Mediator
from src.app.shared_kernel.domain.exception import DomainException
from src.app.shared_kernel.infrastructure.database import Database
from src.app.shared_kernel.infrastructure.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _register_handlers(mediator: Mediator) -> None:
    """Register all module command/query handlers with the mediator."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("Application starting up", app_name=settings.APP_NAME, env=settings.APP_ENV)

    db = Database(url=settings.DATABASE_URL, echo=settings.APP_DEBUG)
    app.state.db = db

    mediator = Mediator()
    _register_handlers(mediator)
    app.state.mediator = mediator

    yield

    await db.dispose()
    logger.info("Application shut down")


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    docs_url="/docs" if settings.APP_DEBUG else None,
    redoc_url="/redoc" if settings.APP_DEBUG else None,
)

setup_middleware(app)

app.add_exception_handler(DomainException, domain_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.get("/health", response_model=ApiResponse[dict])
async def health_check() -> ApiResponse[dict]:
    return ApiResponse.success(data={"status": "healthy"})
```

Note: `_register_handlers` is a placeholder — Task 18 will wire up the example module's handlers.

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/app/interfaces/main.py
git commit -m "refactor: main.py lifespan with Database/Mediator initialization"
```

---

## Task 15: Example Module — Domain Layer

**Files:**
- Create: `src/app/modules/example/__init__.py`
- Create: `src/app/modules/example/domain/__init__.py`
- Create: `src/app/modules/example/domain/note.py`
- Create: `src/app/modules/example/domain/note_created.py`
- Create: `src/app/modules/example/domain/note_repository.py`
- Test: `tests/unit/modules/example/test_note.py`

**Step 1: Write the failing test**

Create `tests/unit/modules/__init__.py`, `tests/unit/modules/example/__init__.py` (empty), then:

```python
# tests/unit/modules/example/test_note.py
import pytest

from app.modules.example.domain.note import Note
from app.modules.example.domain.note_created import NoteCreated


class TestNote:
    def test_create_note(self) -> None:
        note = Note.create(title="Hello", content="World")
        assert note.title == "Hello"
        assert note.content == "World"
        assert note.id is not None

    def test_create_emits_event(self) -> None:
        note = Note.create(title="Hello", content="World")
        events = note.collect_events()
        assert len(events) == 1
        assert isinstance(events[0], NoteCreated)
        assert events[0].note_id == note.id

    def test_create_empty_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            Note.create(title="", content="World")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/modules/example/test_note.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementations**

```python
# src/app/modules/example/__init__.py
```

```python
# src/app/modules/example/domain/__init__.py
```

```python
# src/app/modules/example/domain/note_created.py
from dataclasses import dataclass
from uuid import UUID

from src.app.shared_kernel.domain.domain_event import DomainEvent


@dataclass(frozen=True)
class NoteCreated(DomainEvent):
    note_id: UUID | None = None
    title: str = ""
```

```python
# src/app/modules/example/domain/note.py
from dataclasses import dataclass
from uuid import UUID, uuid4

from src.app.shared_kernel.domain.aggregate_root import AggregateRoot
from .note_created import NoteCreated


@dataclass(eq=False)
class Note(AggregateRoot[UUID]):
    title: str = ""
    content: str = ""

    @classmethod
    def create(cls, title: str, content: str) -> "Note":
        if not title.strip():
            raise ValueError("Note title must not be empty")
        note = cls(id=uuid4(), title=title, content=content)
        note.add_event(NoteCreated(note_id=note.id, title=title))
        return note
```

```python
# src/app/modules/example/domain/note_repository.py
from abc import abstractmethod
from uuid import UUID

from src.app.shared_kernel.domain.repository import Repository
from .note import Note


class NoteRepository(Repository[Note, UUID]):
    @abstractmethod
    async def find_by_id(self, id: UUID) -> Note | None:
        pass

    @abstractmethod
    async def save(self, aggregate: Note) -> None:
        pass

    @abstractmethod
    async def delete(self, aggregate: Note) -> None:
        pass
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/modules/example/test_note.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/app/modules/example/ tests/unit/modules/
git commit -m "feat: add example module domain layer (Note, NoteCreated, NoteRepository)"
```

---

## Task 16: Example Module — Infrastructure Layer

**Files:**
- Create: `src/app/modules/example/infrastructure/__init__.py`
- Create: `src/app/modules/example/infrastructure/models/__init__.py`
- Create: `src/app/modules/example/infrastructure/models/note_model.py`
- Create: `src/app/modules/example/infrastructure/sqlalchemy_note_repository.py`

**Step 1: Write implementations**

```python
# src/app/modules/example/infrastructure/__init__.py
```

```python
# src/app/modules/example/infrastructure/models/__init__.py
```

```python
# src/app/modules/example/infrastructure/models/note_model.py
import uuid

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.app.shared_kernel.infrastructure.database import Base


class NoteModel(Base):
    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
```

```python
# src/app/modules/example/infrastructure/sqlalchemy_note_repository.py
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.app.modules.example.domain.note import Note
from src.app.modules.example.domain.note_repository import NoteRepository
from src.app.shared_kernel.infrastructure.sqlalchemy_repository import SqlAlchemyRepository
from .models.note_model import NoteModel


class SqlAlchemyNoteRepository(SqlAlchemyRepository[Note, UUID], NoteRepository):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, NoteModel)

    def _to_entity(self, model: Any) -> Note:
        return Note(id=model.id, title=model.title, content=model.content)

    def _to_model(self, entity: Note) -> Any:
        return NoteModel(id=entity.id, title=entity.title, content=entity.content)
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/app/modules/example/infrastructure/
git commit -m "feat: add example module infrastructure (NoteModel, SqlAlchemyNoteRepository)"
```

---

## Task 17: Example Module — Application Layer

**Files:**
- Create: `src/app/modules/example/application/__init__.py`
- Create: `src/app/modules/example/application/commands/__init__.py`
- Create: `src/app/modules/example/application/commands/create_note.py`
- Create: `src/app/modules/example/application/commands/create_note_handler.py`
- Create: `src/app/modules/example/application/queries/__init__.py`
- Create: `src/app/modules/example/application/queries/get_note.py`
- Create: `src/app/modules/example/application/queries/get_note_handler.py`
- Test: `tests/unit/modules/example/test_create_note_handler.py`

**Step 1: Write the failing test**

```python
# tests/unit/modules/example/test_create_note_handler.py
import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.modules.example.application.commands.create_note import CreateNoteCommand
from app.modules.example.application.commands.create_note_handler import CreateNoteHandler
from app.modules.example.domain.note_repository import NoteRepository


class TestCreateNoteHandler:
    async def test_creates_and_saves_note(self) -> None:
        repo = AsyncMock(spec=NoteRepository)
        handler = CreateNoteHandler(repository=repo)

        result = await handler.handle(CreateNoteCommand(title="Test", content="Body"))

        assert result is not None
        repo.save.assert_called_once()
        saved_note = repo.save.call_args[0][0]
        assert saved_note.title == "Test"
        assert saved_note.content == "Body"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/modules/example/test_create_note_handler.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementations**

```python
# src/app/modules/example/application/__init__.py
```

```python
# src/app/modules/example/application/commands/__init__.py
```

```python
# src/app/modules/example/application/commands/create_note.py
from dataclasses import dataclass

from src.app.shared_kernel.application.command import Command


@dataclass(frozen=True)
class CreateNoteCommand(Command):
    title: str = ""
    content: str = ""
```

```python
# src/app/modules/example/application/commands/create_note_handler.py
from uuid import UUID

from src.app.modules.example.domain.note import Note
from src.app.modules.example.domain.note_repository import NoteRepository
from src.app.shared_kernel.application.command_handler import CommandHandler
from .create_note import CreateNoteCommand


class CreateNoteHandler(CommandHandler[CreateNoteCommand, UUID]):
    def __init__(self, repository: NoteRepository) -> None:
        self._repository = repository

    async def handle(self, command: CreateNoteCommand) -> UUID:
        note = Note.create(title=command.title, content=command.content)
        await self._repository.save(note)
        return note.id
```

```python
# src/app/modules/example/application/queries/__init__.py
```

```python
# src/app/modules/example/application/queries/get_note.py
from dataclasses import dataclass
from uuid import UUID

from src.app.shared_kernel.application.query import Query


@dataclass(frozen=True)
class GetNoteQuery(Query):
    note_id: UUID | None = None
```

```python
# src/app/modules/example/application/queries/get_note_handler.py
from dataclasses import dataclass
from uuid import UUID

from src.app.modules.example.domain.note_repository import NoteRepository
from src.app.shared_kernel.application.query_handler import QueryHandler
from src.app.shared_kernel.domain.exception import NotFoundException
from .get_note import GetNoteQuery


@dataclass
class NoteReadModel:
    id: UUID
    title: str
    content: str


class GetNoteHandler(QueryHandler[GetNoteQuery, NoteReadModel]):
    def __init__(self, repository: NoteRepository) -> None:
        self._repository = repository

    async def handle(self, query: GetNoteQuery) -> NoteReadModel:
        note = await self._repository.find_by_id(query.note_id)
        if note is None:
            raise NotFoundException(f"Note {query.note_id} not found")
        return NoteReadModel(id=note.id, title=note.title, content=note.content)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/modules/example/test_create_note_handler.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/app/modules/example/application/ tests/unit/modules/example/test_create_note_handler.py
git commit -m "feat: add example module application layer (CreateNote, GetNote commands/queries)"
```

---

## Task 18: Example Module — Interfaces Layer + Wire Up

**Files:**
- Create: `src/app/modules/example/interfaces/__init__.py`
- Create: `src/app/modules/example/interfaces/api/__init__.py`
- Create: `src/app/modules/example/interfaces/api/requests/__init__.py`
- Create: `src/app/modules/example/interfaces/api/requests/create_note_request.py`
- Create: `src/app/modules/example/interfaces/api/responses/__init__.py`
- Create: `src/app/modules/example/interfaces/api/responses/note_response.py`
- Create: `src/app/modules/example/interfaces/api/note_router.py`
- Modify: `src/app/interfaces/main.py` (wire up router + handlers)

**Step 1: Write implementations**

```python
# src/app/modules/example/interfaces/__init__.py
```

```python
# src/app/modules/example/interfaces/api/__init__.py
```

```python
# src/app/modules/example/interfaces/api/requests/__init__.py
```

```python
# src/app/modules/example/interfaces/api/requests/create_note_request.py
from pydantic import BaseModel, Field


class CreateNoteRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(default="")
```

```python
# src/app/modules/example/interfaces/api/responses/__init__.py
```

```python
# src/app/modules/example/interfaces/api/responses/note_response.py
from uuid import UUID

from pydantic import BaseModel


class NoteResponse(BaseModel):
    id: UUID
    title: str
    content: str
```

```python
# src/app/modules/example/interfaces/api/note_router.py
from uuid import UUID

from fastapi import APIRouter, Depends

from src.app.interfaces.dependencies import get_mediator, get_uow
from src.app.interfaces.response import ApiResponse
from src.app.modules.example.application.commands.create_note import CreateNoteCommand
from src.app.modules.example.application.queries.get_note import GetNoteQuery
from src.app.shared_kernel.application.mediator import Mediator
from src.app.shared_kernel.infrastructure.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from .requests.create_note_request import CreateNoteRequest
from .responses.note_response import NoteResponse

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post("", response_model=ApiResponse[dict])
async def create_note(
    body: CreateNoteRequest,
    mediator: Mediator = Depends(get_mediator),
    uow: SqlAlchemyUnitOfWork = Depends(get_uow),
) -> ApiResponse[dict]:
    note_id = await mediator.send(CreateNoteCommand(title=body.title, content=body.content))
    await uow.commit()
    return ApiResponse.success(data={"id": str(note_id)}, message="Note created")


@router.get("/{note_id}", response_model=ApiResponse[NoteResponse])
async def get_note(
    note_id: UUID,
    mediator: Mediator = Depends(get_mediator),
) -> ApiResponse[NoteResponse]:
    result = await mediator.query(GetNoteQuery(note_id=note_id))
    return ApiResponse.success(data=NoteResponse(id=result.id, title=result.title, content=result.content))
```

**Step 2: Wire up in main.py**

Update `_register_handlers` and add the router in `src/app/interfaces/main.py`.

Add these imports at top:

```python
from src.app.modules.example.application.commands.create_note import CreateNoteCommand
from src.app.modules.example.application.commands.create_note_handler import CreateNoteHandler
from src.app.modules.example.application.queries.get_note import GetNoteQuery
from src.app.modules.example.application.queries.get_note_handler import GetNoteHandler
from src.app.modules.example.domain.note_repository import NoteRepository
from src.app.modules.example.infrastructure.sqlalchemy_note_repository import SqlAlchemyNoteRepository
from src.app.modules.example.interfaces.api.note_router import router as note_router
```

Replace `_register_handlers` with:

```python
def _register_handlers(mediator: Mediator, db: Database) -> None:
    """Register all module command/query handlers with the mediator."""

    def create_note_handler() -> CreateNoteHandler:
        session = db.session_factory()
        repo = SqlAlchemyNoteRepository(session)
        return CreateNoteHandler(repository=repo)

    def get_note_handler() -> GetNoteHandler:
        session = db.session_factory()
        repo = SqlAlchemyNoteRepository(session)
        return GetNoteHandler(repository=repo)

    mediator.register_command_handler(CreateNoteCommand, create_note_handler)
    mediator.register_query_handler(GetNoteQuery, get_note_handler)
```

Update the lifespan call:

```python
_register_handlers(mediator, db)
```

After `app.add_exception_handler(Exception, general_exception_handler)`, add:

```python
app.include_router(note_router, prefix="/api/v1")
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/app/modules/example/interfaces/ src/app/interfaces/main.py
git commit -m "feat: add example module API layer, wire up router and handlers in main"
```

---

## Task 19: Remove .placeholder Module

**Files:**
- Delete: `src/app/modules/.placeholder/` (entire directory)

**Step 1: Remove**

```bash
git rm -r src/app/modules/.placeholder/
git commit -m "chore: remove .placeholder module, replaced by example module"
```

---

## Task 20: Docker & Deployment — 修正

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `migrations/env.py`

**Step 1: Replace Dockerfile**

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.interfaces.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Replace docker-compose.yml**

```yaml
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:password@db:5432/myapp
      - APP_ENV=development
      - APP_DEBUG=true
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
```

Changes: removed deprecated `version:`, ecommerce → myapp, added DB healthcheck, postgres 15 → 16, volume only mounts `src/`, no `--reload` in production CMD.

**Step 3: Update migrations/env.py**

Replace `src/app/shared_kernel/infrastructure/database import Base` with the new location:

Replace `migrations/env.py` with:

```python
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.app.config import settings
from src.app.shared_kernel.infrastructure.database import Base

# Import all models so Alembic can detect them
from src.app.modules.example.infrastructure.models.note_model import NoteModel  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml migrations/env.py
git commit -m "fix: multi-stage Docker, DB healthcheck, fix alembic model imports"
```

---

## Task 21: Logging 微调

**Files:**
- Modify: `src/app/shared_kernel/infrastructure/logging.py`

**Step 1: Update to use LOG_LEVEL from config**

Replace `src/app/shared_kernel/infrastructure/logging.py` with:

```python
import logging
import sys
from typing import Any

import structlog

from src.app.config import settings


def configure_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if settings.APP_ENV == "development":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/app/shared_kernel/infrastructure/logging.py
git commit -m "refactor: logging uses LOG_LEVEL from config"
```

---

## Task 22: Test Infrastructure — conftest 增强

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Replace conftest.py**

Replace `tests/conftest.py` with:

```python
import pytest
from unittest.mock import AsyncMock

from app.shared_kernel.application.unit_of_work import UnitOfWork


@pytest.fixture
def mock_uow() -> AsyncMock:
    uow = AsyncMock(spec=UnitOfWork)
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    return uow
```

Also ensure these `__init__.py` files exist (create if missing):
- `tests/unit/__init__.py`
- `tests/unit/shared_kernel/__init__.py`
- `tests/unit/modules/__init__.py`
- `tests/unit/modules/example/__init__.py`
- `tests/integration/__init__.py`

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/
git commit -m "refactor: enhance test conftest with mock_uow, add missing __init__.py"
```

---

## Task 23: Engineering Tooling — Makefile

**Files:**
- Create: `Makefile`

**Step 1: Write Makefile**

```makefile
.PHONY: help dev test lint format type-check migrate docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Run development server
	uvicorn app.interfaces.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run tests
	python -m pytest tests/ -v

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run type checker
	mypy src/

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create msg="description")
	alembic revision --autogenerate -m "$(msg)"

docker-up: ## Start Docker services
	docker compose up -d --build

docker-down: ## Stop Docker services
	docker compose down

docker-logs: ## Tail Docker logs
	docker compose logs -f
```

**Step 2: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile with common dev commands"
```

---

## Task 24: Engineering Tooling — pre-commit

**Files:**
- Create: `.pre-commit-config.yaml`

**Step 1: Write config**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic
          - sqlalchemy[mypy]
        args: [--config-file=pyproject.toml]
```

**Step 2: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit config (ruff + mypy)"
```

---

## Task 25: Engineering Tooling — GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write CI config**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/ -v
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "chore: add GitHub Actions CI (lint, type-check, test)"
```

---

## Task 26: README.md

**Files:**
- Create: `README.md`

**Step 1: Write README**

```markdown
# DDD + FastAPI Project Skeleton

A production-ready project skeleton based on **Domain-Driven Design (DDD)** and **Clean Architecture** principles.

## Tech Stack

- **Web Framework:** FastAPI
- **Database:** PostgreSQL + SQLAlchemy (async)
- **Migrations:** Alembic
- **Config:** pydantic-settings
- **Logging:** structlog
- **Testing:** pytest
- **Linting:** ruff + mypy

## Architecture

```
interfaces → application → domain ← infrastructure
```

| Layer | Responsibility |
|-------|---------------|
| **Domain** | Entities, Value Objects, Aggregate Roots, Repository interfaces, Domain Events |
| **Application** | Commands, Queries, Handlers, Unit of Work, Mediator |
| **Infrastructure** | SQLAlchemy models, Repository implementations, Database config |
| **Interfaces** | FastAPI routes, middleware, exception handling |

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd <project>
pip install -e ".[dev]"

# Set up environment
cp .env.example .env

# Start services
make docker-up

# Run migrations
make migrate

# Start dev server
make dev
```

## Project Structure

```
src/app/
├── shared_kernel/      # Shared base classes
│   ├── domain/         # Entity, ValueObject, AggregateRoot, Repository, DomainEvent
│   ├── application/    # Command, Query, Handlers, UnitOfWork, Mediator
│   └── infrastructure/ # Database, SqlAlchemy implementations
├── modules/            # Business modules (one per bounded context)
│   └── example/        # Example module demonstrating full CQRS flow
└── interfaces/         # FastAPI app entry point
```

## Development

```bash
make test           # Run tests
make lint           # Run linter
make format         # Format code
make type-check     # Run mypy
make migrate-create msg="add users table"  # Create migration
```

## Adding a New Module

1. Create `src/app/modules/<name>/domain/` — entities, events, repository interface
2. Create `src/app/modules/<name>/application/` — commands, queries, handlers
3. Create `src/app/modules/<name>/infrastructure/` — models, repository implementation
4. Create `src/app/modules/<name>/interfaces/api/` — router, requests, responses
5. Register handlers in `src/app/interfaces/main.py:_register_handlers()`
6. Include router in `main.py`
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with architecture overview and quick start guide"
```

---

## Task 27: Cleanup — Remove messaging/ and update __init__.py exports

**Files:**
- Delete: `src/app/shared_kernel/infrastructure/messaging/` (empty, not used)
- Delete: `tests/fixtures/__init__.py` (empty directory)

**Step 1: Clean up**

```bash
git rm -r src/app/shared_kernel/infrastructure/messaging/
git rm tests/fixtures/__init__.py
rmdir tests/fixtures 2>/dev/null || true
```

**Step 2: Commit**

```bash
git commit -m "chore: remove empty messaging/ and fixtures/ directories"
```

---

## Task 28: Final Verification

**Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS (approximately 20+ tests)

**Step 2: Run linter**

```bash
ruff check src/ tests/
```

Expected: No errors (or only pre-existing style issues)

**Step 3: Verify import chain**

```bash
python -c "from app.interfaces.main import app; print('Import OK')"
```

Expected: `Import OK`

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: final cleanup after full refactor verification"
```

---

## Summary

| Phase | Tasks | Commits |
|-------|-------|---------|
| Domain Layer | 1-5 | 5 |
| Application Layer | 6-8 | 3 |
| Infrastructure Layer | 9-11 | 3 |
| Config & Dependencies | 12 | 1 |
| Interfaces Layer | 13-14 | 2 |
| Example Module | 15-19 | 5 |
| Docker & Deployment | 20-21 | 2 |
| Test Infrastructure | 22 | 1 |
| Engineering Tooling | 23-26 | 4 |
| Cleanup & Verification | 27-28 | 2 |
| **Total** | **28** | **28** |
