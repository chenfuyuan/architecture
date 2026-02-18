# DDD FastAPI 项目骨架实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 搭建基于 DDD 思想的 FastAPI 项目骨架，包含共享内核、日志系统、数据库迁移、统一响应、异常处理和配置管理。

**Architecture:** 采用模块化整洁架构，按子领域组织代码，依赖方向为：接口层 → 应用层 → 领域层 ← 基础设施层。

**Tech Stack:** FastAPI, SQLAlchemy (异步), PostgreSQL, Alembic, pydantic-settings, pytest, conda, Docker Compose

---

## 任务列表

### Task 1: 初始化项目基础配置

**Files:**
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md`

**Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ecommerce"
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
  "pydantic-settings>=2.1.0",
  "psycopg[binary,pool]>=3.1.18",
  "structlog>=24.1.0",
  "python-dotenv>=1.0.0"
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4.0",
  "pytest-asyncio>=0.23.0",
  "httpx>=0.26.0",
  "faker>=22.0.0",
  "ruff>=0.1.0",
  "mypy>=1.8.0"
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Step 2: 创建 environment.yml**

```yaml
name: ecommerce
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - -e ".[dev]"
```

**Step 3: 创建 .env.example**

```env
# Application
APP_NAME=ecommerce
APP_ENV=development
APP_DEBUG=true
APP_HOST=0.0.0.0
APP_PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ecommerce
```

**Step 4: 创建 .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

.env
.venv
env/
venv/
ENV/

.pytest_cache/
.coverage
htmlcov/

.mypy_cache/
.ruff_cache/

*.swp
*.swo
*~
.DS_Store
```

**Step 5: 创建 README.md**

```markdown
# E-commerce Project

基于 DDD 思想的 FastAPI 项目骨架。

## 技术栈

- FastAPI
- SQLAlchemy (异步)
- PostgreSQL
- Alembic
- conda
- Docker Compose

## 快速开始

### 使用 conda

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate ecommerce

# 复制环境变量
cp .env.example .env

# 启动开发服务器
uvicorn ecommerce.interfaces.main:app --reload
```

### 使用 Docker Compose

```bash
docker-compose up --build
```
```

---

### Task 2: 创建项目目录结构

**Files:**
- Create: `src/ecommerce/__init__.py`
- Create: `src/ecommerce/config.py`
- Create: `src/ecommerce/shared_kernel/__init__.py`
- Create: `src/ecommerce/shared_kernel/domain/__init__.py`
- Create: `src/ecommerce/shared_kernel/application/__init__.py`
- Create: `src/ecommerce/shared_kernel/infrastructure/__init__.py`
- Create: `src/ecommerce/modules/__init__.py`
- Create: `src/ecommerce/interfaces/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/fixtures/__init__.py`
- Create: `migrations/__init__.py`

**Step 1: 创建所有空目录和 __init__.py 文件**

运行以下命令创建目录结构：

```bash
# 共享内核目录
mkdir -p src/ecommerce/shared_kernel/{domain,application,infrastructure}

# 业务模块目录（含占位结构）
mkdir -p src/ecommerce/modules/.placeholder/{domain,application/{commands,queries},infrastructure/models,interfaces/{requests,responses}}

# 接口层、定时任务、消息事件目录
mkdir -p src/ecommerce/interfaces
mkdir -p src/ecommerce/jobs
mkdir -p src/ecommerce/events/{handlers,publishers}

# 测试、迁移目录
mkdir -p tests/{unit,integration,fixtures}
mkdir -p migrations/versions
```

然后创建所有 `__init__.py` 文件（内容为空）。

**Step 2: 创建 src/ecommerce/config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    APP_NAME: str = "ecommerce"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/ecommerce"


settings = Settings()
```

**Step 3: 创建 tests/conftest.py**

```python
import pytest
import asyncio
from typing import AsyncGenerator, Generator


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

---

### Task 3: 实现共享内核 - 领域层基类

**Files:**
- Create: `src/ecommerce/shared_kernel/domain/entity.py`
- Create: `src/ecommerce/shared_kernel/domain/value_object.py`
- Create: `src/ecommerce/shared_kernel/domain/aggregate_root.py`
- Create: `src/ecommerce/shared_kernel/domain/repository.py`
- Create: `src/ecommerce/shared_kernel/domain/exception.py`
- Test: `tests/unit/shared_kernel/domain/test_entity.py`
- Test: `tests/unit/shared_kernel/domain/test_value_object.py`

**Step 1: 创建 entity.py**

```python
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
```

**Step 2: 创建 value_object.py**

```python
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class ValueObject(ABC):
    pass
```

**Step 3: 创建 aggregate_root.py**

```python
from abc import ABC
from .entity import Entity, Id


class AggregateRoot(Entity[Id], ABC):
    pass
```

**Step 4: 创建 repository.py**

```python
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
```

**Step 5: 创建 exception.py**

```python
class DomainException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NotFoundException(DomainException):
    pass


class ValidationException(DomainException):
    pass
```

---

### Task 4: 实现共享内核 - 应用层基类

**Files:**
- Create: `src/ecommerce/shared_kernel/application/command.py`
- Create: `src/ecommerce/shared_kernel/application/query.py`
- Create: `src/ecommerce/shared_kernel/application/command_handler.py`
- Create: `src/ecommerce/shared_kernel/application/query_handler.py`

**Step 1: 创建 command.py**

```python
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class Command(ABC):
    pass
```

**Step 2: 创建 query.py**

```python
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class Query(ABC):
    pass
```

**Step 3: 创建 command_handler.py**

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .command import Command

Result = TypeVar("Result")


class CommandHandler(ABC, Generic[Command, Result]):
    @abstractmethod
    async def handle(self, command: Command) -> Result:
        pass
```

**Step 4: 创建 query_handler.py**

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .query import Query

Result = TypeVar("Result")


class QueryHandler(ABC, Generic[Query, Result]):
    @abstractmethod
    async def handle(self, query: Query) -> Result:
        pass
```

---

### Task 5: 实现共享内核 - 基础设施层

**Files:**
- Create: `src/ecommerce/shared_kernel/infrastructure/logging.py`
- Create: `src/ecommerce/shared_kernel/infrastructure/database.py`
- Create: `src/ecommerce/shared_kernel/infrastructure/sqlalchemy_repository.py`

**Step 1: 创建 logging.py**

```python
import logging
import sys
from typing import Any

import structlog
from src.ecommerce.config import settings


def configure_logging() -> None:
    if settings.APP_ENV == "development":
        level = logging.DEBUG
        renderer = structlog.dev.ConsoleRenderer()
    else:
        level = logging.INFO
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

**Step 2: 创建 database.py**

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from src.ecommerce.config import settings

Base = declarative_base()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,
    future=True,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
```

**Step 3: 创建 sqlalchemy_repository.py**

```python
from typing import List, Optional, TypeVar
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.ecommerce.shared_kernel.domain.aggregate_root import AggregateRoot
from src.ecommerce.shared_kernel.domain.entity import Id
from src.ecommerce.shared_kernel.domain.repository import Repository

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
```

---

### Task 6: 实现全局接口层 - 统一响应和异常处理

**Files:**
- Create: `src/ecommerce/interfaces/response.py`
- Create: `src/ecommerce/interfaces/exception_handler.py`
- Create: `src/ecommerce/interfaces/main.py`
- Create: `src/ecommerce/interfaces/dependencies.py`
- Create: `src/ecommerce/interfaces/middleware.py`

**Step 1: 创建 response.py**

```python
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    code: int = Field(default=200, description="响应状态码")
    message: str = Field(default="success", description="响应消息")
    data: Optional[DataT] = Field(default=None, description="响应数据")

    @classmethod
    def success(cls, data: Optional[DataT] = None, message: str = "success") -> "ApiResponse[DataT]":
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 500, message: str = "error") -> "ApiResponse[DataT]":
        return cls(code=code, message=message, data=None)
```

**Step 2: 创建 exception_handler.py**

```python
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.ecommerce.interfaces.response import ApiResponse
from src.ecommerce.shared_kernel.domain.exception import DomainException, NotFoundException, ValidationException


async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
    if isinstance(exc, NotFoundException):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, ValidationException):
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        status_code = status.HTTP_400_BAD_REQUEST

    response = ApiResponse.error(code=status_code, message=exc.message)
    return JSONResponse(content=response.model_dump(), status_code=status_code)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    error_messages = []
    for err in errors:
        loc = " -> ".join([str(x) for x in err["loc"] if x != "body"])
        error_messages.append(f"{loc}: {err['msg']}")

    message = "; ".join(error_messages) if error_messages else "Validation error"
    response = ApiResponse.error(code=status.HTTP_422_UNPROCESSABLE_ENTITY, message=message)
    return JSONResponse(content=response.model_dump(), status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    response = ApiResponse.error(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Internal server error")
    return JSONResponse(content=response.model_dump(), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

**Step 3: 创建 middleware.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

**Step 4: 创建 dependencies.py**

```python
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.ecommerce.shared_kernel.infrastructure.database import get_session
from src.ecommerce.shared_kernel.infrastructure.logging import get_logger

logger = get_logger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session
```

**Step 5: 创建 main.py**

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from src.ecommerce.config import settings
from src.ecommerce.interfaces.dependencies import get_logger
from src.ecommerce.interfaces.exception_handler import (
    domain_exception_handler,
    general_exception_handler,
    validation_exception_handler,
)
from src.ecommerce.interfaces.middleware import setup_middleware
from src.ecommerce.interfaces.response import ApiResponse
from src.ecommerce.shared_kernel.domain.exception import DomainException
from src.ecommerce.shared_kernel.infrastructure.logging import configure_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("Application starting up", app_name=settings.APP_NAME, env=settings.APP_ENV)
    yield
    logger.info("Application shutting down")


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

---

### Task 7: 配置 Alembic 数据库迁移

**Files:**
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`

**Step 1: 创建 alembic.ini**

```ini
[alembic]
script_location = migrations
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s
sqlalchemy.url = postgresql+asyncpg://user:password@localhost:5432/ecommerce

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

**Step 2: 创建 migrations/env.py**

```python
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
from src.ecommerce.config import settings
from src.ecommerce.shared_kernel.infrastructure.database import Base

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

**Step 3: 创建 migrations/script.py.mako**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

---

### Task 8: 创建 Docker 和 Docker Compose 配置

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

**Step 1: 创建 Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "ecommerce.interfaces.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**Step 2: 创建 docker-compose.yml**

```yaml
version: "3.8"

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./:/app
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:password@db:5432/ecommerce
      - APP_ENV=development
      - APP_DEBUG=true
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=ecommerce
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

**Step 3: 创建 .dockerignore**

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.env
.venv/
venv/
.git/
.gitignore
*.md
!README.md
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/
.DS_Store
```

---

### Task 9: 验证项目是否正常运行

**Step 1: 创建 conda 环境**
```bash
conda env create -f environment.yml
conda activate ecommerce
```

**Step 2: 复制环境变量**
```bash
cp .env.example .env
```

**Step 3: 启动数据库（可选，使用 Docker）**
```bash
docker-compose up -d db
```

**Step 4: 运行 FastAPI 应用**
```bash
uvicorn ecommerce.interfaces.main:app --reload
```

**Step 5: 访问健康检查接口**
访问 `http://localhost:8000/health`，应该返回统一格式的健康检查响应。

---

## 执行说明

> **Plan complete and saved to `docs/plans/2026-02-19-ddd-fastapi-skeleton-implementation.md`. Two execution options:**
>
> **1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration
>
> **2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints
>
> **Which approach?**
