# DDD FastAPI Project Skeleton

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
conda activate app

# 复制环境变量
cp .env.example .env

# 启动开发服务器
uvicorn app.interfaces.main:app --reload
```

### 使用 Docker Compose

```bash
docker-compose up --build
```
