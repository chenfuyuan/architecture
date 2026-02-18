---
description: 测试编写规则
globs: tests/**/*.py
---

# 测试规则

## 目录

- `tests/unit/` — 领域逻辑，纯 Python，无外部依赖
- `tests/integration/` — 多层协作，使用测试数据库（aiosqlite）

## TDD 流程

1. 写一个明确失败的测试
2. 运行确认失败 (`pytest tests/path/test.py::test_name -v`)
3. 写最小实现让测试通过
4. 运行确认通过
5. Commit

## 命名

- 文件: `test_<被测对象>.py`
- 函数: `test_<行为描述>()`
- 类: `Test<被测类名>`

## 单元测试

- Mock 外部依赖（Repository → `AsyncMock(spec=XxxRepository)`）
- 测试领域行为，不测框架

## 集成测试

- 使用 `aiosqlite` 内存数据库
- 通过 `conftest.py` 中的 fixture 提供 session_factory / UoW

## 反模式

- 不要测试 Python 语言本身的行为
- 不要在测试中硬编码数据库连接字符串
- 不要跳过断言（每个 test 至少一个 assert）
