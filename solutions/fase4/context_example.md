# TaskFlow API — Context File

> This file gives any agent (Copilot, Cursor, Claude Code, OpenCode) all the context to work on this project.

## PROJECT

TaskFlow is a task management REST API built with FastAPI. It exposes CRUD for tasks and projects, JWT authentication, and webhooks for integrations. Internal API used by the product team (~15 devs).

## STACK

- **Runtime:** Python 3.12
- **Framework:** FastAPI 0.115+ with Pydantic v2
- **ORM:** SQLAlchemy 2.0 with async (`AsyncSession`, not `Session`)
- **DB:** PostgreSQL 16 (RDS)
- **Migrations:** Alembic
- **Auth:** python-jose (JWT) + passlib (bcrypt)
- **Tests:** pytest + pytest-asyncio + httpx (`AsyncClient`)
- **Deploy:** Docker → AWS ECS Fargate
- **CI:** GitHub Actions (lint → test → build → deploy)

## STRUCTURE

```
taskflow/
├── api/
│   ├── routes/          # One file per resource: tasks.py, projects.py, auth.py
│   ├── dependencies.py  # get_db(), get_current_user()
│   └── main.py          # App factory: create_app()
├── models/              # SQLAlchemy models (1 file per table)
├── schemas/             # Pydantic schemas (request/response per resource)
├── services/            # Business logic (1 per domain). NEVER accesses the request.
├── repositories/        # Database queries. ONLY place that imports AsyncSession.
├── core/
│   ├── config.py        # Settings with pydantic-settings
│   └── security.py      # JWT encode/decode, password hashing
├── migrations/          # Alembic
├── tests/
│   ├── conftest.py      # Fixtures: async_client, db_session, test_user
│   ├── api/             # Integration tests per endpoint
│   └── services/        # Unit tests per service
└── docker-compose.yml
```

## CONVENTIONS

- **Layers:** Route → Service → Repository. Routes NEVER import SQLAlchemy. Services NEVER import Request/Response.
- **Naming:** `snake_case` for everything. Files in singular (`task.py` not `tasks.py` for models).
- **Schemas:** `TaskCreate`, `TaskUpdate`, `TaskResponse` — always with suffix.
- **Async everywhere:** All DB functions are `async def`. Use `await session.execute()`.
- **Errors:** Raise `HTTPException` ONLY in routes. Services throw `ValueError` or custom exceptions.
- **Tests:** Each endpoint has at least: happy path, validation error, auth error, not found.
- **Commits:** Conventional commits (`feat:`, `fix:`, `refactor:`, `test:`).

## COMMANDS

```bash
# Dev
docker compose up -d db          # Postgres only
uvicorn taskflow.api.main:app --reload

# Test
pytest tests/ -v --tb=short
pytest tests/api/test_tasks.py -k "test_create" -v  # A specific test

# Migrations
alembic revision --autogenerate -m "add column X"
alembic upgrade head

# Lint
ruff check . --fix
ruff format .

# Build & Deploy
docker build -t taskflow .
aws ecs update-service --cluster prod --service taskflow --force-new-deployment
```

## DO NOT

1. **Don't use `Session` (sync).** Always `AsyncSession`. If you see `Session` in new code, it's a bug.
2. **Don't put business logic in routes.** Routes are thin: parse request, call service, return response.
3. **Don't import SQLAlchemy in services.** Services work with Pydantic schemas, not ORM models.
4. **Don't use `*` imports.** Every import must be explicit.
5. **Don't hardcode secrets.** Everything goes in `core/config.py` via env vars.
6. **Don't create empty migrations.** Always verify with `alembic check` before generating.
7. **Don't use `datetime.now()`.** Always `datetime.now(UTC)` (timezone-aware).
8. **Don't use print().** Use `logging.getLogger(__name__)`.
9. **Don't skip tests in CI.** If a test is broken, fix it or delete it — don't mark it `@pytest.mark.skip`.
10. **Don't change the DB schema without a migration.** Not even "it's just a default".
