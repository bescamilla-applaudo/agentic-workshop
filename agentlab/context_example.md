# Tasky API — Agent Context File

> Feed this file to any agent (Claude Code, Copilot, Cursor) before asking it to write code.
> It replaces the need to explain the project every session.

---

## PROJECT

Tasky is an internal task management REST API used by ~20 engineers across 3 product squads.
It handles tasks, assignments, labels, and deadline tracking. Integrates with Slack for
notifications and Jira for ticket mirroring. Not public-facing — clients are internal
dashboards and a mobile app.

**Current focus:** migrating from sync SQLAlchemy 1.4 to async SQLAlchemy 2.0.
All new code must use async. Do not touch sync legacy code unless explicitly asked.

---

## STACK

| Layer | Library | Version | Notes |
|---|---|---|---|
| Runtime | Python | 3.12 | `match` statements ok; no 3.11 workarounds needed |
| Framework | FastAPI | 0.115+ | Pydantic v2 models only |
| ORM | SQLAlchemy | 2.0 | `AsyncSession`, `mapped_column`, `Mapped[T]` syntax |
| Migrations | Alembic | latest | autogenerate only; review before applying |
| Database | PostgreSQL | 16 | `asyncpg` driver (`postgresql+asyncpg://`) |
| Auth | python-jose | 3.x | JWT HS256; tokens issued by `core/security.py` |
| Password hashing | passlib | 1.7 | bcrypt scheme |
| HTTP client (tests) | httpx | latest | `AsyncClient` with `ASGITransport` |
| Test runner | pytest + pytest-asyncio | latest | `asyncio_mode = "auto"` in `pyproject.toml` |
| Linter | ruff | 0.8+ | enforced in CI; fix before committing |
| Container | Docker | — | multi-stage build; prod image is `python:3.12-slim` |

---

## STRUCTURE

```
tasky/
├── api/
│   ├── routes/
│   │   ├── tasks.py        # /tasks endpoints
│   │   ├── assignments.py  # /assignments endpoints
│   │   ├── labels.py       # /labels endpoints
│   │   └── auth.py         # /auth/login, /auth/refresh
│   ├── dependencies.py     # get_db() → AsyncSession, get_current_user() → User
│   └── main.py             # create_app() factory, router registration, lifespan
│
├── models/                 # One file per DB table, named singular
│   ├── task.py             # Task model
│   ├── user.py             # User model
│   └── label.py            # Label model
│
├── schemas/                # Pydantic request/response — never import ORM here
│   ├── task.py             # TaskCreate, TaskUpdate, TaskResponse, TaskListResponse
│   └── user.py             # UserCreate, UserResponse, TokenResponse
│
├── services/               # Business logic — one class per domain
│   ├── task_service.py     # TaskService: create, update, assign, close
│   └── notification_service.py  # SlackNotifier, JiraSync
│
├── repositories/           # ALL SQLAlchemy queries live here — nowhere else
│   ├── task_repo.py        # TaskRepository(session: AsyncSession)
│   └── user_repo.py        # UserRepository(session: AsyncSession)
│
├── core/
│   ├── config.py           # Settings (pydantic-settings); source of all env vars
│   ├── database.py         # async_engine, AsyncSessionLocal, get_async_session()
│   └── security.py         # create_access_token(), verify_token(), hash_password()
│
├── tests/
│   ├── conftest.py         # app fixture, async_client, db_session, seed_user
│   ├── api/                # One file per route file (test_tasks.py, test_auth.py …)
│   └── services/           # Unit tests — mock the repository, test the service
│
└── migrations/
    └── versions/           # Never edit by hand; always autogenerate
```

---

## CONVENTIONS

### Layer rules (strict)

```
Route → Service → Repository
```

- **Routes** (`api/routes/*.py`): parse request, call one service method, return schema.
  Never import `AsyncSession`. Never import a model. Never run queries.
- **Services** (`services/*.py`): business logic only. Accept/return Pydantic schemas or
  primitive values. Never import `AsyncSession` directly — receive it via `__init__`.
- **Repositories** (`repositories/*.py`): the only place that touches SQLAlchemy.
  Return ORM model instances, never Pydantic schemas.

### SQLAlchemy 2.0 patterns

Always use `Mapped[T]` and `mapped_column`:

```python
# models/task.py
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, DateTime
from datetime import datetime, UTC

class Task(Base):
    __tablename__ = "tasks"

    id:          Mapped[int]           = mapped_column(primary_key=True)
    title:       Mapped[str]           = mapped_column(String(200), nullable=False)
    completed:   Mapped[bool]          = mapped_column(default=False)
    due_date:    Mapped[datetime|None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at:  Mapped[datetime]      = mapped_column(default=lambda: datetime.now(UTC))
```

Always use `select()` + `scalars()` in repositories:

```python
# repositories/task_repo.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class TaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, task_id: int) -> Task | None:
        result = await self.session.execute(select(Task).where(Task.id == task_id))
        return result.scalar_one_or_none()

    async def list_open(self, limit: int = 50) -> list[Task]:
        result = await self.session.execute(
            select(Task).where(Task.completed == False).limit(limit)  # noqa: E712
        )
        return list(result.scalars().all())
```

### Schema naming

```
{Resource}Create   — POST body
{Resource}Update   — PATCH body (all fields Optional)
{Resource}Response — what the API returns (never exposes password_hash)
{Resource}ListResponse — paginated wrapper: {"items": [...], "total": int}
```

### Error handling

- Services raise `ValueError` (business rule) or `PermissionError` (authz).
- Routes catch those and convert to `HTTPException`:

```python
# api/routes/tasks.py
@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    try:
        return await task_service.get(db, task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
```

### Tests

Every endpoint must have these 4 cases at minimum:

```python
async def test_create_task_happy_path(async_client, seed_user): ...
async def test_create_task_validation_error(async_client, seed_user): ...   # missing required field
async def test_create_task_unauthenticated(async_client): ...               # no token
async def test_create_task_forbidden(async_client, other_user): ...         # wrong owner
```

Use `AsyncClient` with `ASGITransport` — never spin up a real server in tests:

```python
# tests/conftest.py
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from tasky.api.main import create_app

@pytest_asyncio.fixture
async def async_client(db_session):
    async with AsyncClient(
        transport=ASGITransport(app=create_app()),
        base_url="http://test",
    ) as client:
        yield client
```

### Miscellaneous

- All datetimes: `datetime.now(UTC)` — never `datetime.now()` (naive).
- Logging: `logger = logging.getLogger(__name__)` at module level — never `print()`.
- Config: all env vars go through `core/config.py` — never `os.environ["X"]` inline.
- Commits: `feat:`, `fix:`, `refactor:`, `test:`, `chore:` prefixes enforced by CI hook.

---

## COMMANDS

```bash
# ── Local dev ──────────────────────────────────────────────────────────
docker compose up -d postgres          # start Postgres only
alembic upgrade head                   # apply all pending migrations
uvicorn tasky.api.main:app --reload    # dev server on :8000

# ── Tests ──────────────────────────────────────────────────────────────
pytest tests/ -v --tb=short            # full suite
pytest tests/api/test_tasks.py -v      # single file
pytest -k "test_create" -v             # tests matching a name pattern
pytest --cov=tasky --cov-report=term-missing   # with coverage

# ── Migrations ─────────────────────────────────────────────────────────
alembic revision --autogenerate -m "add due_date to tasks"
alembic upgrade head
alembic downgrade -1                   # rollback one revision

# ── Lint / format ──────────────────────────────────────────────────────
ruff check . --fix
ruff format .

# ── Build ──────────────────────────────────────────────────────────────
docker build -t tasky:local .
docker run --env-file .env -p 8000:8000 tasky:local
```

---

## DO NOT

1. **Never use `Session` (sync ORM).** Always `AsyncSession`. A `Session` import in new
   code is a bug, not a style choice.

2. **Never put queries in routes or services.** If you need a DB query, add a method to
   the appropriate `Repository` class and call it from the service.

3. **Never expose ORM model instances from routes.** Routes must return Pydantic
   `*Response` schemas. FastAPI's `response_model=` enforces this — don't bypass it.

4. **Never use `datetime.now()` without timezone.** Use `datetime.now(UTC)`. Naive
   datetimes break the Slack/Jira integrations that expect UTC offsets.

5. **Never hardcode credentials, URLs, or environment-specific values.** Add them to
   `core/config.py` and read from environment. No exceptions for "local-only" values.

6. **Never create a migration file by hand.** Always `alembic revision --autogenerate`,
   then review the generated file before running `upgrade head`.

7. **Never use `@pytest.mark.skip`.** If a test is broken, fix it or delete it with a
   comment explaining why. Skipped tests in CI are treated as failures by the hook.

8. **Never call `session.commit()` inside a repository method.** Commit control belongs
   to the service layer (or the request lifecycle via `get_db()`). Repositories only
   `add()`, `delete()`, and `flush()`.

9. **Never import from `api/` inside `services/` or `repositories/`.** The dependency
   arrow is one-way: `api → services → repositories`. Circular imports here have caused
   two production incidents.

10. **Never use `SELECT *` via `session.execute(text("SELECT * FROM ..."))`.** All
    queries must go through the ORM so that column renames and migrations are caught
    statically by type checkers.
