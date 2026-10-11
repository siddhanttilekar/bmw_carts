"""SQLite engine/session helpers."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.ai_models import AIBase
from core import decision_models, evidence_models, prioritization_models, requirement_models  # noqa: F401
from core.models import Base


def get_engine(database_path: Path) -> Engine:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{database_path}", future=True)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def reset_schema(engine: Engine) -> None:
    """Drop and recreate the *ingestion* tables (a full, idempotent rebuild).
    AI-layer tables (``AIBase``) are untouched so paid-for results survive."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    ensure_ai_schema(engine)


def ensure_ai_schema(engine: Engine) -> list[str]:
    """Safe, additive migration for AI tables: create missing tables and add
    missing columns (as nullable). Never drops or rewrites existing data.
    Returns the list of changes applied."""
    changes: list[str] = []
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    for table in AIBase.metadata.sorted_tables:
        if table.name not in existing:
            table.create(engine)
            changes.append(f"created table {table.name}")
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            ddl = column.type.compile(engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}'))
            changes.append(f"added column {table.name}.{column.name}")
    return changes


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
