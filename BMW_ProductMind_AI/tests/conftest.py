"""Shared fixtures: build a database from the real supplied datasets once per session."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings  # noqa: E402
from core.database import get_engine, session_scope  # noqa: E402
from scripts.initialize_database import initialize  # noqa: E402


@pytest.fixture(scope="session")
def data_dir() -> Path:
    path = get_settings().data_dir
    if not any(path.rglob("*.xlsx")):
        pytest.skip(f"Supplied datasets not found under {path}")
    return path


@pytest.fixture(scope="session")
def built(tmp_path_factory, data_dir):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    report = initialize(db_path, data_dir, tmp_path_factory.mktemp("reports"))
    return {"db_path": db_path, "report": report}


@pytest.fixture()
def session(built):
    engine = get_engine(built["db_path"])
    with session_scope(engine) as s:
        yield s
    engine.dispose()
