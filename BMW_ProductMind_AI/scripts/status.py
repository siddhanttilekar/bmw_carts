"""Command-line database status (formerly `python app.py`).

    python -m scripts.status
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.database import get_engine, session_scope  # noqa: E402
from core.models import SourceDocument  # noqa: E402
from services.data_validation import TABLE_MODELS  # noqa: E402


def main() -> None:
    settings = get_settings()
    if not settings.database_path.exists():
        print(f"No database at {settings.database_path}. Run: python -m scripts.initialize_database")
        return
    engine = get_engine(settings.database_path)
    with session_scope(engine) as session:
        print(f"Database: {settings.database_path}")
        for name, model in TABLE_MODELS.items():
            print(f"  {name:<26} {session.scalar(select(func.count()).select_from(model)):>7}")
        print("Source documents:")
        for doc in session.scalars(select(SourceDocument).order_by(SourceDocument.data_origin, SourceDocument.file_name)):
            print(f"  [{doc.data_origin:<15}] {doc.ingestion_status:<16} {doc.file_name}")
    engine.dispose()


if __name__ == "__main__":
    main()
