"""Build the SQLite database from the supplied datasets and write the
data-quality report.

Usage (from the project root):
    python -m scripts.initialize_database
    python -m scripts.initialize_database --db database/other.db --data data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings  # noqa: E402
from core.database import get_engine, reset_schema, session_scope  # noqa: E402
from services.data_ingestion import run_full_ingestion  # noqa: E402
from services.data_validation import (  # noqa: E402
    build_quality_report,
    summarize_for_console,
    validate_counts,
    write_reports,
)


def initialize(database_path: Path, data_dir: Path, reports_dir: Path | None) -> dict:
    engine = get_engine(database_path)
    reset_schema(engine)
    with session_scope(engine) as session:
        ingestion_report = run_full_ingestion(session, data_dir)
        count_results = validate_counts(session, data_dir)
        report = build_quality_report(session, data_dir, ingestion_report, count_results)
    if reports_dir is not None:
        report["report_files"] = [str(p) for p in write_reports(report, reports_dir)]
    engine.dispose()
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=settings.database_path)
    parser.add_argument("--data", type=Path, default=settings.data_dir)
    parser.add_argument("--reports", type=Path, default=settings.reports_dir)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = initialize(args.db.resolve(), args.data.resolve(), args.reports.resolve())

    print(json.dumps(summarize_for_console(report), indent=2))
    for name, info in report["ingestion"]["files"].items():
        print(f"  {info['status']:<16} {name}" + (f"  -> {info['error']}" if "error" in info else ""))
    print("Reports:", *report.get("report_files", []), sep="\n  ")
    mismatches = [r for r in report["count_validation"] if r["status"] != "ok"]
    failed = [n for n, i in report["ingestion"]["files"].items() if i["status"] in ("failed", "missing")]
    failed_checks = [c["check"] for c in report["integrity_checks"] if not c["passed"]]
    for check in failed_checks:
        print("  FAILED CHECK:", check)
    return 1 if mismatches or failed or failed_checks else 0


if __name__ == "__main__":
    raise SystemExit(main())
