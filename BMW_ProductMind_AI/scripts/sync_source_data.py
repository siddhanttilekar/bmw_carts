"""Discover dataset files in a source folder, select canonical copies and copy
them into data/ (never moving or deleting originals).

Usage (from the project root):
    python -m scripts.sync_source_data mydata            # copy + update manifest
    python -m scripts.sync_source_data mydata --dry-run  # report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings  # noqa: E402
from services.source_discovery import sync, write_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="folder containing the supplied files")
    parser.add_argument("--data", type=Path, default=get_settings().data_dir)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source = args.source if args.source.is_absolute() else (Path.cwd() / args.source)
    if not source.is_dir():
        print(f"Source folder not found: {source}")
        return 2
    entries = sync(source.resolve(), args.data.resolve(), apply=not args.dry_run)
    for e in entries:
        dup = f"  (excluded duplicates: {len(e.excluded_duplicates)})" if e.excluded_duplicates else ""
        print(f"{e.action:<26} {e.dataset_type:<27} {e.canonical_name:<45} <- {Path(e.selected_source).name}{dup}")
        if e.note and e.action != "already_present" and e.action != "copied":
            print(f"{'':<26} note: {e.note}")
    if not args.dry_run:
        print("Manifest:", write_manifest(entries, args.data.resolve(), source.resolve()))
    return 1 if any(e.action == "conflict_not_overwritten" for e in entries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
