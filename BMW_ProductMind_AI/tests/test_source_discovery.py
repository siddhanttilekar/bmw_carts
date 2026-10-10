"""Content-based discovery / canonical selection, using copies of the real files."""
from __future__ import annotations

import shutil

from core import config
from services.data_ingestion import file_sha256
from services.source_discovery import identify_file, normalized_stem, sync


def test_normalized_stem_strips_copy_suffixes():
    assert normalized_stem("BMW_G60_Public_Competitor_Evidence(1).xlsx") == "BMW_G60_Public_Competitor_Evidence.xlsx"
    assert normalized_stem("sales_volumes (2).xlsx") == "sales_volumes.xlsx"
    assert normalized_stem("G60_feedback_hackathon.xlsx") == "G60_feedback_hackathon.xlsx"


def test_files_identified_by_content_not_name(tmp_path, data_dir):
    # A misleading name must not change the identification.
    src = next(data_dir.rglob(config.FEEDBACK_FILES["G60"]))
    disguised = tmp_path / "sales_volumes(1).xlsx"
    shutil.copy2(src, disguised)
    ident = identify_file(disguised)
    assert ident.dataset_type == "bmw_feedback" and ident.canonical_name == config.FEEDBACK_FILES["G60"]
    consolidated = identify_file(next(data_dir.rglob(config.CONSOLIDATED_FILE)))
    assert consolidated.dataset_type == "consolidated_workbook" and consolidated.imported is False


def test_sync_selects_one_canonical_copy_and_never_touches_originals(tmp_path, data_dir):
    source = tmp_path / "source"
    source.mkdir()
    comp = next(data_dir.rglob(config.COMPETITOR_EVIDENCE_FILE))
    shutil.copy2(comp, source / "BMW_G60_Public_Competitor_Evidence(1).xlsx")
    shutil.copy2(comp, source / "BMW_G60_Public_Competitor_Evidence(2).xlsx")
    shutil.copy2(next(data_dir.rglob(config.SALES_FILE)), source / "sales_volumes.xlsx")
    before = {p.name: file_sha256(p) for p in source.iterdir()}

    target = tmp_path / "data"
    entries = {e.canonical_name: e for e in sync(source, target)}
    competitor = entries[config.COMPETITOR_EVIDENCE_FILE]
    assert competitor.action == "copied" and len(competitor.excluded_duplicates) == 1
    assert (target / "external" / config.COMPETITOR_EVIDENCE_FILE).exists()
    assert (target / "internal" / config.SALES_FILE).exists()
    assert {p.name: file_sha256(p) for p in source.iterdir()} == before  # originals untouched

    again = {e.canonical_name: e.action for e in sync(source, target)}
    assert set(again.values()) == {"already_present"}


def test_sync_does_not_overwrite_a_different_destination(tmp_path, data_dir):
    source, target = tmp_path / "source", tmp_path / "data" / "internal"
    source.mkdir()
    target.mkdir(parents=True)
    shutil.copy2(next(data_dir.rglob(config.SALES_FILE)), source / "sales_volumes.xlsx")
    existing = target / config.SALES_FILE
    shutil.copy2(next(data_dir.rglob(config.CUSTOMER_STUDIES_FILE)), existing)  # different content
    digest = file_sha256(existing)
    entry = next(e for e in sync(source, tmp_path / "data") if e.canonical_name == config.SALES_FILE)
    assert entry.action == "conflict_not_overwritten" and file_sha256(existing) == digest
