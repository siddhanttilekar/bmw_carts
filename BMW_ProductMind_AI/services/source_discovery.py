"""Identify supplied dataset files by their *content*, select one canonical
copy per dataset, and copy it into ``data/`` under its normalised name.

File names are not trusted on their own ("x (1).xlsx" may be a duplicate or a
different file): each file is fingerprinted (SHA-256) and classified from its
worksheet names / header columns or PDF metadata. Copies with identical
content collapse to one canonical file. Originals are never moved or deleted.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pymupdf

from core import config
from services.data_ingestion import file_sha256

MANIFEST_NAME = "source_manifest.json"
DUPLICATE_SUFFIX_RE = re.compile(r"(\s*\(\d+\)|\s*-\s*copy|\s+copy)$", re.IGNORECASE)


@dataclass
class Identification:
    dataset_type: str  # bmw_feedback, customer_studies, ... or "unknown"
    canonical_name: str | None  # normalised destination file name
    subfolder: str | None  # internal | external | briefing
    imported: bool  # False for the consolidated workbook (overlap check only)
    evidence: str  # what in the content identified it


@dataclass
class DiscoveredFile:
    path: str
    file_name: str
    sha256: str
    size_bytes: int
    identification: Identification


@dataclass
class SyncEntry:
    canonical_name: str
    dataset_type: str
    destination: str
    sha256: str
    action: str  # copied | already_present | conflict_not_overwritten | unknown_not_copied
    imported: bool
    selected_source: str
    original_file_names: list[str] = field(default_factory=list)
    original_source_paths: list[str] = field(default_factory=list)
    excluded_duplicates: list[str] = field(default_factory=list)
    note: str | None = None


def normalized_stem(name: str) -> str:
    path = Path(name)
    return DUPLICATE_SUFFIX_RE.sub("", path.stem).strip() + path.suffix


def _sheet_headers(path: Path) -> dict[str, list[str]]:
    headers = {}
    with pd.ExcelFile(path) as workbook:
        for sheet in workbook.sheet_names:
            head = pd.read_excel(workbook, sheet_name=sheet, nrows=1, header=0)
            headers[sheet] = [str(c) for c in head.columns]
    return headers


def identify_file(path: Path) -> Identification:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        sheets = _sheet_headers(path)
        names = set(sheets)
        if "START HERE" in names and "G60 Feedback All" in names:
            return Identification("consolidated_workbook", config.CONSOLIDATED_FILE, "internal", False,
                                  "sheets 'START HERE' + 'G60 Feedback All'")
        if config.FEEDBACK_SHEET in names and set(config.FEEDBACK_COLUMNS) <= set(sheets[config.FEEDBACK_SHEET]):
            codes = pd.read_excel(path, sheet_name=config.FEEDBACK_SHEET, usecols=["Derivate (e-code)"])
            codes = sorted(codes["Derivate (e-code)"].dropna().astype(str).unique())
            if len(codes) == 1 and codes[0] in config.FEEDBACK_FILES:
                return Identification("bmw_feedback", config.FEEDBACK_FILES[codes[0]], "internal", True,
                                      f"sheet {config.FEEDBACK_SHEET}, e-code {codes[0]}")
            return Identification("unknown", None, None, False, f"feedback layout with e-codes {codes}")
        if {"CN_EU_2025", "US_2025"} <= names:
            return Identification("customer_studies", config.CUSTOMER_STUDIES_FILE, "internal", True,
                                  "sheets CN_EU_2025 + US_2025")
        if set(config.SALES_SHEET_VEHICLES) <= names and all(
            {"market", "market_code"} <= set(sheets[s]) for s in config.SALES_SHEET_VEHICLES
        ):
            return Identification("sales_volumes", config.SALES_FILE, "internal", True,
                                  "sheets F70/G60_G68/G70 with market + market_code")
        if "Public G60 Feedback" in names:
            return Identification("public_owner_feedback", config.PUBLIC_FEEDBACK_FILE, "external", True,
                                  "sheet 'Public G60 Feedback'")
        if "Public Evidence" in names:
            return Identification("public_competitor_evidence", config.COMPETITOR_EVIDENCE_FILE, "external", True,
                                  "sheet 'Public Evidence'")
        return Identification("unknown", None, None, False, f"unrecognised sheets {sorted(names)}")
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            title = (pdf.metadata or {}).get("title") or ""
            first_page = pdf[0].get_text() if pdf.page_count else ""
        catalogue_titles = {"F70": "BMW 1er", "G60": "BMW 5er", "G70": "BMW 7er"}
        for code, name in config.OPTION_LIST_FILES.items():
            if "Katalog" in title and title.startswith(catalogue_titles[code]):
                return Identification("option_list", name, "internal", True, f"PDF title '{title}'")
        if normalized_stem(path.name) == config.BRIEFING_PDF_FILE:
            return Identification("briefing", config.BRIEFING_PDF_FILE, "briefing", True, "file name (image-only PDF)")
        return Identification("unknown", None, None, False, f"PDF title '{title}', first text {first_page[:40]!r}")
    if suffix == ".pptx" and normalized_stem(path.name) == config.KEYNOTE_PPTX_FILE:
        return Identification("briefing", config.KEYNOTE_PPTX_FILE, "briefing", True, "file name")
    return Identification("unknown", None, None, False, f"unsupported type {suffix}")


def discover(source_dir: Path) -> list[DiscoveredFile]:
    found = []
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and not p.name.startswith((".", "~$"))):
        found.append(
            DiscoveredFile(str(path), path.name, file_sha256(path), path.stat().st_size, identify_file(path))
        )
    return found


def sync(source_dir: Path, data_dir: Path, apply: bool = True) -> list[SyncEntry]:
    """Select one canonical copy per dataset and copy it to data/<subfolder>/.

    * identical content (same SHA-256) -> one canonical file, the rest excluded
    * destination already identical -> left untouched ("already_present")
    * destination differs -> NOT overwritten; reported as a conflict
    """
    discovered = discover(source_dir)
    by_name: dict[str, list[DiscoveredFile]] = {}
    entries: list[SyncEntry] = []
    for item in discovered:
        ident = item.identification
        if ident.canonical_name is None:
            entries.append(SyncEntry(item.file_name, ident.dataset_type, "", item.sha256, "unknown_not_copied", False,
                                     item.path, [item.file_name], [item.path], note=ident.evidence))
            continue
        by_name.setdefault(ident.canonical_name, []).append(item)

    for canonical, items in sorted(by_name.items()):
        hashes = {i.sha256 for i in items}
        ident = items[0].identification
        if len(hashes) > 1:
            entries.append(SyncEntry(canonical, ident.dataset_type, "", "", "conflict_not_overwritten", ident.imported,
                                     "", [i.file_name for i in items], [i.path for i in items],
                                     note="Copies with the same identity but different content; resolve manually."))
            continue
        # Prefer the copy whose name already equals the canonical name.
        items.sort(key=lambda i: (i.file_name != canonical, len(i.file_name), i.path))
        chosen = items[0]
        dest = data_dir / ident.subfolder / canonical
        if dest.exists() and file_sha256(dest) == chosen.sha256:
            action = "already_present"
        elif dest.exists():
            action = "conflict_not_overwritten"
        else:
            action = "copied"
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chosen.path, dest)
        entries.append(
            SyncEntry(canonical, ident.dataset_type, str(dest.relative_to(data_dir)), chosen.sha256, action, ident.imported,
                      chosen.path, sorted({i.file_name for i in items}), [i.path for i in items],
                      [i.path for i in items[1:]], note=ident.evidence)
        )
    return entries


def write_manifest(entries: list[SyncEntry], data_dir: Path, source_dir: Path) -> Path:
    """Merge entries into data/source_manifest.json (keyed by canonical name)."""
    path = data_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text()) if path.exists() else {"datasets": {}}
    manifest["last_sync_source"] = str(source_dir)
    for entry in entries:
        if entry.action == "unknown_not_copied":
            manifest.setdefault("unrecognised", {})[entry.selected_source] = asdict(entry)
            continue
        previous = manifest["datasets"].get(entry.canonical_name, {})
        record = asdict(entry)
        for key in ("original_file_names", "original_source_paths"):
            record[key] = sorted(set(previous.get(key, [])) | set(record[key]))
        manifest["datasets"][entry.canonical_name] = record
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def load_manifest(data_dir: Path) -> dict:
    path = data_dir / MANIFEST_NAME
    return json.loads(path.read_text()).get("datasets", {}) if path.exists() else {}
