"""Validation and data-quality reporting.

Expected counts are recomputed directly from the raw files (independently of
the ingestion code paths) and compared against the database. The data-quality
report is generated from the raw-file profiles plus database queries.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
from pptx import Presentation
from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from core import config
from core.models import (
    CompetitorEvidence,
    CustomerStudy,
    DatasetNote,
    DatasetOverlap,
    Equipment,
    EquipmentAvailability,
    Feedback,
    FeedbackClassification,
    IngestionAudit,
    PossibleDuplicate,
    PublicFeedback,
    SalesVolume,
    SourceDocument,
    SourceDocumentPage,
)
from services.data_ingestion import utc_now

TABLE_MODELS = {
    "source_documents": SourceDocument,
    "source_document_pages": SourceDocumentPage,
    "dataset_notes": DatasetNote,
    "feedback": Feedback,
    "feedback_classifications": FeedbackClassification,
    "public_feedback": PublicFeedback,
    "competitor_evidence": CompetitorEvidence,
    "customer_studies": CustomerStudy,
    "sales": SalesVolume,
    "equipment": Equipment,
    "equipment_availability": EquipmentAvailability,
    "dataset_overlaps": DatasetOverlap,
    "possible_duplicates": PossibleDuplicate,
    "ingestion_audit": IngestionAudit,
}


# ---------------------------------------------------------------------------
# Raw-file profiling
# ---------------------------------------------------------------------------


def _infer_type(series: pd.Series) -> str:
    return pd.api.types.infer_dtype(series.dropna(), skipna=True) if series.notna().any() else "empty"


def profile_sheet(path: Path, sheet: str, header: int | None = 0, key_column: str | None = None) -> dict:
    df = pd.read_excel(path, sheet_name=sheet, header=header, dtype=object)
    columns = []
    for col in df.columns:
        s = df[col]
        columns.append(
            {
                "column": str(col),
                "inferred_type": _infer_type(s),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean()) * 100, 2) if len(s) else 0.0,
                "distinct": int(s.nunique(dropna=True)),
            }
        )
    profile = {
        "file": path.name,
        "sheet": sheet,
        "header_row": None if header is None else header + 1,
        "rows": int(len(df)),
        "columns": columns,
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    if key_column and key_column in df.columns:
        counts = df[key_column].value_counts()
        profile["key_column"] = key_column
        profile["distinct_keys"] = int(counts.size)
        profile["keys_on_multiple_rows"] = int((counts > 1).sum())
        profile["max_rows_per_key"] = int(counts.max()) if counts.size else 0
    return profile


SHEET_PROFILE_PLAN: list[tuple[str, str, int | None, str | None]] = [
    *[(name, config.FEEDBACK_SHEET, 0, "ID") for name in config.FEEDBACK_FILES.values()],
    (config.CUSTOMER_STUDIES_FILE, "CN_EU_2025", None, None),
    (config.CUSTOMER_STUDIES_FILE, "US_2025", None, None),
    (config.SALES_FILE, "F70", 0, None),
    (config.SALES_FILE, "G60_G68", 0, None),
    (config.SALES_FILE, "G70", 0, None),
    (config.PUBLIC_FEEDBACK_FILE, "Public G60 Feedback", 0, "ID"),
    (config.PUBLIC_FEEDBACK_FILE, "README", None, None),
    (config.COMPETITOR_EVIDENCE_FILE, "Public Evidence", 0, None),
    (config.COMPETITOR_EVIDENCE_FILE, "Method and caveats", 0, None),
]


def find_file(data_dir: Path, name: str) -> Path:
    matches = list(data_dir.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} under {data_dir}, found {len(matches)}")
    return matches[0]


def profile_all_sheets(data_dir: Path) -> list[dict]:
    profiles = []
    for name, sheet, header, key in SHEET_PROFILE_PLAN:
        profiles.append(profile_sheet(find_file(data_dir, name), sheet, header, key))
    consolidated = find_file(data_dir, config.CONSOLIDATED_FILE)
    with pd.ExcelFile(consolidated) as workbook:
        for sheet in workbook.sheet_names:
            header = None if sheet in ("Study CN EU 2025", "Study US 2025", "START HERE") else 0
            profiles.append(profile_sheet(consolidated, sheet, header, "ID" if sheet == "G60 Feedback All" else None))
    return profiles


# ---------------------------------------------------------------------------
# Count validation (independent recount from raw files)
# ---------------------------------------------------------------------------


def _numeric_cells(df: pd.DataFrame, first_row: int, first_col: int) -> int:
    block = df.iloc[first_row:, first_col:]
    return int(block.map(lambda v: isinstance(v, (int, float)) and not pd.isna(v)).to_numpy().sum())


def _non_empty_rows(df: pd.DataFrame) -> int:
    return int(df.notna().any(axis=1).sum())


def expected_counts(data_dir: Path) -> list[dict]:
    """(file, sheet, table, expected, basis, filter) tuples derived from raw files."""
    out: list[dict] = []

    def add(file, sheet, table, expected, basis, where=None):
        out.append({"file": file, "sheet": sheet, "table": table, "expected": int(expected), "basis": basis, "where": where or {}})

    for code, name in config.FEEDBACK_FILES.items():
        df = pd.read_excel(find_file(data_dir, name), sheet_name=config.FEEDBACK_SHEET, dtype=object)
        add(name, config.FEEDBACK_SHEET, "feedback_classifications", len(df), "worksheet data rows")
        add(name, config.FEEDBACK_SHEET, "feedback", df["ID"].nunique(), "distinct values in ID column")

    studies = find_file(data_dir, config.CUSTOMER_STUDIES_FILE)
    cn = pd.read_excel(studies, sheet_name="CN_EU_2025", header=None, dtype=object)
    add(config.CUSTOMER_STUDIES_FILE, "CN_EU_2025", "customer_studies", _numeric_cells(cn, 3, 1),
        "numeric cells below header rows 1-3, columns B+", {"source_sheet": "CN_EU_2025"})
    us = pd.read_excel(studies, sheet_name="US_2025", header=None, dtype=object)
    add(config.CUSTOMER_STUDIES_FILE, "US_2025", "customer_studies", _numeric_cells(us, 0, 2),
        "numeric cells in columns C+", {"source_sheet": "US_2025"})

    sales = find_file(data_dir, config.SALES_FILE)
    for sheet in config.SALES_SHEET_VEHICLES:
        df = pd.read_excel(sales, sheet_name=sheet)
        year_cols = [c for c in df.columns if str(c).startswith("volume_")]
        add(config.SALES_FILE, sheet, "sales", len(df) * len(year_cols), "data rows x volume_YYYY columns",
            {"source_sheet": sheet})

    for name in config.OPTION_LIST_FILES.values():
        with pymupdf.open(find_file(data_dir, name)) as pdf:
            add(name, None, "source_document_pages", pdf.page_count, "PDF page count")
    with pymupdf.open(find_file(data_dir, config.BRIEFING_PDF_FILE)) as pdf:
        add(config.BRIEFING_PDF_FILE, None, "source_document_pages", pdf.page_count, "PDF page count")
    deck = Presentation(str(find_file(data_dir, config.KEYNOTE_PPTX_FILE)))
    add(config.KEYNOTE_PPTX_FILE, None, "source_document_pages", len(deck.slides), "slide count")

    public = find_file(data_dir, config.PUBLIC_FEEDBACK_FILE)
    add(config.PUBLIC_FEEDBACK_FILE, "Public G60 Feedback", "public_feedback",
        len(pd.read_excel(public, sheet_name="Public G60 Feedback")), "worksheet data rows")
    add(config.PUBLIC_FEEDBACK_FILE, "README", "dataset_notes",
        _non_empty_rows(pd.read_excel(public, sheet_name="README", header=None)), "non-empty rows",
        {"source_sheet": "README"})
    comp = find_file(data_dir, config.COMPETITOR_EVIDENCE_FILE)
    add(config.COMPETITOR_EVIDENCE_FILE, "Public Evidence", "competitor_evidence",
        len(pd.read_excel(comp, sheet_name="Public Evidence")), "worksheet data rows")
    add(config.COMPETITOR_EVIDENCE_FILE, "Method and caveats", "dataset_notes",
        _non_empty_rows(pd.read_excel(comp, sheet_name="Method and caveats", header=None)), "non-empty rows",
        {"source_sheet": "Method and caveats"})
    cons = find_file(data_dir, config.CONSOLIDATED_FILE)
    add(config.CONSOLIDATED_FILE, "START HERE", "dataset_notes",
        _non_empty_rows(pd.read_excel(cons, sheet_name="START HERE", header=None)), "non-empty rows",
        {"source_sheet": "START HERE"})
    with pd.ExcelFile(cons) as workbook:
        add(config.CONSOLIDATED_FILE, None, "dataset_overlaps", len(workbook.sheet_names), "one comparison per sheet")
    return out


def validate_counts(session: Session, data_dir: Path) -> list[dict]:
    checked_at = utc_now()
    results = []
    for exp in expected_counts(data_dir):
        model = TABLE_MODELS[exp["table"]]
        if exp["table"] == "dataset_overlaps":
            stmt = select(func.count()).select_from(model).where(model.consolidated_file == exp["file"])
        else:
            stmt = select(func.count()).select_from(model).where(model.source_file == exp["file"])
            for column, value in exp["where"].items():
                stmt = stmt.where(getattr(model, column) == value)
        imported = session.scalar(stmt) or 0
        status = "ok" if imported == exp["expected"] else "mismatch"
        session.add(
            IngestionAudit(
                source_file=exp["file"],
                source_sheet=exp["sheet"],
                target_table=exp["table"],
                expected_count=exp["expected"],
                imported_count=imported,
                count_basis=exp["basis"],
                status=status,
                checked_at=checked_at,
            )
        )
        results.append({**{k: exp[k] for k in ("file", "sheet", "table", "expected", "basis")}, "imported": imported, "status": status})
    session.flush()
    return results


# ---------------------------------------------------------------------------
# Database-side quality checks
# ---------------------------------------------------------------------------


def table_counts(session: Session) -> dict[str, int]:
    return {name: session.scalar(select(func.count()).select_from(model)) or 0 for name, model in TABLE_MODELS.items()}


def provenance_gaps(session: Session) -> dict[str, int]:
    """Records missing any mandatory provenance field (should all be 0)."""
    gaps = {}
    for name, model in TABLE_MODELS.items():
        if not hasattr(model, "record_id") or not hasattr(model, "source_file"):
            continue
        stmt = select(func.count()).select_from(model).where(
            (model.record_id.is_(None)) | (model.source_file.is_(None)) | (model.imported_at.is_(None))
            | ((model.source_sheet.is_(None)) & (model.source_page.is_(None)))
        )
        gaps[name] = session.scalar(stmt) or 0
    return gaps


def feedback_quality(session: Session) -> list[dict]:
    rows = []
    for code in config.FEEDBACK_FILES:
        fc = FeedbackClassification
        base = select(func.count()).select_from(fc).where(fc.vehicle_code == code)
        status_counts = dict(
            session.execute(
                select(fc.derived_classification_status, func.count()).where(fc.vehicle_code == code)
                .group_by(fc.derived_classification_status)
            ).all()
        )
        multi = session.scalar(
            select(func.count()).select_from(Feedback).where(Feedback.vehicle_code == code, Feedback.classification_row_count > 1)
        )
        l23_mismatch = session.scalar(
            base.where(
                fc.vfc_level2_name.is_not(None), fc.vfc_level3_name.is_not(None),
                fc.vfc_level2_3_name != fc.vfc_level2_name + " : " + fc.vfc_level3_name,
            )
        )
        rows.append(
            {
                "vehicle": code,
                "classification_rows": session.scalar(base),
                "feedback_ids": session.scalar(select(func.count()).select_from(Feedback).where(Feedback.vehicle_code == code)),
                "ids_with_multiple_classifications": multi,
                "feedback_type_missing": session.scalar(base.where(fc.feedback_type.is_(None))),
                "vfc_level2_missing": status_counts.get("missing", 0),
                "vfc_no_class_found": status_counts.get("no_class_found", 0),
                "vfc_level3_missing": session.scalar(base.where(fc.vfc_level3_name.is_(None))),
                "vfc_classified": status_counts.get("classified", 0),
                "vfc_level2_3_separator_variants": l23_mismatch,
                "distinct_vfc_level2": session.scalar(
                    select(func.count(func.distinct(fc.vfc_level2_name))).where(fc.vehicle_code == code)
                ),
            }
        )
    return rows


def study_proportion_sums(session: Session) -> dict:
    """US_2025 answer proportions per (attribute, vehicle) should sum to ~1."""
    cs = CustomerStudy
    rows = session.execute(
        select(cs.attribute, cs.vehicle_label, func.sum(cs.value))
        .where(cs.study_name == "US_2025", cs.measure != "Sample total")
        .group_by(cs.attribute, cs.vehicle_label)
    ).all()
    off = {f"{a} | {v}": round(t, 3) for a, v, t in rows if abs(t - 1) > 0.02}
    return {"blocks": len(rows), "blocks_not_summing_to_1_within_0.02": off}


def vehicle_coverage(session: Session) -> dict[str, dict[str, bool]]:
    vehicles = ["F70", "G60", "G68", "G70"]
    feedback = {v for (v,) in session.execute(select(Feedback.vehicle_code).distinct())}
    studies = {v for (v,) in session.execute(select(CustomerStudy.vehicle_code).distinct())}
    sales = set()
    for (codes,) in session.execute(select(SalesVolume.vehicle_codes).distinct()):
        sales.update(codes.split(","))
    equipment = {v for (v,) in session.execute(select(Equipment.vehicle_code).distinct())}
    return {
        v: {
            "feedback": v in feedback,
            "customer_studies": v in studies,
            "sales": v in sales,
            "equipment_option_list": v in equipment,
        }
        for v in vehicles
    }


def study_vehicle_markets(session: Session) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for study, market, label in session.execute(
        select(CustomerStudy.study_name, CustomerStudy.survey_market, CustomerStudy.vehicle_label).distinct()
        .order_by(CustomerStudy.study_name, CustomerStudy.survey_market)
    ):
        out[f"{study} / {market}"].append(label)
    return dict(out)


def shared_vfc_level2(session: Session) -> dict:
    sets = {}
    for code in config.FEEDBACK_FILES:
        sets[code] = {
            v for (v,) in session.execute(
                select(FeedbackClassification.vfc_level2_name).where(
                    FeedbackClassification.vehicle_code == code,
                    FeedbackClassification.derived_classification_status == "classified",
                ).distinct()
            )
        }
    all_three = set.intersection(*sets.values())
    return {"per_vehicle": {k: len(v) for k, v in sets.items()}, "shared_by_all": len(all_three),
            "g60_only": len(sets["G60"] - sets["F70"] - sets["G70"])}


def country_distribution(session: Session) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for code, country, n in session.execute(
        select(Feedback.vehicle_code, Feedback.country, func.count()).group_by(Feedback.vehicle_code, Feedback.country)
    ):
        out[code][country or "NULL"] = n
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in out.items()}


def equipment_summary(session: Session) -> list[dict]:
    out = []
    for code in config.OPTION_LIST_FILES:
        e = Equipment
        out.append(
            {
                "vehicle": code,
                "option_entries": session.scalar(select(func.count()).select_from(e).where(e.vehicle_code == code)),
                "distinct_option_codes": session.scalar(
                    select(func.count(func.distinct(e.option_code))).where(e.vehicle_code == code)
                ),
                "pages_with_codes": session.scalar(select(func.count(func.distinct(e.source_page))).where(e.vehicle_code == code)),
                "availability_cells": session.scalar(
                    select(func.count()).select_from(EquipmentAvailability).where(EquipmentAvailability.vehicle_code == code)
                ),
                "entries_without_name": session.scalar(select(func.count()).select_from(e).where(e.vehicle_code == code, e.name.is_(None))),
                "entries_with_parse_note": session.scalar(
                    select(func.count()).select_from(e).where(e.vehicle_code == code, e.parse_note.is_not(None))
                ),
                "variants": sorted(
                    v for (v,) in session.execute(
                        select(EquipmentAvailability.variant).where(EquipmentAvailability.vehicle_code == code).distinct()
                    )
                ),
            }
        )
    return out


def pdf_limitations(session: Session) -> list[dict]:
    out = []
    for doc in session.scalars(select(SourceDocument).where(SourceDocument.file_type.in_(["pdf", "pptx"]))):
        pages = session.scalars(select(SourceDocumentPage).where(SourceDocumentPage.source_document_id == doc.document_id)).all()
        out.append(
            {
                "file": doc.file_name,
                "pages": len(pages),
                "pages_without_text": [p.page_number for p in pages if p.char_count == 0],
                "pages_with_images": sum(1 for p in pages if p.image_count),
                "total_chars": sum(p.char_count for p in pages),
            }
        )
    return out


REQUIRED_DATASETS = [
    config.FEEDBACK_FILES["G60"],
    config.CUSTOMER_STUDIES_FILE,
    config.SALES_FILE,
    config.COMPETITOR_EVIDENCE_FILE,
    config.PUBLIC_FEEDBACK_FILE,
]


def _dup_groups(session: Session, model, *columns) -> int:
    sub = select(*columns).group_by(*columns).having(func.count() > 1).subquery()
    return session.scalar(select(func.count()).select_from(sub)) or 0


def integrity_checks(session: Session, data_dir: Path, count_results: list[dict]) -> list[dict]:
    """The Part 2 integration checks, each recomputed against the database."""
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    docs = {d.file_name: d for d in session.scalars(select(SourceDocument))}
    not_ok = [f for f in REQUIRED_DATASETS if f not in docs or docs[f].ingestion_status != "imported"]
    add("1. Selected datasets imported", not not_ok,
        "all 5 imported" if not not_ok else f"not imported: {not_ok}")

    fb = [r for r in count_results if r["table"] in ("feedback", "feedback_classifications")]
    add("2. Feedback counts match source files", all(r["status"] == "ok" for r in fb),
        "; ".join(f"{r['file']} {r['table']} {r['imported']}/{r['expected']}" for r in fb))

    dups = {
        "feedback (file, feedback ID)": _dup_groups(session, Feedback, Feedback.source_file, Feedback.original_feedback_id),
        "feedback_classifications (file, row)": _dup_groups(
            session, FeedbackClassification, FeedbackClassification.source_file, FeedbackClassification.source_row),
        "public_feedback (public ID)": _dup_groups(session, PublicFeedback, PublicFeedback.original_public_id),
        "competitor_evidence (row)": _dup_groups(session, CompetitorEvidence, CompetitorEvidence.source_row),
        "customer_studies (sheet, row, column)": _dup_groups(
            session, CustomerStudy, CustomerStudy.source_sheet, CustomerStudy.source_row, CustomerStudy.source_column),
        "sales (sheet, row, column)": _dup_groups(
            session, SalesVolume, SalesVolume.source_sheet, SalesVolume.source_row, SalesVolume.source_column),
    }
    from_consolidated = sum(
        session.scalar(select(func.count()).select_from(m).where(m.source_file == config.CONSOLIDATED_FILE)) or 0
        for m in (Feedback, FeedbackClassification, PublicFeedback, CompetitorEvidence, CustomerStudy, SalesVolume)
    )
    add("3. No duplicate imports", not any(dups.values()) and from_consolidated == 0,
        f"duplicate key groups: {dups}; rows imported from consolidated workbook: {from_consolidated}")

    mismatched = 0
    for code, name in config.FEEDBACK_FILES.items():
        raw = pd.read_excel(find_file(data_dir, name), sheet_name=config.FEEDBACK_SHEET, dtype=object)
        raw_text = raw.drop_duplicates("ID").set_index("ID")["Customer Feedback"].to_dict()
        stored = dict(session.execute(
            select(Feedback.original_feedback_id, Feedback.customer_feedback).where(Feedback.source_file == name)
        ).all())
        mismatched += sum(1 for k, v in raw_text.items() if stored.get(k) != v) + len(set(stored) - set(raw_text))
    add("4. Original customer feedback preserved verbatim", mismatched == 0,
        f"{mismatched} feedback texts differ from source (compared every ID in all feedback files)")

    internal = set(session.scalars(select(Feedback.data_origin).distinct()))
    public = set(session.scalars(select(PublicFeedback.data_origin).distinct())) | set(
        session.scalars(select(CompetitorEvidence.data_origin).distinct()))
    public_files_in_internal = session.scalar(select(func.count()).select_from(Feedback).where(
        Feedback.source_file.in_([config.PUBLIC_FEEDBACK_FILE, config.COMPETITOR_EVIDENCE_FILE]))) or 0
    add("5. Public and internal feedback distinguishable",
        internal == {"internal_bmw"} and public == {"public_external"} and public_files_in_internal == 0,
        f"feedback.data_origin={sorted(internal)}; public tables data_origin={sorted(public)}; "
        "separate tables (feedback vs public_feedback / competitor_evidence)")

    wrong = [
        f"{g}: {v}" for g, v in session.execute(select(SalesVolume.sales_group, SalesVolume.vehicle_codes).distinct())
        if v != ",".join(config.SALES_SHEET_VEHICLES.get(g, []))
    ]
    add("6. Sales associated with correct models", not wrong,
        "sheet -> vehicles: " + ", ".join(f"{k}={'+'.join(v)}" for k, v in config.SALES_SHEET_VEHICLES.items())
        + (f"; mismatches: {wrong}" if wrong else ""))

    combined = session.execute(
        select(func.count(), func.sum(SalesVolume.is_combined_models.cast(Integer)))
        .where(SalesVolume.sales_group == "G60_G68")
    ).one()
    g60_only_rows = session.scalar(select(func.count()).select_from(SalesVolume).where(SalesVolume.vehicle_codes == "G60")) or 0
    label = session.scalar(select(SalesVolume.scope_label).where(SalesVolume.sales_group == "G60_G68").limit(1))
    add("7. G60/G68 combined sales not presented as G60-only",
        combined[0] > 0 and combined[0] == combined[1] and g60_only_rows == 0,
        f"{combined[0]} G60_G68 rows, all is_combined_models=True, scope_label='{label}'; G60-only sales rows: {g60_only_rows}")
    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_none_\n"
    columns = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns) + " |")
    return "\n".join(lines) + "\n"


def build_quality_report(session: Session, data_dir: Path, ingestion_report: dict, count_results: list[dict]) -> dict:
    docs = session.scalars(select(SourceDocument).order_by(SourceDocument.data_origin, SourceDocument.file_name)).all()
    overlaps = session.scalars(select(DatasetOverlap)).all()
    dupes = session.execute(
        select(PossibleDuplicate.table_name, PossibleDuplicate.rule, func.count(), func.sum(PossibleDuplicate.record_count))
        .group_by(PossibleDuplicate.table_name, PossibleDuplicate.rule)
    ).all()
    sales_zero = session.execute(
        select(SalesVolume.sales_group, SalesVolume.market_code, SalesVolume.year).where(SalesVolume.volume == 0)
    ).all()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "documents": [
            {
                "file": d.file_name, "dataset_type": d.dataset_type, "original_file_names": d.original_file_names or "",
                "origin": d.data_origin, "type": d.file_type, "sheets_or_pages": d.sheet_or_page_count,
                "source_of_truth": d.is_source_of_truth, "status": d.ingestion_status, "notes": d.ingestion_notes or "",
            }
            for d in docs
        ],
        "table_counts": table_counts(session),
        "count_validation": count_results,
        "integrity_checks": integrity_checks(session, data_dir, count_results),
        "provenance_gaps": provenance_gaps(session),
        "sheet_profiles": profile_all_sheets(data_dir),
        "feedback_quality": feedback_quality(session),
        "study_proportion_check": study_proportion_sums(session),
        "overlaps": [
            {
                "consolidated_sheet": o.consolidated_sheet, "original": f"{o.original_file or '-'} / {o.original_sheet or '-'}",
                "rows_consolidated": o.consolidated_rows, "rows_original": o.original_rows, "result": o.comparison_result,
                "differing": o.differing_cells, "decision": o.decision,
            }
            for o in overlaps
        ],
        "possible_duplicates": [
            {"table": t, "rule": r, "groups": g, "records_involved": n} for t, r, g, n in dupes
        ],
        "sales_zero_volumes": [f"{g} {m} {y}" for g, m, y in sales_zero],
        "vehicle_coverage": vehicle_coverage(session),
        "study_vehicle_markets": study_vehicle_markets(session),
        "shared_vfc_level2": shared_vfc_level2(session),
        "feedback_country_distribution": country_distribution(session),
        "equipment": equipment_summary(session),
        "text_extraction": pdf_limitations(session),
        "ingestion": ingestion_report,
    }


def render_markdown(report: dict) -> str:
    md: list[str] = []
    w = md.append
    w("# BMW ProductMind AI - Data Quality Report (Part 1)\n")
    w(f"Generated: {report['generated_at']}\n")
    w("All figures below are computed from the supplied files and the SQLite database. "
      "No values were imputed, generated or merged.\n")

    w("## 1. Source documents\n")
    w(_md_table(report["documents"]))

    w("## 2. Database tables and record counts\n")
    w(_md_table([{"table": k, "records": v} for k, v in report["table_counts"].items()]))

    w("## Integration checks\n")
    w(_md_table([{**c, "passed": "PASS" if c["passed"] else "FAIL"} for c in report["integrity_checks"]],
                ["check", "passed", "detail"]))

    w("## 3. Count validation (raw-file recount vs. database)\n")
    ok = sum(r["status"] == "ok" for r in report["count_validation"])
    w(f"{ok} of {len(report['count_validation'])} checks match.\n")
    w(_md_table(report["count_validation"], ["file", "sheet", "table", "basis", "expected", "imported", "status"]))
    w("Records missing mandatory provenance (record_id, source_file, sheet/page, imported_at):\n")
    w(_md_table([{"table": k, "records_with_gaps": v} for k, v in report["provenance_gaps"].items()]))

    w("## 4. Worksheet profiles\n")
    for p in report["sheet_profiles"]:
        w(f"### {p['file']} / {p['sheet']}\n")
        extra = ""
        if "key_column" in p:
            extra = (f"; key `{p['key_column']}`: {p['distinct_keys']} distinct, "
                     f"{p['keys_on_multiple_rows']} keys on >1 row (max {p['max_rows_per_key']})")
        header = "no header row (cross-tab / notes layout)" if p["header_row"] is None else f"header row {p['header_row']}"
        w(f"{p['rows']} rows, {header}; exact duplicate rows: {p['exact_duplicate_rows']}{extra}\n")
        w(_md_table(p["columns"], ["column", "inferred_type", "missing", "missing_pct", "distinct"]))

    w("## 5. Internal feedback quality\n")
    w(_md_table(report["feedback_quality"]))
    w("* One source row = one (feedback ID x VFC classification). IDs with several rows are multi-label, "
      "not duplicates; stored once in `feedback` and once per row in `feedback_classifications`.\n"
      "* `vfc_level2_3_separator_variants`: rows where `Vfc level2 3 Name` is not exactly "
      "`<level2> : <level3>` (e.g. `...Intervention: Malfunction`). Original strings preserved.\n"
      "* Missing `Feedback Type` / VFC values are kept as NULL; `no_class_found` is BMW's own label and is preserved.\n")

    w("## 6. Overlap between consolidated workbook and originals\n")
    w(_md_table(report["overlaps"]))

    w("## 7. Possible duplicates (flagged for review, never merged)\n")
    w(_md_table(report["possible_duplicates"]))
    w("Public feedback also carries explicit relations from its `Evidence note` column "
      "(e.g. 'Same review as P001'), stored in `derived_related_public_id` / `derived_relation_type`.\n")

    w("## 8. Customer studies checks\n")
    sc = report["study_proportion_check"]
    w(f"US_2025: {sc['blocks']} attribute x vehicle blocks; blocks whose seven answer proportions do not sum to 1 (+/-0.02): "
      f"{len(sc['blocks_not_summing_to_1_within_0.02'])}\n")
    if sc["blocks_not_summing_to_1_within_0.02"]:
        w(_md_table([{"block": k, "sum": v} for k, v in sc["blocks_not_summing_to_1_within_0.02"].items()]))

    w("## 9. Equipment (option-list PDFs)\n")
    w(_md_table(report["equipment"]))

    w("## 10. Text extraction (PDF / PPTX)\n")
    w(_md_table(report["text_extraction"]))

    w("## 11. Relationships between datasets\n")
    w("Vehicle coverage by dataset:\n")
    w(_md_table([{"vehicle": v, **c} for v, c in report["vehicle_coverage"].items()]))
    w("Vehicles per customer-study market:\n")
    w(_md_table([{"study / market": k, "vehicles": ", ".join(v)} for k, v in report["study_vehicle_markets"].items()]))
    sv = report["shared_vfc_level2"]
    w(f"VFC level-2 taxonomy: distinct classified level-2 names per vehicle {sv['per_vehicle']}; "
      f"{sv['shared_by_all']} shared by F70, G60 and G70; {sv['g60_only']} appear only for G60.\n")
    w("Feedback country distribution (feedback IDs):\n")
    w(_md_table([{"vehicle": k, "countries": ", ".join(f"{c}={n}" for c, n in v.items())}
                 for k, v in report["feedback_country_distribution"].items()]))

    countries = sorted({c for v in report["feedback_country_distribution"].values() for c in v})
    w("Join keys identified (for later parts; nothing below is materialised as data):\n")
    w(_md_table([
        {"from": "feedback.vehicle_code", "to": "customer_studies.vehicle_code / sales.vehicle_codes / equipment.vehicle_code",
         "type": "exact key", "caveat": "Sales for G60 are combined with G68 (sheet G60_G68); G68 has no feedback or option list."},
        {"from": "feedback.country (ISO-2)", "to": "sales.market_code (EU/CN/US/RoW)",
         "type": "candidate mapping", "caveat": f"Countries present: {', '.join(countries)}. No CN feedback. "
                                                "Assigning GB/NO and non-EU countries to EU vs RoW is an assumption to confirm."},
        {"from": "feedback.country", "to": "customer_studies.survey_market (China/EU/US)",
         "type": "candidate mapping", "caveat": "Survey markets are regions; same caveat as above."},
        {"from": "feedback_classifications.vfc_level2_name", "to": "public_feedback.theme / competitor_evidence.theme / customer_studies.attribute",
         "type": "thematic only", "caveat": "Different vocabularies; no shared key. Requires curated mapping (Part 2)."},
        {"from": "equipment (German option names, SA codes)", "to": "VFC feature names (English)",
         "type": "thematic only", "caveat": "No shared code; e.g. 'Driving Assistant Professional' vs VFC level-2 names."},
        {"from": "public_feedback.source_url", "to": "competitor_evidence.source_url",
         "type": "exact (URL)", "caveat": "Same public source cited in both external files; see possible_duplicates."},
        {"from": "BMW_G60_Complete_Prototype_Data.xlsx sheets", "to": "original files",
         "type": "exact copy", "caveat": "See section 6; originals are the source of truth."},
    ]))

    w("## 12. Zero sales volumes (stored as 0, not missing)\n")
    w(", ".join(report["sales_zero_volumes"]) + "\n" if report["sales_zero_volumes"] else "_none_\n")

    w("## 13. Files present but not registered\n")
    w(", ".join(report["ingestion"].get("unregistered_files", [])) + "\n" if report["ingestion"].get("unregistered_files") else "_none_\n")
    failed = {k: v for k, v in report["ingestion"]["files"].items() if v["status"] in ("failed", "missing")}
    w("## 14. Failed or missing files\n")
    w(_md_table([{"file": k, "status": v["status"], "error": v.get("error", "")} for k, v in failed.items()]))
    return "\n".join(md)


def write_reports(report: dict, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "data_quality_report.json"
    md_path = reports_dir / "data_quality_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return md_path, json_path


def summarize_for_console(report: dict) -> dict[str, Any]:
    return {
        "tables": report["table_counts"],
        "count_checks_ok": sum(r["status"] == "ok" for r in report["count_validation"]),
        "count_checks_total": len(report["count_validation"]),
        "provenance_gaps": sum(report["provenance_gaps"].values()),
        "integrity_checks_passed": f"{sum(c['passed'] for c in report['integrity_checks'])}/{len(report['integrity_checks'])}",
    }
