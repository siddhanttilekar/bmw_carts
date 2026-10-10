"""Ingestion tests against the real supplied datasets."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest
from sqlalchemy import func, select

from core import config
from core.models import (
    CompetitorEvidence,
    CustomerStudy,
    DatasetOverlap,
    Equipment,
    EquipmentAvailability,
    Feedback,
    FeedbackClassification,
    PossibleDuplicate,
    PublicFeedback,
    SalesVolume,
    SourceDocument,
    SourceDocumentPage,
)
from scripts.initialize_database import initialize
from services.data_ingestion import make_record_id


def raw_feedback(data_dir, code: str) -> pd.DataFrame:
    path = next(data_dir.rglob(config.FEEDBACK_FILES[code]))
    return pd.read_excel(path, sheet_name=config.FEEDBACK_SHEET, dtype=object)


def count(session, model, *where) -> int:
    return session.scalar(select(func.count()).select_from(model).where(*where))


# --- files and counts -------------------------------------------------------


def test_every_supplied_file_is_registered_and_none_failed(built, session):
    files = built["report"]["ingestion"]["files"]
    assert not [n for n, i in files.items() if i["status"] in ("failed", "missing")]
    assert built["report"]["ingestion"]["unregistered_files"] == []
    assert count(session, SourceDocument) == 13


def test_raw_recount_matches_database(built):
    results = built["report"]["count_validation"]
    assert results and all(r["status"] == "ok" for r in results), [r for r in results if r["status"] != "ok"]


def test_every_record_has_complete_provenance(built):
    assert all(v == 0 for v in built["report"]["provenance_gaps"].values())


def test_feedback_source_rows_cover_the_worksheet_exactly(session, data_dir):
    for code in config.FEEDBACK_FILES:
        n_raw = len(raw_feedback(data_dir, code))
        rows = session.scalars(
            select(FeedbackClassification.source_row).where(FeedbackClassification.vehicle_code == code)
        ).all()
        assert sorted(rows) == list(range(2, n_raw + 2))


# --- integrity: preservation, no fabrication ---------------------------------


def test_vfc_classifications_preserved_verbatim(session, data_dir):
    raw = raw_feedback(data_dir, "G60")
    db = {
        r.source_row: r
        for r in session.scalars(select(FeedbackClassification).where(FeedbackClassification.vehicle_code == "G60"))
    }
    columns = {
        "Feedback Type": "feedback_type",
        "Vfc level2 3 Name": "vfc_level2_3_name",
        "Vfc level2 Name": "vfc_level2_name",
        "Vfc level3 Name": "vfc_level3_name",
        "ID": "original_feedback_id",
    }
    for idx, rec in raw.iterrows():
        stored = db[idx + 2]
        for raw_col, attr in columns.items():
            expected = None if pd.isna(rec[raw_col]) else rec[raw_col]
            assert getattr(stored, attr) == expected, (idx + 2, raw_col)


def test_missing_values_stay_missing(session, data_dir):
    for code in config.FEEDBACK_FILES:
        raw = raw_feedback(data_dir, code)
        fc = FeedbackClassification
        assert count(session, fc, fc.vehicle_code == code, fc.feedback_type.is_(None)) == raw["Feedback Type"].isna().sum()
        assert count(session, fc, fc.vehicle_code == code, fc.vfc_level2_name.is_(None)) == raw["Vfc level2 Name"].isna().sum()
        assert count(session, fc, fc.vehicle_code == code, fc.vfc_level3_name.is_(None)) == raw["Vfc level3 Name"].isna().sum()


def test_feedback_text_is_stored_unchanged(session, data_dir):
    raw = raw_feedback(data_dir, "F70").drop_duplicates("ID").set_index("ID")["Customer Feedback"]
    stored = dict(session.execute(select(Feedback.original_feedback_id, Feedback.customer_feedback).where(Feedback.vehicle_code == "F70")).all())
    assert stored == raw.to_dict()


def test_similar_texts_are_flagged_not_merged(session):
    assert count(session, Feedback, Feedback.vehicle_code == "G60") == 4000
    assert count(session, Feedback, Feedback.vehicle_code == "G70") == 4000
    flagged = session.scalars(
        select(PossibleDuplicate).where(PossibleDuplicate.rule == "identical_normalized_text_different_ids")
    ).all()
    assert flagged and all(d.review_status == "unreviewed" and d.record_count >= 2 for d in flagged)


def test_record_ids_are_deterministic():
    assert make_record_id("a.xlsx", "Sheet", None, 2) == make_record_id("a.xlsx", "Sheet", None, 2)
    assert make_record_id("a.xlsx", "Sheet", None, 2) != make_record_id("a.xlsx", "Sheet", None, 3)


# --- separation of public vs internal, URLs ---------------------------------


def test_public_feedback_is_separate_from_internal(session):
    assert set(session.scalars(select(Feedback.data_origin).distinct())) == {"internal_bmw"}
    assert set(session.scalars(select(PublicFeedback.data_origin).distinct())) == {"public_external"}
    assert set(session.scalars(select(CompetitorEvidence.data_origin).distinct())) == {"public_external"}
    assert count(session, Feedback, Feedback.original_feedback_id.like("P0%")) == 0
    public_files = {config.PUBLIC_FEEDBACK_FILE, config.COMPETITOR_EVIDENCE_FILE}
    assert not public_files & set(session.scalars(select(Feedback.source_file).distinct()))


def test_source_urls_preserved(session, data_dir):
    path = next(data_dir.rglob(config.PUBLIC_FEEDBACK_FILE))
    raw = pd.read_excel(path, sheet_name="Public G60 Feedback")
    stored = dict(session.execute(select(PublicFeedback.original_public_id, PublicFeedback.source_url)).all())
    assert stored == dict(zip(raw["ID"], raw["Source URL"]))
    assert count(session, CompetitorEvidence, CompetitorEvidence.source_url.is_(None)) == 0


def test_public_feedback_relations_parsed_from_evidence_note(session):
    p002 = session.scalar(select(PublicFeedback).where(PublicFeedback.original_public_id == "P002"))
    assert (p002.derived_related_public_id, p002.derived_relation_type) == ("P001", "same_review")
    assert p002.evidence_note.startswith("Same review as P001")


# --- single source of truth --------------------------------------------------


def test_consolidated_workbook_not_double_imported(session):
    for model in (Feedback, FeedbackClassification, CompetitorEvidence, CustomerStudy, SalesVolume):
        assert count(session, model, model.source_file == config.CONSOLIDATED_FILE) == 0
    doc = session.scalar(select(SourceDocument).where(SourceDocument.file_name == config.CONSOLIDATED_FILE))
    assert doc.is_source_of_truth is False and doc.ingestion_status == "registered_only"
    results = dict(session.execute(select(DatasetOverlap.consolidated_sheet, DatasetOverlap.comparison_result)).all())
    for sheet in ("G60 Feedback All", "Public Competitors", "Sales G60 G68", "Study CN EU 2025", "Study US 2025"):
        assert results[sheet] == "exact_copy"
    assert results["Feedback Theme Counts"] == "derived_aggregate"


# --- studies, sales ---------------------------------------------------------


def test_customer_study_cells_trace_back_to_source(session, data_dir):
    path = next(data_dir.rglob(config.CUSTOMER_STUDIES_FILE))
    for sheet in ("CN_EU_2025", "US_2025"):
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for rec in session.scalars(select(CustomerStudy).where(CustomerStudy.study_name == sheet)):
            col = ord(rec.source_column) - ord("A")
            assert raw.iat[rec.source_row - 1, col] == pytest.approx(rec.value)
    g60_eu = session.scalar(
        select(CustomerStudy).where(
            CustomerStudy.study_name == "CN_EU_2025", CustomerStudy.vehicle_code == "G60",
            CustomerStudy.attribute == "Overall product satisfaction",
        )
    )
    assert g60_eu.survey_market == "EU" and g60_eu.measure == "Mean"


def test_sales_volumes_and_planning_flag(session):
    rec = session.scalar(
        select(SalesVolume).where(SalesVolume.sales_group == "G60_G68", SalesVolume.market_code == "EU", SalesVolume.year == 2024)
    )
    assert rec.volume == 110000 and rec.vehicle_codes == "G60,G68" and not rec.is_planning_figure
    assert count(session, SalesVolume, SalesVolume.year == 2030, SalesVolume.is_planning_figure.is_(False)) == 0


# --- PDFs ------------------------------------------------------------------


def test_g60_equipment_extracted_with_page_numbers(session):
    dap = session.scalar(select(Equipment).where(Equipment.vehicle_code == "G60", Equipment.option_code == "5AU"))
    assert dap.name == "Driving Assistant Professional"
    assert dap.price_gross_eur == 2100.0 and dap.source_page == 18
    adaptive = session.scalar(
        select(EquipmentAvailability).where(
            EquipmentAvailability.vehicle_code == "G60", EquipmentAvailability.option_code == "2VV",
            EquipmentAvailability.variant == "i5 xDrive40", EquipmentAvailability.condition_text.is_(None),
        )
    )
    assert adaptive.cell_raw == "■" and adaptive.availability == "standard"


def test_pdf_pages_preserve_numbering(session):
    pages = session.scalars(
        select(SourceDocumentPage.page_number).where(SourceDocumentPage.source_file == config.OPTION_LIST_FILES["G60"])
    ).all()
    assert sorted(pages) == list(range(1, 29))


def test_image_only_briefing_is_documented_not_invented(session):
    doc = session.scalar(select(SourceDocument).where(SourceDocument.file_name == config.BRIEFING_PDF_FILE))
    assert doc.ingestion_status == "no_text_layer" and "OCR not performed" in doc.ingestion_notes
    texts = session.scalars(select(SourceDocumentPage.text).where(SourceDocumentPage.source_document_id == doc.document_id)).all()
    assert len(texts) == 2 and all(t.strip() == "" for t in texts)


# --- failure handling --------------------------------------------------------


def test_unparseable_file_is_documented_and_others_still_load(tmp_path, data_dir):
    """Drop a required column from a copy of a real file: that file must be
    marked failed with a reason, and no partial rows from it may remain."""
    broken_dir = tmp_path / "data"
    shutil.copytree(data_dir, broken_dir)
    target = next(broken_dir.rglob(config.FEEDBACK_FILES["G70"]))
    df = pd.read_excel(target, sheet_name=config.FEEDBACK_SHEET, dtype=object).drop(columns=["Vfc level2 Name"])
    df.to_excel(target, sheet_name=config.FEEDBACK_SHEET, index=False)

    report = initialize(tmp_path / "broken.db", broken_dir, None)
    status = report["ingestion"]["files"][config.FEEDBACK_FILES["G70"]]
    assert status["status"] == "failed" and "Vfc level2 Name" in status["error"]
    assert report["ingestion"]["files"][config.FEEDBACK_FILES["G60"]]["status"] == "imported"
    assert report["table_counts"]["feedback"] == 2128 + 4000  # F70 + G60 only
    mismatched = {r["file"] for r in report["count_validation"] if r["status"] != "ok"}
    assert mismatched == {config.FEEDBACK_FILES["G70"]}
