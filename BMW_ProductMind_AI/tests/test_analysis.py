"""Analysis-service tests. Expected values are computed independently with
pandas from the raw G60 workbook, never from the database."""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import func, select

from core import config
from core.models import Feedback, FeedbackClassification, SalesVolume
from scripts.initialize_database import initialize
from services.customer_insights import (
    category_profile,
    customer_study_values,
    feedback_overview,
    public_evidence_summary,
    sales_context,
    top_categories_for_feedback_type,
)
from services.evidence_retrieval import (
    competitor_evidence,
    feedback_for_category,
    get_feedback_by_id,
    get_feedback_by_ids,
    missing_ids,
    public_owner_feedback,
    search_feedback_text,
)
from services.feedback_analysis import (
    FeedbackFilter,
    count_feedback,
    distribution_by,
    feedback_type_distribution,
    filter_feedback,
    matching_feedback_ids,
    vfc_category_counts,
)


@pytest.fixture(scope="module")
def g60(data_dir) -> pd.DataFrame:
    path = next(data_dir.rglob(config.FEEDBACK_FILES["G60"]))
    df = pd.read_excel(path, sheet_name=config.FEEDBACK_SHEET, dtype=object)
    return df.astype(object).where(df.notna(), None)


def _key(value):
    return "<None>" if value is None else value


# --- counts and grouping ------------------------------------------------------


def test_count_matches_source(session, g60):
    result = count_feedback(session)
    assert (result.feedback_ids, result.classification_rows) == (g60["ID"].nunique(), len(g60))


def test_vfc_category_counts_match_source(session, g60):
    expected_rows = g60.groupby(g60["Vfc level2 Name"].map(_key))["ID"].size().to_dict()
    expected_ids = g60.groupby(g60["Vfc level2 Name"].map(_key))["ID"].nunique().to_dict()
    result = vfc_category_counts(session)
    assert {_key(c.category): c.classification_rows for c in result} == expected_rows
    assert {_key(c.category): c.feedback_ids for c in result} == expected_ids
    assert sum(c.classification_rows for c in result) == len(g60)
    assert [c.feedback_ids for c in result] == sorted((c.feedback_ids for c in result), reverse=True)


def test_vfc_level3_counts_match_source(session, g60):
    expected = g60.groupby(g60["Vfc level3 Name"].map(_key)).size().to_dict()
    assert {_key(c.category): c.classification_rows for c in vfc_category_counts(session, level="level3")} == expected


def test_feedback_type_distribution_matches_source(session, g60):
    expected = g60["Feedback Type"].map(_key).value_counts().to_dict()
    result = feedback_type_distribution(session)
    assert {_key(t.feedback_type): t.classification_rows for t in result} == expected
    assert sum(t.share_of_rows for t in result) == pytest.approx(1.0)
    # Missing values are reported as missing, never relabelled.
    assert {t.feedback_type for t in result} == {None, "Likes", "Difficult to Use", "Defect", "Wants"}


def test_distribution_by_engine_and_country(session, g60):
    by_engine = {d.value: d.feedback_ids for d in distribution_by(session, "engine_type")}
    assert by_engine == g60.drop_duplicates("ID")["Engine Type"].value_counts().to_dict()
    by_country = {d.value: d.feedback_ids for d in distribution_by(session, "country")}
    assert by_country == g60.drop_duplicates("ID")["Country"].value_counts().to_dict()


# --- filters ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "flt_kwargs, mask",
    [
        ({"countries": ["US"]}, lambda d: d["Country"] == "US"),
        ({"engine_types": ["BEVE"]}, lambda d: d["Engine Type"] == "BEVE"),
        ({"countries": ["US"], "engine_types": ["PHEV"]}, lambda d: (d["Country"] == "US") & (d["Engine Type"] == "PHEV")),
        ({"countries": ["KR", "JP"], "engine_types": ["ICE"]},
         lambda d: d["Country"].isin(["KR", "JP"]) & (d["Engine Type"] == "ICE")),
        ({"feedback_types": ["Defect"]}, lambda d: d["Feedback Type"] == "Defect"),
        ({"feedback_types": [None]}, lambda d: d["Feedback Type"].isna()),
        ({"vfc_level2": ["Drivetrain"], "engine_types": ["BEVE"]},
         lambda d: (d["Vfc level2 Name"] == "Drivetrain") & (d["Engine Type"] == "BEVE")),
    ],
)
def test_filters_return_exactly_the_matching_ids(session, g60, flt_kwargs, mask):
    expected = sorted(g60.loc[mask(g60), "ID"].unique())
    flt = FeedbackFilter(**flt_kwargs)
    assert matching_feedback_ids(session, flt) == expected
    assert count_feedback(session, flt).feedback_ids == len(expected)


def test_filter_feedback_pages_return_full_records(session, g60):
    flt = FeedbackFilter(countries=["DK"])
    page = filter_feedback(session, flt, limit=5, offset=0)
    expected_ids = sorted(g60.loc[g60["Country"] == "DK", "ID"].unique())
    assert page.total_feedback_ids == len(expected_ids)
    assert [r.original_feedback_id for r in page.items] == expected_ids[:5]
    assert all(r.country == "DK" for r in page.items)


def test_unknown_filter_values_return_nothing(session):
    assert count_feedback(session, FeedbackFilter(countries=["XX"])).feedback_ids == 0


# --- G60 scoping ------------------------------------------------------------------


def test_g60_analysis_excludes_other_vehicles(session):
    ids = matching_feedback_ids(session)
    assert ids and all(i.startswith("G60-") for i in ids)
    assert get_feedback_by_id(session, "F70-0001") is None
    assert get_feedback_by_id(session, "G70-0001") is None
    assert get_feedback_by_id(session, "F70-0001", vehicle_code=None).vehicle_code == "F70"


def test_unknown_vehicle_raises(session):
    with pytest.raises(ValueError):
        count_feedback(session, FeedbackFilter(vehicle_code="G68"))


# --- retrieval and provenance -------------------------------------------------------


def test_retrieve_original_feedback_by_id(session, g60):
    rows = g60[g60["ID"] == "G60-0002"]
    record = get_feedback_by_id(session, "G60-0002")
    assert record.customer_feedback == rows["Customer Feedback"].iloc[0]
    assert record.country == rows["Country"].iloc[0] and record.engine_type == rows["Engine Type"].iloc[0]
    assert [c.source.source_row for c in record.classifications] == [i + 2 for i in rows.index]
    assert [c.vfc_level2_3_name for c in record.classifications] == rows["Vfc level2 3 Name"].tolist()


def test_multi_label_feedback_keeps_every_classification(session, g60):
    counts = g60["ID"].value_counts()
    feedback_id = counts.index[0]
    record = get_feedback_by_id(session, feedback_id)
    assert len(record.classifications) == counts.iloc[0] > 1


def test_every_retrieved_record_has_a_source_reference(session):
    records = filter_feedback(session, FeedbackFilter(feedback_types=["Defect"]), limit=100).items
    records += feedback_for_category(session, "Seats", limit=20)
    records += search_feedback_text(session, "touchscreen")[1]
    assert records
    for r in records:
        assert r.source.record_id and r.source.source_file == config.FEEDBACK_FILES["G60"]
        assert r.source.source_sheet == config.FEEDBACK_SHEET and r.source.source_row >= 2
        assert r.source.data_origin == "internal_bmw"
        for c in r.classifications:
            assert c.source.record_id and c.source.source_row >= 2
    for p in public_owner_feedback(session):
        assert p.source.data_origin == "public_external" and p.source.source_row and p.source_url
    for c in competitor_evidence(session):
        assert c.source.data_origin == "public_external" and c.source.source_row and c.source_url


def test_get_by_ids_keeps_order_and_reports_missing(session):
    ids = ["G60-0010", "G60-0003", "G60-9999"]
    assert [r.original_feedback_id for r in get_feedback_by_ids(session, ids)] == ["G60-0010", "G60-0003"]
    assert missing_ids(session, ids) == ["G60-9999"]


def test_text_search_is_literal(session, g60):
    total, records = search_feedback_text(session, "TouchScreen", limit=500)
    expected = g60.loc[g60["Customer Feedback"].str.lower().str.contains("touchscreen", regex=False), "ID"].nunique()
    assert total == expected == len(records)
    assert search_feedback_text(session, "100%")[0] == g60.loc[
        g60["Customer Feedback"].str.contains("100%", regex=False), "ID"].nunique()


# --- insights -------------------------------------------------------------------------


def test_top_defect_categories_match_source(session, g60):
    defects = g60[g60["Feedback Type"] == "Defect"]
    expected = defects.groupby("Vfc level2 Name")["ID"].nunique()
    for cat in top_categories_for_feedback_type(session, "Defect", top_n=5):
        assert cat.feedback_ids == expected[cat.category]
    assert top_categories_for_feedback_type(session, "Defect", top_n=1)[0].feedback_ids == expected.max()


def test_category_profile(session, g60):
    profile = category_profile(session, "Drivetrain")
    assert profile.counts.feedback_ids == g60.loc[g60["Vfc level2 Name"] == "Drivetrain", "ID"].nunique()
    assert sum(t.classification_rows for t in profile.feedback_types) == (g60["Vfc level2 Name"] == "Drivetrain").sum()


def test_overview_documents_feedback_type_limitation(session):
    overview = feedback_overview(session)
    assert any("not comment-level sentiment" in n for n in overview.notes)
    assert overview.model_dump()["counts"]["feedback_ids"] == 4000


def test_sales_context_never_presents_combined_volumes_as_g60(session):
    ctx = sales_context(session, "G60")
    assert ctx.records and all(r.is_combined_models and r.vehicles_covered == ["G60", "G68"] for r in ctx.records)
    assert all("combined" in r.scope_label for r in ctx.records)
    assert any("No G60-only volumes" in w for w in ctx.warnings)
    assert all(not r.is_combined_models for r in sales_context(session, "G70").records)


def test_customer_study_values_for_g60(session):
    values = customer_study_values(session, "G60")
    assert {v.study_name for v in values} == {"CN_EU_2025", "US_2025"}
    assert all(v.vehicle_code == "G60" and v.source.source_row for v in values)
    assert {v.survey_market for v in values if v.study_name == "CN_EU_2025"} == {"EU"}


def test_public_evidence_summary_counts(session):
    summary = public_evidence_summary(session)
    assert sum(r["observations"] for r in summary.public_feedback_by_theme_and_sentiment) == 15
    assert sum(r["observations"] for r in summary.competitor_evidence_by_brand_and_sentiment) == 18


# --- repeated ingestion ---------------------------------------------------------------


def test_repeated_ingestion_does_not_duplicate(tmp_path, data_dir):
    db = tmp_path / "twice.db"
    first = initialize(db, data_dir, None)
    second = initialize(db, data_dir, None)
    assert first["table_counts"] == second["table_counts"]
    assert all(c["passed"] for c in second["integrity_checks"])

    from core.database import get_engine, session_scope

    engine = get_engine(db)
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Feedback)) == 2128 + 4000 + 4000
        assert s.scalar(select(func.count()).select_from(FeedbackClassification)) == 4076 + 5005 + 4690
        assert s.scalar(select(func.count()).select_from(SalesVolume)) == 36
    engine.dispose()
