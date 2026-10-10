"""Deterministic customer-intelligence views built on the stored data.

No scores, sentiment or priorities are invented: every number is a count,
share, survey value or sales volume read from the database, and every result
states its scope (vehicle, units, caveats). Sales for G60 are only available
combined with G68 and are always labelled that way.
"""
from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.models import CompetitorEvidence, CustomerStudy, PublicFeedback, SalesVolume
from services.feedback_analysis import (
    FEEDBACK_TYPE_NOTE,
    CategoryCount,
    DimensionCount,
    FeedbackCount,
    FeedbackFilter,
    FeedbackTypeCount,
    SourceReference,
    count_feedback,
    distribution_by,
    feedback_type_distribution,
    vfc_category_counts,
)


class FeedbackOverview(BaseModel):
    counts: FeedbackCount
    feedback_types: list[FeedbackTypeCount]
    top_vfc_level2: list[CategoryCount]
    by_country: list[DimensionCount]
    by_engine_type: list[DimensionCount]
    by_feedback_source: list[DimensionCount]
    notes: list[str]


class CategoryProfile(BaseModel):
    vfc_level2: str | None
    counts: FeedbackCount
    feedback_types: list[FeedbackTypeCount]
    vfc_level3: list[CategoryCount]
    by_country: list[DimensionCount]
    by_engine_type: list[DimensionCount]
    share_of_vehicle_feedback_ids: float


class EngineTypeCategoryShare(BaseModel):
    engine_type: str | None
    engine_type_feedback_ids: int
    categories: list[CategoryCount]


class StudyValue(BaseModel):
    study_name: str
    survey_market: str
    vehicle_label: str
    vehicle_code: str | None
    attribute: str
    measure: str
    value_type: str
    value: float
    source: SourceReference


class SalesRecord(BaseModel):
    sales_group: str
    vehicles_covered: list[str]
    scope_label: str
    is_combined_models: bool
    market: str
    market_code: str
    year: int
    volume: int | None
    is_planning_figure: bool
    source: SourceReference


class SalesContext(BaseModel):
    requested_vehicle: str
    records: list[SalesRecord]
    warnings: list[str]


class PublicEvidenceSummary(BaseModel):
    public_feedback_by_theme_and_sentiment: list[dict]
    competitor_evidence_by_brand_and_sentiment: list[dict]
    notes: list[str]


# ---------------------------------------------------------------------------
# Internal feedback views
# ---------------------------------------------------------------------------


def feedback_overview(session: Session, flt: FeedbackFilter | None = None, top_n: int = 15) -> FeedbackOverview:
    flt = flt or FeedbackFilter()
    return FeedbackOverview(
        counts=count_feedback(session, flt),
        feedback_types=feedback_type_distribution(session, flt),
        top_vfc_level2=vfc_category_counts(session, flt, "level2")[:top_n],
        by_country=distribution_by(session, "country", flt),
        by_engine_type=distribution_by(session, "engine_type", flt),
        by_feedback_source=distribution_by(session, "feedback_source", flt),
        notes=[
            FEEDBACK_TYPE_NOTE,
            "feedback_ids = distinct customer comments; classification_rows = source rows (ID x VFC label).",
            "Counts are frequencies in the supplied export, not population prevalence.",
            "'no_class_found' is BMW's own VFC label; '(missing VFC classification)' means the source cell is empty.",
        ],
    )


def top_categories_for_feedback_type(
    session: Session, feedback_type: str | None, flt: FeedbackFilter | None = None, top_n: int = 10
) -> list[CategoryCount]:
    """VFC level-2 categories ranked by comments with the given Feedback Type
    (e.g. 'Defect'). Shares are relative to comments with that type."""
    flt = (flt or FeedbackFilter()).model_copy(update={"feedback_types": [feedback_type]})
    return vfc_category_counts(session, flt, "level2")[:top_n]


def category_profile(session: Session, vfc_level2: str | None, flt: FeedbackFilter | None = None) -> CategoryProfile:
    base = flt or FeedbackFilter()
    scoped = base.model_copy(update={"vfc_level2": [vfc_level2]})
    counts = count_feedback(session, scoped)
    vehicle_total = count_feedback(session, base).feedback_ids
    return CategoryProfile(
        vfc_level2=vfc_level2,
        counts=counts,
        feedback_types=feedback_type_distribution(session, scoped),
        vfc_level3=vfc_category_counts(session, scoped, "level3"),
        by_country=distribution_by(session, "country", scoped),
        by_engine_type=distribution_by(session, "engine_type", scoped),
        share_of_vehicle_feedback_ids=round(counts.feedback_ids / vehicle_total, 6) if vehicle_total else 0.0,
    )


def engine_type_category_shares(
    session: Session, flt: FeedbackFilter | None = None, top_n: int = 10
) -> list[EngineTypeCategoryShare]:
    """Per engine type (ICE / BEVE / PHEV as supplied), the top VFC level-2
    categories with shares normalised by that engine type's comment count."""
    base = flt or FeedbackFilter()
    out = []
    for engine in distribution_by(session, "engine_type", base):
        scoped = base.model_copy(update={"engine_types": [engine.value]})
        out.append(EngineTypeCategoryShare(
            engine_type=engine.value, engine_type_feedback_ids=engine.feedback_ids,
            categories=vfc_category_counts(session, scoped, "level2")[:top_n],
        ))
    return out


# ---------------------------------------------------------------------------
# Customer studies and sales
# ---------------------------------------------------------------------------


def customer_study_values(
    session: Session, vehicle_code: str = "G60", study_name: str | None = None, attribute_contains: str | None = None
) -> list[StudyValue]:
    """Survey cells for one vehicle. CN/EU values are means (scale not stated
    in the source); US values are answer-share proportions per attribute."""
    stmt = select(CustomerStudy).where(CustomerStudy.vehicle_code == vehicle_code)
    if study_name:
        stmt = stmt.where(CustomerStudy.study_name == study_name)
    if attribute_contains:
        stmt = stmt.where(func.lower(CustomerStudy.attribute).contains(attribute_contains.lower()))
    stmt = stmt.order_by(CustomerStudy.study_name, CustomerStudy.source_row, CustomerStudy.source_column)
    return [
        StudyValue(
            study_name=s.study_name, survey_market=s.survey_market, vehicle_label=s.vehicle_label,
            vehicle_code=s.vehicle_code, attribute=s.attribute, measure=s.measure, value_type=s.value_type,
            value=s.value,
            source=SourceReference(record_id=s.record_id, source_file=s.source_file, source_sheet=s.source_sheet,
                                   source_row=s.source_row, data_origin="internal_bmw"),
        )
        for s in session.scalars(stmt)
    ]


def sales_context(session: Session, vehicle_code: str = "G60") -> SalesContext:
    """Sales rows that *include* the vehicle. Combined groups are returned with
    their full scope and a warning; volumes are never split or attributed."""
    rows = [
        s for s in session.scalars(select(SalesVolume).order_by(SalesVolume.market_code, SalesVolume.year))
        if vehicle_code in s.vehicle_codes.split(",")
    ]
    warnings = []
    if any(r.is_combined_models for r in rows):
        groups = sorted({r.sales_group for r in rows if r.is_combined_models})
        warnings.append(
            f"No {vehicle_code}-only volumes exist. Group(s) {groups} combine several models "
            f"({rows[0].scope_label}); do not present them as {vehicle_code} volumes."
        )
    if any(r.is_planning_figure for r in rows):
        warnings.append("Years flagged is_planning_figure (2030) are plan/forecast figures, not historical sales.")
    if not rows:
        warnings.append(f"No sales data includes {vehicle_code}.")
    return SalesContext(
        requested_vehicle=vehicle_code,
        warnings=warnings,
        records=[
            SalesRecord(
                sales_group=r.sales_group, vehicles_covered=r.vehicle_codes.split(","), scope_label=r.scope_label,
                is_combined_models=r.is_combined_models, market=r.market, market_code=r.market_code, year=r.year,
                volume=r.volume, is_planning_figure=r.is_planning_figure,
                source=SourceReference(record_id=r.record_id, source_file=r.source_file, source_sheet=r.source_sheet,
                                       source_row=r.source_row, data_origin="internal_bmw"),
            )
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# Public evidence
# ---------------------------------------------------------------------------


def public_evidence_summary(session: Session) -> PublicEvidenceSummary:
    pf = session.execute(
        select(PublicFeedback.theme, PublicFeedback.sentiment, func.count())
        .group_by(PublicFeedback.theme, PublicFeedback.sentiment).order_by(PublicFeedback.theme)
    ).all()
    ce = session.execute(
        select(CompetitorEvidence.brand, CompetitorEvidence.sentiment, func.count())
        .group_by(CompetitorEvidence.brand, CompetitorEvidence.sentiment).order_by(CompetitorEvidence.brand)
    ).all()
    return PublicEvidenceSummary(
        public_feedback_by_theme_and_sentiment=[{"theme": t, "sentiment": s, "observations": n} for t, s, n in pf],
        competitor_evidence_by_brand_and_sentiment=[{"brand": b, "sentiment": s, "observations": n} for b, s, n in ce],
        notes=[
            "Public data is curated, paraphrased evidence (data_origin=public_external), kept separate from BMW feedback.",
            "Sentiment values are as labelled in the source files, not computed here.",
            "Rows are observations, not independent customers: several rows can come from one review "
            "(see related_public_id). Do not compute population percentages.",
        ],
    )
