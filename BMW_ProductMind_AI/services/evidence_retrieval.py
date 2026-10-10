"""Retrieve original evidence records with their source references.

Internal BMW feedback and public evidence are returned as different types
(``FeedbackRecord`` vs ``PublicFeedbackRecord`` / ``CompetitorEvidenceRecord``),
each carrying ``data_origin`` and a ``SourceReference`` back to file/sheet/row.
Text search is a literal, case-insensitive substring match - no semantic or
generated matching.
"""
from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.models import CompetitorEvidence, Feedback, PublicFeedback, SourceDocument
from services.feedback_analysis import (
    FeedbackFilter,
    FeedbackRecord,
    SourceReference,
    _joined,
    build_feedback_records,
    filter_conditions,
)


class PublicFeedbackRecord(BaseModel):
    original_public_id: str
    model: str | None
    model_year: str | None
    review_date: str | None
    source_name: str | None
    source_type: str | None
    market: str | None
    sentiment_as_labelled_in_source: str | None
    theme: str | None
    paraphrased_feedback: str | None
    product_implication: str | None
    source_url: str | None
    evidence_note: str | None
    related_public_id: str | None
    relation_type: str | None
    source: SourceReference


class CompetitorEvidenceRecord(BaseModel):
    brand: str | None
    model: str | None
    is_bmw: bool
    source_type: str | None
    date_or_period: str | None
    market: str | None
    theme: str | None
    sentiment_as_labelled_in_source: str | None
    extracted_insight: str | None
    evidence_limitation: str | None
    source_url: str | None
    source: SourceReference


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _contains(column, text: str):
    return func.lower(column).like(f"%{_escape_like(text.lower())}%", escape="\\")


def _names(session: Session) -> dict[str, str | None]:
    return dict(session.execute(select(SourceDocument.document_id, SourceDocument.original_file_names)).all())


def _ref(obj, names: dict) -> SourceReference:
    return SourceReference(record_id=obj.record_id, source_file=obj.source_file, source_sheet=obj.source_sheet,
                           source_row=obj.source_row, source_page=obj.source_page, data_origin=obj.data_origin,
                           original_file_names=names.get(obj.source_document_id))


# ---------------------------------------------------------------------------
# Internal BMW feedback
# ---------------------------------------------------------------------------


def get_feedback_by_id(session: Session, feedback_id: str, vehicle_code: str | None = "G60") -> FeedbackRecord | None:
    """Original comment + all its VFC classification rows. ``vehicle_code``
    scopes the lookup (pass ``None`` to search every vehicle)."""
    stmt = select(Feedback).where(Feedback.original_feedback_id == feedback_id)
    if vehicle_code is not None:
        stmt = stmt.where(Feedback.vehicle_code == vehicle_code)
    rows = session.scalars(stmt).all()
    if len(rows) > 1:
        raise ValueError(f"Feedback ID {feedback_id!r} is ambiguous across files; pass vehicle_code")
    records = build_feedback_records(session, rows)
    return records[0] if records else None


def get_feedback_by_ids(session: Session, feedback_ids: list[str], vehicle_code: str = "G60") -> list[FeedbackRecord]:
    """Records in the order requested; unknown IDs are skipped (see missing_ids)."""
    rows = session.scalars(
        select(Feedback).where(Feedback.vehicle_code == vehicle_code, Feedback.original_feedback_id.in_(feedback_ids))
    ).all()
    by_id = {r.original_feedback_id: r for r in build_feedback_records(session, rows)}
    return [by_id[i] for i in feedback_ids if i in by_id]


def missing_ids(session: Session, feedback_ids: list[str], vehicle_code: str = "G60") -> list[str]:
    found = set(session.scalars(
        select(Feedback.original_feedback_id)
        .where(Feedback.vehicle_code == vehicle_code, Feedback.original_feedback_id.in_(feedback_ids))
    ))
    return [i for i in feedback_ids if i not in found]


def feedback_for_category(
    session: Session, vfc_level2: str | None, flt: FeedbackFilter | None = None, limit: int = 20, offset: int = 0
) -> list[FeedbackRecord]:
    """Verbatim comments classified under a VFC level-2 category (``None`` =
    rows without classification), ordered by feedback ID."""
    flt = (flt or FeedbackFilter()).model_copy(update={"vfc_level2": [vfc_level2]})
    ids = session.scalars(
        _joined(Feedback.original_feedback_id).where(*filter_conditions(flt)).distinct()
        .order_by(Feedback.original_feedback_id).limit(limit).offset(offset)
    ).all()
    return get_feedback_by_ids(session, list(ids), flt.vehicle_code)


def search_feedback_text(
    session: Session, phrase: str, flt: FeedbackFilter | None = None, limit: int = 20
) -> tuple[int, list[FeedbackRecord]]:
    """Literal case-insensitive substring search in the original comment text.
    Returns (total matches, first ``limit`` records)."""
    if not phrase.strip():
        raise ValueError("phrase must not be empty")
    flt = flt or FeedbackFilter()
    base = _joined(Feedback.original_feedback_id).where(*filter_conditions(flt), _contains(Feedback.customer_feedback, phrase))
    total = session.scalar(select(func.count()).select_from(base.distinct().subquery()))
    ids = session.scalars(base.distinct().order_by(Feedback.original_feedback_id).limit(limit)).all()
    return total or 0, get_feedback_by_ids(session, list(ids), flt.vehicle_code)


# ---------------------------------------------------------------------------
# Public evidence (separate from BMW feedback)
# ---------------------------------------------------------------------------


def public_owner_feedback(
    session: Session, theme_contains: str | None = None, sentiment: str | None = None,
    text_contains: str | None = None, public_id: str | None = None,
) -> list[PublicFeedbackRecord]:
    stmt = select(PublicFeedback).order_by(PublicFeedback.original_public_id)
    if theme_contains:
        stmt = stmt.where(_contains(PublicFeedback.theme, theme_contains))
    if sentiment:
        stmt = stmt.where(PublicFeedback.sentiment == sentiment)
    if text_contains:
        stmt = stmt.where(_contains(PublicFeedback.paraphrased_feedback, text_contains))
    if public_id:
        stmt = stmt.where(PublicFeedback.original_public_id == public_id)
    names = _names(session)
    return [
        PublicFeedbackRecord(
            original_public_id=p.original_public_id, model=p.model, model_year=p.model_year, review_date=p.review_date,
            source_name=p.source_name, source_type=p.source_type, market=p.market,
            sentiment_as_labelled_in_source=p.sentiment, theme=p.theme, paraphrased_feedback=p.paraphrased_feedback,
            product_implication=p.product_implication, source_url=p.source_url, evidence_note=p.evidence_note,
            related_public_id=p.derived_related_public_id, relation_type=p.derived_relation_type, source=_ref(p, names),
        )
        for p in session.scalars(stmt)
    ]


def competitor_evidence(
    session: Session, brand: str | None = None, theme_contains: str | None = None,
    text_contains: str | None = None, bmw_only: bool | None = None,
) -> list[CompetitorEvidenceRecord]:
    stmt = select(CompetitorEvidence).order_by(CompetitorEvidence.source_row)
    if brand:
        stmt = stmt.where(func.lower(CompetitorEvidence.brand) == brand.lower())
    if theme_contains:
        stmt = stmt.where(_contains(CompetitorEvidence.theme, theme_contains))
    if text_contains:
        stmt = stmt.where(_contains(CompetitorEvidence.extracted_insight, text_contains))
    if bmw_only is not None:
        stmt = stmt.where(CompetitorEvidence.derived_is_bmw.is_(bmw_only))
    names = _names(session)
    return [
        CompetitorEvidenceRecord(
            brand=c.brand, model=c.model, is_bmw=c.derived_is_bmw, source_type=c.source_type,
            date_or_period=c.date_or_period, market=c.market, theme=c.theme,
            sentiment_as_labelled_in_source=c.sentiment, extracted_insight=c.extracted_insight,
            evidence_limitation=c.evidence_limitation, source_url=c.source_url, source=_ref(c, names),
        )
        for c in session.scalars(stmt)
    ]
