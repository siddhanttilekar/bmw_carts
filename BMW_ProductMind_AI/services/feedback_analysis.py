"""Deterministic analysis of internal BMW customer feedback.

Every function is scoped to exactly one vehicle (default ``G60``), so F70/G70
feedback can never leak into G60 analysis. Results are Pydantic models
(``.model_dump()`` gives plain dicts for later visualisation).

Counting units
--------------
* ``feedback_ids``         distinct original feedback IDs (customer comments)
* ``classification_rows``  source worksheet rows (one per ID x VFC classification)
A comment with three VFC labels is 1 feedback ID and 3 classification rows.

Feedback Type is NOT sentiment
------------------------------
``Feedback Type`` (Likes / Difficult to Use / Defect / Wants / missing) is a
BMW label on each *classification row*, not on the comment: the same comment
can carry a "Likes" row and a "Defect" row, some channels never set it
(Source D) and some never use "Likes"/"Wants" (Source C), and ~15% of G60 rows
have no value. Values are therefore reported exactly as supplied and no
positive/negative sentiment is derived from them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import false, func, or_, select
from sqlalchemy.orm import Session

from core.models import Feedback, FeedbackClassification, SourceDocument

FEEDBACK_TYPE_NOTE = (
    "Feedback Type is BMW's label per classification row (Likes, Difficult to Use, Defect, Wants). "
    "It is not comment-level sentiment: one comment can have both 'Likes' and 'Defect' rows, channels "
    "use the labels differently, and missing values are kept as missing."
)
MISSING_LABEL = "(not provided)"

VfcLevel = Literal["level2", "level3", "level2_3"]
Dimension = Literal["country", "engine_type", "feedback_source"]

_VFC_COLUMNS = {
    "level2": FeedbackClassification.vfc_level2_name,
    "level3": FeedbackClassification.vfc_level3_name,
    "level2_3": FeedbackClassification.vfc_level2_3_name,
}
_DIMENSION_COLUMNS = {
    "country": Feedback.country,
    "engine_type": Feedback.engine_type,
    "feedback_source": Feedback.feedback_source,
}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class FeedbackFilter(BaseModel):
    """Comment-level filters (country, engine type, source) and
    classification-level filters (feedback type, VFC level 2). A comment
    matches if at least one of its classification rows matches. ``None`` inside
    ``feedback_types`` / ``vfc_level2`` selects rows where that value is missing."""

    vehicle_code: str = "G60"
    countries: list[str] | None = None
    engine_types: list[str] | None = None
    feedback_sources: list[str] | None = None
    feedback_types: list[str | None] | None = None
    vfc_level2: list[str | None] | None = None


class SourceReference(BaseModel):
    record_id: str
    source_file: str
    source_sheet: str | None
    source_row: int | None
    source_page: int | None = None
    data_origin: str
    original_file_names: str | None = None


class Classification(BaseModel):
    feedback_type: str | None
    vfc_level2_3_name: str | None
    vfc_level2_name: str | None
    vfc_level3_name: str | None
    classification_status: str
    source: SourceReference


class FeedbackRecord(BaseModel):
    original_feedback_id: str
    vehicle_code: str
    customer_feedback: str
    country: str | None
    engine_type: str | None
    feedback_source: str | None
    brand: str | None
    sales_description: str | None
    classifications: list[Classification]
    source: SourceReference


class FeedbackCount(BaseModel):
    vehicle_code: str
    feedback_ids: int
    classification_rows: int
    filter: FeedbackFilter


class CategoryCount(BaseModel):
    category: str | None
    label: str
    classification_rows: int
    feedback_ids: int
    share_of_feedback_ids: float = Field(description="feedback_ids / distinct IDs in the filtered set")


class FeedbackTypeCount(BaseModel):
    feedback_type: str | None
    label: str
    classification_rows: int
    feedback_ids: int
    share_of_rows: float


class DimensionCount(BaseModel):
    value: str | None
    feedback_ids: int
    classification_rows: int
    share_of_feedback_ids: float


class MatrixCell(BaseModel):
    category: str | None
    feedback_type: str | None
    classification_rows: int
    feedback_ids: int


class FeedbackPage(BaseModel):
    total_feedback_ids: int
    limit: int
    offset: int
    items: list[FeedbackRecord]


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def _in_or_null(column, values: list[str | None]):
    present = [v for v in values if v is not None]
    clauses = [column.in_(present)] if present else []
    if any(v is None for v in values):
        clauses.append(column.is_(None))
    return or_(*clauses) if clauses else false()


def filter_conditions(flt: FeedbackFilter) -> list:
    conds = [Feedback.vehicle_code == flt.vehicle_code, FeedbackClassification.vehicle_code == flt.vehicle_code]
    if flt.countries is not None:
        conds.append(Feedback.country.in_(flt.countries))
    if flt.engine_types is not None:
        conds.append(Feedback.engine_type.in_(flt.engine_types))
    if flt.feedback_sources is not None:
        conds.append(Feedback.feedback_source.in_(flt.feedback_sources))
    if flt.feedback_types is not None:
        conds.append(_in_or_null(FeedbackClassification.feedback_type, flt.feedback_types))
    if flt.vfc_level2 is not None:
        conds.append(_in_or_null(FeedbackClassification.vfc_level2_name, flt.vfc_level2))
    return conds


def _joined(*columns):
    return select(*columns).select_from(FeedbackClassification).join(
        Feedback, Feedback.record_id == FeedbackClassification.feedback_record_id
    )


def _require_vehicle(session: Session, vehicle_code: str) -> None:
    if not session.scalar(select(func.count()).select_from(Feedback).where(Feedback.vehicle_code == vehicle_code)):
        raise ValueError(f"No internal feedback for vehicle {vehicle_code!r}")


def _share(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


# ---------------------------------------------------------------------------
# Counting and grouping
# ---------------------------------------------------------------------------


def count_feedback(session: Session, flt: FeedbackFilter | None = None) -> FeedbackCount:
    flt = flt or FeedbackFilter()
    _require_vehicle(session, flt.vehicle_code)
    ids, rows = session.execute(
        _joined(func.count(func.distinct(Feedback.record_id)), func.count()).where(*filter_conditions(flt))
    ).one()
    return FeedbackCount(vehicle_code=flt.vehicle_code, feedback_ids=ids, classification_rows=rows, filter=flt)


def vfc_category_counts(
    session: Session, flt: FeedbackFilter | None = None, level: VfcLevel = "level2"
) -> list[CategoryCount]:
    """Counts per original BMW VFC category, sorted by feedback IDs (desc) then name.
    Missing classifications form their own ``category=None`` group."""
    flt = flt or FeedbackFilter()
    total = count_feedback(session, flt).feedback_ids
    column = _VFC_COLUMNS[level]
    rows = session.execute(
        _joined(column, func.count(), func.count(func.distinct(Feedback.record_id)))
        .where(*filter_conditions(flt)).group_by(column)
    ).all()
    result = [
        CategoryCount(category=c, label=c if c is not None else "(missing VFC classification)",
                      classification_rows=n_rows, feedback_ids=n_ids, share_of_feedback_ids=_share(n_ids, total))
        for c, n_rows, n_ids in rows
    ]
    return sorted(result, key=lambda r: (-r.feedback_ids, r.label))


def feedback_type_distribution(session: Session, flt: FeedbackFilter | None = None) -> list[FeedbackTypeCount]:
    flt = flt or FeedbackFilter()
    total_rows = count_feedback(session, flt).classification_rows
    col = FeedbackClassification.feedback_type
    rows = session.execute(
        _joined(col, func.count(), func.count(func.distinct(Feedback.record_id)))
        .where(*filter_conditions(flt)).group_by(col)
    ).all()
    result = [
        FeedbackTypeCount(feedback_type=t, label=t if t is not None else MISSING_LABEL, classification_rows=n_rows,
                          feedback_ids=n_ids, share_of_rows=_share(n_rows, total_rows))
        for t, n_rows, n_ids in rows
    ]
    return sorted(result, key=lambda r: (-r.classification_rows, r.label))


def category_feedback_type_matrix(
    session: Session, flt: FeedbackFilter | None = None, level: VfcLevel = "level2"
) -> list[MatrixCell]:
    flt = flt or FeedbackFilter()
    _require_vehicle(session, flt.vehicle_code)
    column, ftype = _VFC_COLUMNS[level], FeedbackClassification.feedback_type
    rows = session.execute(
        _joined(column, ftype, func.count(), func.count(func.distinct(Feedback.record_id)))
        .where(*filter_conditions(flt)).group_by(column, ftype)
    ).all()
    cells = [MatrixCell(category=c, feedback_type=t, classification_rows=r, feedback_ids=i) for c, t, r, i in rows]
    return sorted(cells, key=lambda m: (m.category or "", m.feedback_type or ""))


def distribution_by(session: Session, dimension: Dimension, flt: FeedbackFilter | None = None) -> list[DimensionCount]:
    """Distribution of comments by country, engine type or feedback channel."""
    flt = flt or FeedbackFilter()
    total = count_feedback(session, flt).feedback_ids
    column = _DIMENSION_COLUMNS[dimension]
    rows = session.execute(
        _joined(column, func.count(func.distinct(Feedback.record_id)), func.count())
        .where(*filter_conditions(flt)).group_by(column)
    ).all()
    result = [DimensionCount(value=v, feedback_ids=i, classification_rows=r, share_of_feedback_ids=_share(i, total))
              for v, i, r in rows]
    return sorted(result, key=lambda d: (-d.feedback_ids, d.value or ""))


def available_values(session: Session, vehicle_code: str = "G60") -> dict[str, list]:
    """Distinct filter values present in the data (for building filter UIs)."""
    def distinct(column, model):
        return sorted(
            {v for (v,) in session.execute(select(column).where(model.vehicle_code == vehicle_code).distinct())},
            key=lambda v: (v is None, v or ""),
        )
    return {
        "countries": distinct(Feedback.country, Feedback),
        "engine_types": distinct(Feedback.engine_type, Feedback),
        "feedback_sources": distinct(Feedback.feedback_source, Feedback),
        "feedback_types": distinct(FeedbackClassification.feedback_type, FeedbackClassification),
        "vfc_level2": distinct(FeedbackClassification.vfc_level2_name, FeedbackClassification),
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _original_names(session: Session) -> dict[str, str | None]:
    return dict(session.execute(select(SourceDocument.document_id, SourceDocument.original_file_names)).all())


def build_feedback_records(session: Session, feedback_rows: list[Feedback]) -> list[FeedbackRecord]:
    if not feedback_rows:
        return []
    names = _original_names(session)
    by_parent: dict[str, list[FeedbackClassification]] = {}
    ids = [f.record_id for f in feedback_rows]
    for chunk_start in range(0, len(ids), 500):
        for c in session.scalars(
            select(FeedbackClassification)
            .where(FeedbackClassification.feedback_record_id.in_(ids[chunk_start:chunk_start + 500]))
            .order_by(FeedbackClassification.source_row)
        ):
            by_parent.setdefault(c.feedback_record_id, []).append(c)

    def ref(obj, origin: str) -> SourceReference:
        return SourceReference(record_id=obj.record_id, source_file=obj.source_file, source_sheet=obj.source_sheet,
                               source_row=obj.source_row, source_page=obj.source_page, data_origin=origin,
                               original_file_names=names.get(obj.source_document_id))

    return [
        FeedbackRecord(
            original_feedback_id=f.original_feedback_id, vehicle_code=f.vehicle_code, customer_feedback=f.customer_feedback,
            country=f.country, engine_type=f.engine_type, feedback_source=f.feedback_source, brand=f.brand,
            sales_description=f.sales_description, source=ref(f, f.data_origin),
            classifications=[
                Classification(feedback_type=c.feedback_type, vfc_level2_3_name=c.vfc_level2_3_name,
                               vfc_level2_name=c.vfc_level2_name, vfc_level3_name=c.vfc_level3_name,
                               classification_status=c.derived_classification_status, source=ref(c, f.data_origin))
                for c in by_parent.get(f.record_id, [])
            ],
        )
        for f in feedback_rows
    ]


def filter_feedback(
    session: Session, flt: FeedbackFilter | None = None, limit: int = 50, offset: int = 0
) -> FeedbackPage:
    """Matching comments ordered by original feedback ID. Each record carries
    *all* of its original classification rows, not only the matching ones."""
    flt = flt or FeedbackFilter()
    total = count_feedback(session, flt).feedback_ids
    matching = _joined(Feedback.record_id).where(*filter_conditions(flt)).distinct().subquery()
    rows = session.scalars(
        select(Feedback).where(Feedback.record_id.in_(select(matching.c.record_id)))
        .order_by(Feedback.original_feedback_id).limit(limit).offset(offset)
    ).all()
    return FeedbackPage(total_feedback_ids=total, limit=limit, offset=offset, items=build_feedback_records(session, rows))


def matching_feedback_ids(session: Session, flt: FeedbackFilter | None = None) -> list[str]:
    flt = flt or FeedbackFilter()
    _require_vehicle(session, flt.vehicle_code)
    return sorted(
        session.scalars(_joined(Feedback.original_feedback_id).where(*filter_conditions(flt)).distinct()).all()
    )


__all__ = [
    "FEEDBACK_TYPE_NOTE", "FeedbackFilter", "FeedbackRecord", "SourceReference", "count_feedback",
    "vfc_category_counts", "feedback_type_distribution", "category_feedback_type_matrix", "distribution_by",
    "available_values", "filter_feedback", "matching_feedback_ids", "build_feedback_records",
]
