"""Read models for the Streamlit dashboard (Part 3).

Thin composition over the existing services - no Streamlit import here, so
everything is unit-testable. All statistics are computed from SQLite at call
time via SQLAlchemy (bound parameters); comments are always counted as
COUNT(DISTINCT feedback record) and never as classification rows.

Filter semantics: market / engine type apply to the comment; need category,
VFC category and Feedback Type apply to classification rows, and must all hold
on the same row ("a comment that has at least one classification row with
need X, VFC Y and type Z").
"""
from __future__ import annotations

import json

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.ai_models import (
    AIFeedbackExtraction,
    CrossSourceEvidenceLink,
    CustomerNeedCategory,
    CustomerNeedCluster,
    VfcNeedMapping,
)
from core.models import CustomerStudy, Feedback, FeedbackClassification, IngestionAudit, PublicFeedback, SourceDocument
from services.cross_source_mapping import evidence_for_need, vfc_coalesced
from services.customer_insights import StudyValue, sales_context
from services.evidence_retrieval import competitor_evidence, public_owner_feedback
from services.feedback_analysis import (
    CategoryCount,
    DimensionCount,
    FeedbackFilter,
    FeedbackTypeCount,
    SourceReference,
    _joined,
    count_feedback,
    distribution_by,
    feedback_type_distribution,
    filter_conditions,
    vfc_category_counts,
)
from services.need_taxonomy import MISSING_VFC

VEHICLE = "G60"
MISSING_TYPE_LABEL = "(not provided)"


class DashboardFilter(BaseModel):
    countries: list[str] | None = None
    engine_types: list[str] | None = None
    need_codes: list[str] | None = None
    vfc_names: list[str] | None = None  # MISSING_VFC selects empty VFC cells
    feedback_types: list[str | None] | None = None  # None selects missing Feedback Type


def to_feedback_filter(session: Session, f: DashboardFilter) -> FeedbackFilter:
    """Translate need categories into their mapped VFC names and intersect with
    an explicit VFC selection. An empty intersection matches nothing."""
    vfc: set[str] | None = None
    if f.need_codes:
        vfc = set(session.scalars(select(VfcNeedMapping.vfc_level2_name).where(VfcNeedMapping.need_code.in_(f.need_codes))))
    if f.vfc_names:
        vfc = set(f.vfc_names) if vfc is None else vfc & set(f.vfc_names)
    return FeedbackFilter(
        vehicle_code=VEHICLE,
        countries=f.countries or None,
        engine_types=f.engine_types or None,
        feedback_types=f.feedback_types or None,
        vfc_level2=None if vfc is None else sorted((None if v == MISSING_VFC else v) for v in vfc),
    )


# ---------------------------------------------------------------------------
# Page 1 - Customer Intelligence
# ---------------------------------------------------------------------------


class NeedCount(BaseModel):
    need_code: str
    need_name: str
    unique_feedback_count: int
    classification_count: int


class Overview(BaseModel):
    unique_feedback: int
    classification_rows: int
    need_categories_total: int
    need_categories_with_feedback: int
    vfc_clusters_total: int
    vfc_clusters_with_feedback: int
    by_need: list[NeedCount]
    top_vfc: list[CategoryCount]
    by_market: list[DimensionCount]
    by_engine: list[DimensionCount]
    feedback_types: list[FeedbackTypeCount]


def need_category_counts(session: Session, flt: FeedbackFilter) -> list[NeedCount]:
    """Distinct comments per need category (a comment can count in several
    categories when its VFC labels map to different needs)."""
    names = dict(session.execute(select(CustomerNeedCategory.need_code, CustomerNeedCategory.name)).all())
    order = dict(session.execute(select(CustomerNeedCategory.need_code, CustomerNeedCategory.sort_order)).all())
    rows = session.execute(
        _joined(VfcNeedMapping.need_code, func.count(func.distinct(Feedback.record_id)), func.count())
        .join(VfcNeedMapping, VfcNeedMapping.vfc_level2_name == vfc_coalesced())
        .where(*filter_conditions(flt)).group_by(VfcNeedMapping.need_code)
    ).all()
    out = [NeedCount(need_code=c, need_name=names.get(c, c), unique_feedback_count=i, classification_count=r)
           for c, i, r in rows]
    return sorted(out, key=lambda n: (-n.unique_feedback_count, order.get(n.need_code, 99)))


def overview(session: Session, f: DashboardFilter, top_n: int = 15) -> Overview:
    flt = to_feedback_filter(session, f)
    counts = count_feedback(session, flt)
    by_need = need_category_counts(session, flt)
    vfc = vfc_category_counts(session, flt, "level2")
    return Overview(
        unique_feedback=counts.feedback_ids,
        classification_rows=counts.classification_rows,
        need_categories_total=session.scalar(select(func.count()).select_from(CustomerNeedCategory)) or 0,
        need_categories_with_feedback=sum(1 for n in by_need if n.unique_feedback_count),
        vfc_clusters_total=session.scalar(select(func.count()).select_from(CustomerNeedCluster).where(
            CustomerNeedCluster.vehicle_code == VEHICLE, CustomerNeedCluster.cluster_level == "vfc_category")) or 0,
        vfc_clusters_with_feedback=sum(1 for c in vfc if c.feedback_ids),
        by_need=by_need,
        # "Topics" exclude rows without a usable class (empty cell / BMW no_class_found).
        top_vfc=[c for c in vfc if c.category not in (None, "no_class_found")][:top_n],
        by_market=distribution_by(session, "country", flt),
        by_engine=distribution_by(session, "engine_type", flt),
        feedback_types=feedback_type_distribution(session, flt),
    )


def filter_options(session: Session) -> dict:
    needs = list(session.execute(select(CustomerNeedCategory.need_code, CustomerNeedCategory.name)
                                 .order_by(CustomerNeedCategory.sort_order)).all())
    vfc = sorted(session.scalars(select(vfc_coalesced()).where(FeedbackClassification.vehicle_code == VEHICLE).distinct()))
    return {
        "countries": sorted(session.scalars(select(Feedback.country).where(Feedback.vehicle_code == VEHICLE).distinct())),
        "engine_types": sorted(session.scalars(select(Feedback.engine_type).where(Feedback.vehicle_code == VEHICLE).distinct())),
        "needs": [{"code": c, "name": n} for c, n in needs],
        "vfc_names": vfc,
        "feedback_types": sorted(
            {t for t in session.scalars(select(FeedbackClassification.feedback_type)
                                        .where(FeedbackClassification.vehicle_code == VEHICLE).distinct())},
            key=lambda t: (t is None, t or "")),
    }


# ---------------------------------------------------------------------------
# Page 2 - Evidence Explorer
# ---------------------------------------------------------------------------


class ClusterOption(BaseModel):
    cluster_id: str
    cluster_level: str
    need_code: str
    label: str
    vfc_level2_name: str | None
    unique_feedback_count: int
    requires_review: bool


def cluster_options(session: Session) -> list[ClusterOption]:
    rows = session.scalars(select(CustomerNeedCluster).where(CustomerNeedCluster.vehicle_code == VEHICLE)
                           .order_by(CustomerNeedCluster.need_code, CustomerNeedCluster.cluster_level,
                                     CustomerNeedCluster.unique_feedback_count.desc())).all()
    return [ClusterOption(cluster_id=c.cluster_id, cluster_level=c.cluster_level, need_code=c.need_code, label=c.label,
                          vfc_level2_name=c.vfc_level2_name, unique_feedback_count=c.unique_feedback_count,
                          requires_review=c.requires_review) for c in rows]


def vfc_mapping_status(session: Session) -> dict[str, tuple[str, str, str | None]]:
    """VFC name -> (need_code, mapping_status, rationale)."""
    return {m.vfc_level2_name: (m.need_code, m.mapping_status, m.rationale) for m in session.scalars(select(VfcNeedMapping))}


class EvidenceItem(BaseModel):
    tier: str  # SUPPORTING | CONTEXTUAL
    kind: str  # public_owner_feedback | public_bmw_observation | customer_study | competitor_observation
    key: str
    title: str
    detail: str | None
    source_url: str | None
    requires_review: bool
    mapping_note: str | None
    source: SourceReference
    study_values: list[StudyValue] = []


def related_evidence(session: Session, need_code: str) -> list[EvidenceItem]:
    """Supporting and contextual evidence for a need, resolved to source records.
    Direct evidence (internal feedback) is listed separately by the caller."""
    ev = evidence_for_need(session, need_code, VEHICLE)
    competitor = {c.source.source_row: c for c in competitor_evidence(session)}
    items: list[EvidenceItem] = []
    for link in ev.supporting + ev.contextual:
        tier = link.evidence_tier.upper()
        if link.source_table == "public_feedback":
            p = public_owner_feedback(session, public_id=link.source_key)[0]
            items.append(EvidenceItem(
                tier=tier, kind="public_owner_feedback", key=p.original_public_id,
                title=f"{p.original_public_id} · {p.model} · {p.market} · {p.theme}",
                detail=f"{p.paraphrased_feedback} (sentiment as labelled in source: {p.sentiment_as_labelled_in_source}; "
                       f"note: {p.evidence_note})",
                source_url=p.source_url, requires_review=link.requires_review, mapping_note=link.note, source=p.source))
        elif link.source_table == "competitor_evidence":
            c = competitor[int(link.source_key.removeprefix("COMP-R"))]
            items.append(EvidenceItem(
                tier=tier, kind="public_bmw_observation" if c.is_bmw else "competitor_observation", key=link.source_key,
                title=f"{c.brand} {c.model} · {c.market} · {c.theme}",
                detail=f"{c.extracted_insight} (limitation: {c.evidence_limitation}; sentiment as labelled: "
                       f"{c.sentiment_as_labelled_in_source})",
                source_url=c.source_url, requires_review=link.requires_review, mapping_note=link.note, source=c.source))
        elif link.source_table == "customer_studies":
            cells = study_cells(session, link.record_ids)
            first = cells[0].source if cells else None
            items.append(EvidenceItem(
                tier=tier, kind="customer_study", key=link.source_key, title=link.source_label, detail=None,
                source_url=None, requires_review=link.requires_review, mapping_note=link.note,
                source=first or SourceReference(record_id="", source_file="", source_sheet=None, source_row=None,
                                                data_origin="internal_bmw"),
                study_values=cells))
    return items


def study_cells(session: Session, record_ids: list[str]) -> list[StudyValue]:
    rows = session.scalars(select(CustomerStudy).where(CustomerStudy.record_id.in_(record_ids))
                           .order_by(CustomerStudy.source_row, CustomerStudy.source_column)).all()
    return [StudyValue(study_name=r.study_name, survey_market=r.survey_market, vehicle_label=r.vehicle_label,
                       vehicle_code=r.vehicle_code, attribute=r.attribute, measure=r.measure, value_type=r.value_type,
                       value=r.value, source=SourceReference(record_id=r.record_id, source_file=r.source_file,
                                                             source_sheet=r.source_sheet, source_row=r.source_row,
                                                             data_origin="internal_bmw"))
            for r in rows]


# ---------------------------------------------------------------------------
# Page 4 - Data Quality
# ---------------------------------------------------------------------------


def source_documents(session: Session) -> list[dict]:
    return [{"file": d.file_name, "dataset_type": d.dataset_type, "origin": d.data_origin, "type": d.file_type,
             "source_of_truth": d.is_source_of_truth, "status": d.ingestion_status,
             "original_names": ", ".join(json.loads(d.original_file_names or "[]")), "notes": d.ingestion_notes or ""}
            for d in session.scalars(select(SourceDocument).order_by(SourceDocument.data_origin, SourceDocument.file_name))]


def audit_rows(session: Session) -> list[dict]:
    """Count checks recorded at the last ingestion (raw-file recount vs database)."""
    return [{"file": a.source_file, "sheet": a.source_sheet, "table": a.target_table, "expected": a.expected_count,
             "imported": a.imported_count, "basis": a.count_basis, "status": a.status, "checked_at": a.checked_at}
            for a in session.scalars(select(IngestionAudit).order_by(IngestionAudit.audit_id))]


def review_mappings(session: Session) -> list[dict]:
    """VFC -> need mappings still marked review_required, with G60 comment counts."""
    counts = dict(session.execute(
        select(vfc_coalesced(), func.count(func.distinct(FeedbackClassification.feedback_record_id)))
        .where(FeedbackClassification.vehicle_code == VEHICLE).group_by(vfc_coalesced())).all())
    return sorted(({"vfc_level2_name": m.vfc_level2_name, "need_code": m.need_code, "reason": m.rationale,
                    "g60_comments": counts.get(m.vfc_level2_name, 0)}
                   for m in session.scalars(select(VfcNeedMapping).where(VfcNeedMapping.mapping_status == "review_required"))),
                  key=lambda r: -r["g60_comments"])


def review_links(session: Session) -> list[dict]:
    return [{"need_code": l.need_code, "tier": l.evidence_tier, "source_table": l.source_table, "source_key": l.source_key,
             "label": l.source_label, "note": l.note}
            for l in session.scalars(select(CrossSourceEvidenceLink).where(
                CrossSourceEvidenceLink.requires_review.is_(True), CrossSourceEvidenceLink.evidence_tier != "direct")
                .order_by(CrossSourceEvidenceLink.need_code, CrossSourceEvidenceLink.link_id))]


class Limitation(BaseModel):
    title: str
    evidence: str  # computed from the database


def known_limitations(session: Session) -> list[Limitation]:
    """The eight known limitations, each with a figure computed live."""
    fc = FeedbackClassification
    likes = select(fc.feedback_record_id).where(fc.vehicle_code == VEHICLE, fc.feedback_type == "Likes")
    both = session.scalar(select(func.count(func.distinct(fc.feedback_record_id))).where(
        fc.vehicle_code == VEHICLE, fc.feedback_type.in_(["Defect", "Difficult to Use"]), fc.feedback_record_id.in_(likes))) or 0
    missing_type = session.scalar(select(func.count()).select_from(fc).where(fc.vehicle_code == VEHICLE, fc.feedback_type.is_(None))) or 0
    multi = session.scalar(select(func.count()).select_from(Feedback).where(
        Feedback.vehicle_code == VEHICLE, Feedback.classification_row_count > 1)) or 0
    total = session.scalar(select(func.count()).select_from(Feedback).where(Feedback.vehicle_code == VEHICLE)) or 0
    markets = distribution_by(session, "country", FeedbackFilter(vehicle_code=VEHICLE))
    top = markets[0] if markets else None
    public_n = session.scalar(select(func.count()).select_from(PublicFeedback)) or 0
    sales = sales_context(session, VEHICLE)
    review_n = session.scalar(select(func.count()).select_from(CrossSourceEvidenceLink).where(
        CrossSourceEvidenceLink.requires_review.is_(True))) or 0
    review_direct = session.scalar(select(func.count()).select_from(CrossSourceEvidenceLink).where(
        CrossSourceEvidenceLink.requires_review.is_(True), CrossSourceEvidenceLink.evidence_tier == "direct")) or 0
    links_n = session.scalar(select(func.count()).select_from(CrossSourceEvidenceLink)) or 0
    extractions = session.scalar(select(func.count()).select_from(AIFeedbackExtraction).where(
        AIFeedbackExtraction.status == "succeeded", AIFeedbackExtraction.vehicle_code == VEHICLE)) or 0
    return [
        Limitation(title="Feedback Type is not reliable sentiment",
                   evidence=f"{both} G60 comments carry both a 'Likes' row and a 'Defect'/'Difficult to Use' row; "
                            f"{missing_type} classification rows have no Feedback Type. No sentiment shares are shown."),
        Limitation(title="Some comments have multiple VFC classifications",
                   evidence=f"{multi} of {total} G60 comments have more than one classification row; "
                            "comment counts use COUNT(DISTINCT feedback)."),
        Limitation(title="Most feedback originates from the US",
                   evidence=(f"{top.value}: {top.feedback_ids} of {total} comments ({top.share_of_feedback_ids:.0%})."
                             if top else "No market data.")),
        Limitation(title="Public feedback is small and curated",
                   evidence=f"{public_n} paraphrased public observations; several rows can come from one review; "
                            "not a representative sample."),
        Limitation(title="Survey sample sizes are unavailable",
                   evidence="The customer-study workbook provides means (CN/EU) and answer proportions (US) only; "
                            "no respondent counts are stored or inferred."),
        Limitation(title="G60/G68 sales are combined",
                   evidence=" ".join(sales.warnings[:1]) or "No combined sales rows found."),
        Limitation(title="Some cross-source mappings require review",
                   evidence=f"{review_n} of {links_n} evidence links are marked requires_review: {review_direct} "
                            f"VFC-category → need mappings (direct) and {review_n - review_direct} supporting/contextual "
                            "links to public feedback, studies and competitor rows."),
        Limitation(title="No real LLM extraction has run yet" if extractions == 0 else "LLM extraction coverage is partial",
                   evidence=(f"{extractions} AI extractions stored. All insights are deterministic templates built from "
                             "original VFC classifications." if extractions == 0 else
                             f"{extractions} of {total} G60 comments have stored AI extractions.")),
    ]
