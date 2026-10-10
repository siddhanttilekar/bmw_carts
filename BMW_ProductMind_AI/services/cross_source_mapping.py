"""Cross-source evidence mapping for normalised customer needs.

Evidence tiers
--------------
* DIRECT      original internal BMW G60 feedback (via the VFC -> need mapping)
* SUPPORTING  public BMW owner observations (incl. the BMW rows of the
              competitor-evidence file) and BMW customer-study attributes
* CONTEXTUAL  competitor (non-BMW) observations - never proof of BMW
              customer dissatisfaction

All links come from the curated tables in ``services.need_taxonomy``; every
curated key is resolved to real database records and an unresolved key is an
error. Uncertain mappings carry ``requires_review=True``.
"""
from __future__ import annotations

import json

from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from core.ai_models import CrossSourceEvidenceLink, VfcNeedMapping
from core.models import CompetitorEvidence, CustomerStudy, Feedback, FeedbackClassification, PublicFeedback
from services.need_taxonomy import (
    COMPETITOR_LINKS,
    MAPPING_VERSION,
    MISSING_VFC,
    NEED_CODES,
    PUBLIC_FEEDBACK_LINKS,
    STUDY_LINKS,
)

PUBLIC_CAVEAT = "Curated public observation; not a representative sample and rows may share one review."
STUDY_CAVEAT = "Survey perception item; related to but not identical with VFC feedback classes; no sample size given."
CONTEXT_CAVEAT = "Competitor observation: market context only, not evidence about BMW customers."


def competitor_key(row: int) -> str:
    return f"COMP-R{row}"


def study_key(study: str, attribute: str) -> str:
    return f"{study}::{attribute}"


def vfc_coalesced():
    return func.coalesce(FeedbackClassification.vfc_level2_name, MISSING_VFC)


def build_cross_source_links(session: Session, vehicle_code: str = "G60") -> dict[str, int]:
    """Rebuild cross_source_evidence_links from the curated mappings."""
    session.execute(delete(CrossSourceEvidenceLink))
    session.flush()
    rows: list[CrossSourceEvidenceLink] = []

    def add(**kw):
        rows.append(CrossSourceEvidenceLink(mapping_version=MAPPING_VERSION, **kw))

    # DIRECT: VFC categories that occur in this vehicle's feedback.
    present = set(session.scalars(
        select(vfc_coalesced()).where(FeedbackClassification.vehicle_code == vehicle_code).distinct()
    ))
    mappings = session.scalars(select(VfcNeedMapping)).all()
    if not mappings:
        raise RuntimeError("VFC mappings missing - run sync_taxonomy first")
    for m in mappings:
        if m.vfc_level2_name not in present:
            continue
        add(need_code=m.need_code, evidence_tier="direct", source_table="feedback_classifications",
            source_key=m.vfc_level2_name, source_record_ids_json=None, source_label=f"VFC: {m.vfc_level2_name}",
            data_origin="internal_bmw", mapping_basis="vfc_need_mapping",
            requires_review=m.mapping_status == "review_required", note=m.rationale)

    # SUPPORTING: public owner feedback.
    public = {p.original_public_id: p for p in session.scalars(select(PublicFeedback))}
    for need, links in PUBLIC_FEEDBACK_LINKS.items():
        for pid, review, note in links:
            p = public.get(pid)
            if p is None:
                raise KeyError(f"Curated public ID {pid} not found in public_feedback")
            add(need_code=need, evidence_tier="supporting", source_table="public_feedback", source_key=pid,
                source_record_ids_json=json.dumps([p.record_id]),
                source_label=f"{pid} {p.model} ({p.market}): {p.theme} [sentiment as labelled: {p.sentiment}]",
                data_origin="public_external", mapping_basis="curated_public_theme", requires_review=review,
                note="; ".join(x for x in (note, PUBLIC_CAVEAT) if x))

    # SUPPORTING (BMW rows) / CONTEXTUAL (other brands): competitor-evidence file.
    comp = {c.source_row: c for c in session.scalars(select(CompetitorEvidence))}
    for need, links in COMPETITOR_LINKS.items():
        for row, review, note in links:
            c = comp.get(row)
            if c is None:
                raise KeyError(f"Curated competitor row {row} not found")
            tier = "supporting" if c.derived_is_bmw else "contextual"
            caveat = PUBLIC_CAVEAT if c.derived_is_bmw else CONTEXT_CAVEAT
            add(need_code=need, evidence_tier=tier, source_table="competitor_evidence", source_key=competitor_key(row),
                source_record_ids_json=json.dumps([c.record_id]),
                source_label=f"{c.brand} {c.model} ({c.market}): {c.theme} [sentiment as labelled: {c.sentiment}]",
                data_origin="public_external", mapping_basis="curated_competitor_theme", requires_review=review,
                note="; ".join(x for x in (note, caveat) if x))

    # SUPPORTING: customer-study attributes (cells for this vehicle only).
    for need, links in STUDY_LINKS.items():
        for study, attribute, review in links:
            ids = list(session.scalars(select(CustomerStudy.record_id).where(
                CustomerStudy.study_name == study, CustomerStudy.attribute == attribute,
                CustomerStudy.vehicle_code == vehicle_code).order_by(CustomerStudy.source_row, CustomerStudy.source_column)))
            if not ids:
                raise KeyError(f"No {vehicle_code} cells for study attribute {study}::{attribute!r}")
            add(need_code=need, evidence_tier="supporting", source_table="customer_studies",
                source_key=study_key(study, attribute), source_record_ids_json=json.dumps(ids),
                source_label=f"{study}: {attribute.strip()}", data_origin="internal_bmw",
                mapping_basis="curated_study_attribute", requires_review=review, note=STUDY_CAVEAT)

    bad = {r.need_code for r in rows} - NEED_CODES
    if bad:
        raise ValueError(f"Unknown need codes: {bad}")
    session.add_all(rows)
    session.flush()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.evidence_tier] = counts.get(r.evidence_tier, 0) + 1
    counts["requires_review"] = sum(r.requires_review for r in rows)
    return counts


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


class DirectEvidence(BaseModel):
    vfc_level2_name: str
    unique_feedback_count: int
    classification_count: int
    requires_review: bool
    note: str | None


class LinkedEvidence(BaseModel):
    evidence_tier: str
    source_table: str
    source_key: str
    source_label: str
    data_origin: str
    record_ids: list[str]
    requires_review: bool
    note: str | None


class NeedEvidence(BaseModel):
    need_code: str
    vehicle_code: str
    direct: list[DirectEvidence]
    direct_unique_feedback_count: int
    supporting: list[LinkedEvidence]
    contextual: list[LinkedEvidence]
    caveats: list[str]


def evidence_for_need(session: Session, need_code: str, vehicle_code: str = "G60") -> NeedEvidence:
    links = session.scalars(
        select(CrossSourceEvidenceLink).where(CrossSourceEvidenceLink.need_code == need_code)
        .order_by(CrossSourceEvidenceLink.evidence_tier, CrossSourceEvidenceLink.source_table, CrossSourceEvidenceLink.link_id)
    ).all()
    direct_names = [l.source_key for l in links if l.evidence_tier == "direct"]
    stats = {
        name: (ids, rows) for name, ids, rows in session.execute(
            select(vfc_coalesced(), func.count(func.distinct(FeedbackClassification.feedback_record_id)), func.count())
            .where(FeedbackClassification.vehicle_code == vehicle_code, vfc_coalesced().in_(direct_names))
            .group_by(vfc_coalesced())
        )
    }
    direct_total = session.scalar(
        select(func.count(func.distinct(FeedbackClassification.feedback_record_id)))
        .where(FeedbackClassification.vehicle_code == vehicle_code, vfc_coalesced().in_(direct_names))
    ) or 0
    direct = sorted(
        (DirectEvidence(vfc_level2_name=l.source_key, unique_feedback_count=stats.get(l.source_key, (0, 0))[0],
                        classification_count=stats.get(l.source_key, (0, 0))[1], requires_review=l.requires_review,
                        note=l.note) for l in links if l.evidence_tier == "direct"),
        key=lambda d: (-d.unique_feedback_count, d.vfc_level2_name),
    )

    def linked(tier: str) -> list[LinkedEvidence]:
        return [LinkedEvidence(evidence_tier=l.evidence_tier, source_table=l.source_table, source_key=l.source_key,
                               source_label=l.source_label, data_origin=l.data_origin,
                               record_ids=json.loads(l.source_record_ids_json or "[]"),
                               requires_review=l.requires_review, note=l.note)
                for l in links if l.evidence_tier == tier]

    return NeedEvidence(
        need_code=need_code, vehicle_code=vehicle_code, direct=direct, direct_unique_feedback_count=direct_total,
        supporting=linked("supporting"), contextual=linked("contextual"),
        caveats=[PUBLIC_CAVEAT, STUDY_CAVEAT, CONTEXT_CAVEAT,
                 "Direct counts are distinct comments in the supplied export, not population prevalence."],
    )


def comments_for_need(session: Session, need_code: str, vehicle_code: str = "G60") -> list[str]:
    """Original feedback IDs with >=1 classification mapped to the need (SQL)."""
    names = select(VfcNeedMapping.vfc_level2_name).where(VfcNeedMapping.need_code == need_code)
    return sorted(session.scalars(
        select(Feedback.original_feedback_id).join(
            FeedbackClassification, FeedbackClassification.feedback_record_id == Feedback.record_id)
        .where(Feedback.vehicle_code == vehicle_code, vfc_coalesced().in_(names)).distinct()
    ))
