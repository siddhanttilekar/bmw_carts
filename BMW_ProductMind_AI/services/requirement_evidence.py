"""Evidence package for requirement generation (Part 4).

Given a cluster (need category ``NEED-*``, VFC category ``VFC-*`` or AI need
``XN-*``) or an insight, assemble everything a requirement may cite, using
actual database IDs only:

* DIRECT      the cluster's original G60 comments (cluster members)
* SUPPORTING  public BMW owner observations and BMW customer-study items,
              linked at NEED-CATEGORY level (never proof of a specific complaint)
* CONTEXTUAL  competitor observations and G60 option-list equipment

Reuses Part 2/3 services (insight facts, cross-source evidence, retrieval).
New queries here: equipment context, and distinct comments per VFC level-3
label group.
"""
from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.ai_models import CustomerNeedCluster, GeneratedCustomerInsight, VfcNeedMapping
from core.models import Equipment, FeedbackClassification
from services.cross_source_mapping import vfc_coalesced
from services.dashboard_queries import EvidenceItem, related_evidence
from services.evidence_retrieval import get_feedback_by_ids
from services.insight_generator import build_facts, insight_id_for
from services.need_taxonomy import MISSING_VFC, NEED_NAMES

VEHICLE = "G60"
DIRECT_PREVIEW = 25
OPTION_LIST_NOTE = ("G60 option list (German market, July 2026 edition): shows what is currently offered and at "
                    "which price, not customer evidence.")

# Curated: which option-list sections / SA codes give equipment context for a need.
EQUIPMENT_CONTEXT: dict[str, dict[str, list[str]]] = {
    "DRIVER_ASSISTANCE": {"sections": ["FAHRASSISTENZ UND SICHERHEIT"], "codes": []},
    "CLIMATE": {"sections": ["KLIMAKOMFORT"], "codes": ["248"]},
    "INFOTAINMENT": {"sections": ["KONNEKTIVITÄT UND INFOTAINMENT"], "codes": []},
    "DRIVING_DYNAMICS": {"sections": ["ANTRIEB / FAHRDYNAMIK / FAHRWERK"], "codes": ["704"]},
    "INTERIOR_COMFORT": {"sections": ["SITZKOMFORT"], "codes": ["4MA"]},
    "INTERIOR_QUALITY": {"sections": ["INTERIEUR", "INTERIEURLEISTEN"], "codes": []},
    "PRACTICALITY": {"sections": ["TRANSPORT UND STAURAUM"], "codes": []},
    "EXTERIOR": {"sections": ["EXTERIEUR"], "codes": []},
    "CHARGING": {"sections": [], "codes": ["4U6"]},
}

# Original BMW VFC level-3 labels grouped by what they describe (not sentiment).
POSITIVE_LABELS = {"Look / design appeals", "Likes comfort", "Likes size", "Workmanship appeals"}
CATCH_ALL_VFC = {"Exterior", "Interior", "Accessories / Parts", "General enquiry: customer feedback", "no_class_found",
                 MISSING_VFC}


def label_group(label: str | None) -> str:
    if not label:
        return "unlabelled"
    if label.startswith("Like") or label in POSITIVE_LABELS:
        return "positive"
    if label.startswith("Customer request"):
        return "request"
    return "issue"


class DirectItem(BaseModel):
    feedback_id: str
    record_id: str
    source_file: str
    source_sheet: str | None
    source_row: int | None
    country: str | None
    engine_type: str | None
    text: str
    vfc_labels: list[str]


class EquipmentItem(BaseModel):
    option_code: str
    name: str | None
    section: str | None
    price_gross_eur_text: str | None
    source_file: str
    source_page: int | None
    record_id: str


class EvidencePackage(BaseModel):
    cluster_id: str
    cluster_level: str
    cluster_label: str
    vfc_level2_name: str | None
    need_code: str
    need_name: str
    vehicle_code: str
    insight_id: str | None
    unique_feedback_count: int
    classification_count: int
    markets: list[dict]
    vfc_level3_labels: list[dict]
    label_groups: dict[str, int]  # distinct comments per label group (a comment can be in several)
    direct_feedback_ids: list[str]
    direct_evidence: list[DirectItem]  # preview with full text and source refs
    supporting_evidence: list[EvidenceItem]
    contextual_evidence: list[EvidenceItem]
    equipment_context: list[EquipmentItem]
    mapping_requires_review: bool
    mapping_note: str | None
    extracted_need_count: int
    limitations: list[str]
    missing_evidence: list[str]


def _cluster(session: Session, cluster_id: str) -> CustomerNeedCluster:
    cluster = session.get(CustomerNeedCluster, cluster_id)
    if cluster is None:
        raise ValueError(f"Unknown cluster {cluster_id!r}")
    if cluster.vehicle_code != VEHICLE:
        raise ValueError(f"Cluster {cluster_id} is not a {VEHICLE} cluster")
    return cluster


def cluster_for_insight(session: Session, insight_id: str) -> str:
    insight = session.get(GeneratedCustomerInsight, insight_id)
    if insight is None:
        raise ValueError(f"Unknown insight {insight_id!r}")
    return insight.cluster_id


def equipment_context(session: Session, need_code: str) -> list[EquipmentItem]:
    spec = EQUIPMENT_CONTEXT.get(need_code)
    if not spec:
        return []
    cond = Equipment.section.in_(spec["sections"])
    if spec["codes"]:
        cond = cond | Equipment.option_code.in_(spec["codes"])
    rows = session.scalars(select(Equipment).where(Equipment.vehicle_code == VEHICLE, cond)
                           .order_by(Equipment.source_page, Equipment.option_code)).all()
    seen, out = set(), []
    for e in rows:
        if e.option_code in seen:
            continue
        seen.add(e.option_code)
        out.append(EquipmentItem(option_code=e.option_code, name=e.name, section=e.section,
                                 price_gross_eur_text=e.price_gross_eur_text, source_file=e.source_file,
                                 source_page=e.source_page, record_id=e.record_id))
    return out


def _label_groups(session: Session, member_ids: list[str], scope: list[str]) -> dict[str, int]:
    from core.models import Feedback

    rows = session.execute(
        select(FeedbackClassification.feedback_record_id, FeedbackClassification.vfc_level3_name)
        .join(Feedback, Feedback.record_id == FeedbackClassification.feedback_record_id)
        .where(Feedback.original_feedback_id.in_(member_ids), Feedback.vehicle_code == VEHICLE,
               vfc_coalesced().in_(scope))).all()
    groups: dict[str, set[str]] = {"issue": set(), "request": set(), "positive": set(), "unlabelled": set()}
    for rid, label in rows:
        groups[label_group(label)].add(rid)
    return {k: len(v) for k, v in groups.items()}


def build_evidence_package(session: Session, *, cluster_id: str | None = None, insight_id: str | None = None,
                           need_code: str | None = None) -> EvidencePackage:
    """Exactly one selector: a cluster, an insight, or a whole need category."""
    if sum(x is not None for x in (cluster_id, insight_id, need_code)) != 1:
        raise ValueError("Pass exactly one of cluster_id, insight_id, need_code")
    if insight_id:
        cluster_id = cluster_for_insight(session, insight_id)
    if need_code:
        cluster_id = f"NEED-{need_code}"
    cluster = _cluster(session, cluster_id)
    facts = build_facts(session, cluster)  # SQL-computed counts, members, markets, links, limitations

    if cluster.vfc_level2_name is not None:
        scope = [cluster.vfc_level2_name]
    else:
        scope = list(session.scalars(select(VfcNeedMapping.vfc_level2_name)
                                     .where(VfcNeedMapping.need_code == cluster.need_code)))
    mapping = session.get(VfcNeedMapping, cluster.vfc_level2_name) if cluster.vfc_level2_name else None

    records = get_feedback_by_ids(session, facts.internal_evidence_ids[:DIRECT_PREVIEW], VEHICLE)
    direct = [DirectItem(feedback_id=r.original_feedback_id, record_id=r.source.record_id, source_file=r.source.source_file,
                         source_sheet=r.source.source_sheet, source_row=r.source.source_row, country=r.country,
                         engine_type=r.engine_type, text=r.customer_feedback,
                         vfc_labels=[c.vfc_level2_3_name or "(empty VFC cell)" for c in r.classifications])
              for r in records]

    linked = related_evidence(session, cluster.need_code)
    supporting = [i for i in linked if i.tier == "SUPPORTING"]
    contextual = [i for i in linked if i.tier == "CONTEXTUAL"]
    equipment = equipment_context(session, cluster.need_code)
    insight = session.get(GeneratedCustomerInsight, insight_id_for(cluster.cluster_id))

    missing: list[str] = []
    if facts.unique_feedback_count < 5:
        missing.append(f"Only {facts.unique_feedback_count} direct comments - too few to define a specific requirement.")
    if not facts.extracted_needs:
        missing.append("No AI extraction of customer wording: the customer need is derived from VFC labels only.")
    if not any(i.kind in ("public_owner_feedback", "public_bmw_observation") for i in supporting):
        missing.append("No public BMW owner observation is mapped to this need.")
    if not any(i.kind == "customer_study" for i in supporting):
        missing.append("No customer-study attribute is mapped to this need.")
    if not equipment:
        missing.append("No G60 option-list equipment context is mapped to this need.")
    if cluster.cluster_level == "need_category":
        missing.append("Need-category selection spans several VFC classes; the affected function is not specific.")
    if (cluster.vfc_level2_name or "") in CATCH_ALL_VFC or "overall technology" in (cluster.vfc_level2_name or ""):
        missing.append("VFC class is a broad catch-all; the affected function is not specific.")
    review_items = [i.key for i in supporting + contextual if i.requires_review]
    if review_items:
        missing.append(f"{len(review_items)} linked supporting/contextual items have uncertain mappings (requires review).")

    return EvidencePackage(
        cluster_id=cluster.cluster_id, cluster_level=cluster.cluster_level, cluster_label=cluster.label,
        vfc_level2_name=cluster.vfc_level2_name, need_code=cluster.need_code, need_name=NEED_NAMES[cluster.need_code],
        vehicle_code=VEHICLE, insight_id=insight.insight_id if insight else None,
        unique_feedback_count=facts.unique_feedback_count, classification_count=facts.classification_count,
        markets=facts.markets, vfc_level3_labels=facts.vfc_level3_labels,
        label_groups=_label_groups(session, facts.internal_evidence_ids, scope),
        direct_feedback_ids=facts.internal_evidence_ids, direct_evidence=direct, supporting_evidence=supporting,
        contextual_evidence=contextual, equipment_context=equipment,
        mapping_requires_review=bool(cluster.requires_review), mapping_note=(mapping.rationale if mapping else cluster.review_note),
        extracted_need_count=len(facts.extracted_needs),
        limitations=facts.limitations + [OPTION_LIST_NOTE] * bool(equipment),
        missing_evidence=missing,
    )
