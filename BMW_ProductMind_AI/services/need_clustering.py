"""Customer-need clustering for G60 feedback.

Three levels, all linked back to original comments via cluster_feedback_links:

1. ``need_category``  comments with >=1 VFC classification mapped to the need
2. ``vfc_category``   comments with a given original VFC level-2 class
                      (child of its need category)
3. ``extracted_need`` AI-extracted needs grouped within one need category:
                      - deterministic: identical normalised need statements
                      - optional ``llm`` grouping: the model only assigns need
                        IDs to semantic groups; it never sees or returns counts,
                        and every returned ID is validated.

Counts are ALWAYS computed with SQL:
* ``unique_feedback_count`` - distinct comments linked to the cluster
* ``classification_count``  - VFC classification rows of those comments that
  fall in the cluster's scope (need category / VFC class). For extracted-need
  clusters: rows of member comments whose VFC maps to the same need category
  ("aligned classification rows"); can be 0 when the AI category differs.
* ``extracted_need_count``  - AI need statements linked to the cluster
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import delete, func, insert, literal, select
from sqlalchemy.orm import Session

from core.ai_models import (
    AIFeedbackExtraction,
    AIProcessingJob,
    ClusterFeedbackLink,
    CustomerNeedCluster,
    ExtractedCustomerNeed,
    VfcNeedMapping,
)
from core.models import Feedback, FeedbackClassification
from services.cross_source_mapping import vfc_coalesced
from services.llm_client import LLMClient, LLMError, estimate_cost_usd
from services.need_taxonomy import NEED_NAMES, TAXONOMY

GROUPING_PROMPT_VERSION = "group-v1"
_ORDER = {c.code: i for i, c in enumerate(TAXONOMY)}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _digest(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# Stable cluster identifiers: derived from canonical keys, never from build order,
# so rebuilding clusters keeps IDs (and insight/evidence references) unchanged.
def need_cluster_id(need_code: str) -> str:
    return f"NEED-{need_code}"


def vfc_cluster_id(vfc_level2_name: str) -> str:
    """From the original (immutable) VFC level-2 name; '<missing>' for empty cells."""
    return f"VFC-{_digest('vfc|' + vfc_level2_name)}"


def extracted_cluster_id(need_code: str, canonical_key: str) -> str:
    return f"XN-{_digest(f'xn|{need_code}|{canonical_key}')}"


def normalize_need(text: str) -> str:
    return re.sub(r"[^\w\s]", "", " ".join(text.split()).casefold()).strip()


# ---------------------------------------------------------------------------
# Count helpers (SQL only)
# ---------------------------------------------------------------------------


def _unique_count(session: Session, cluster_id: str) -> int:
    return session.scalar(select(func.count(func.distinct(ClusterFeedbackLink.feedback_record_id)))
                          .where(ClusterFeedbackLink.cluster_id == cluster_id)) or 0


def _member_ids(cluster_id: str):
    return select(ClusterFeedbackLink.feedback_record_id).where(ClusterFeedbackLink.cluster_id == cluster_id)


def _classification_count(session: Session, cluster_id: str, vfc_names) -> int:
    return session.scalar(
        select(func.count()).select_from(FeedbackClassification).where(
            FeedbackClassification.feedback_record_id.in_(_member_ids(cluster_id)), vfc_coalesced().in_(vfc_names))
    ) or 0


def _refresh_counts(session: Session, cluster: CustomerNeedCluster, vfc_scope) -> None:
    session.flush()
    cluster.unique_feedback_count = _unique_count(session, cluster.cluster_id)
    cluster.classification_count = _classification_count(session, cluster.cluster_id, vfc_scope)
    cluster.extracted_need_count = session.scalar(
        select(func.count()).select_from(ClusterFeedbackLink).where(
            ClusterFeedbackLink.cluster_id == cluster.cluster_id, ClusterFeedbackLink.extracted_need_id.is_not(None))
    ) or 0


# ---------------------------------------------------------------------------
# LLM-assisted grouping (optional)
# ---------------------------------------------------------------------------

_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "customer_need": {"type": "string"},
                    "member_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "customer_need", "member_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

_GROUP_SYSTEM = """You group customer-need statements extracted from BMW 5 Series (G60) feedback.
Put statements in the same group only if they express the same underlying customer need (same function and same kind of need). Do NOT group statements merely because they share words or the same vehicle system; a complaint about touchscreen responsiveness and a wish for physical climate buttons are different needs.
Return each statement ID at most once, using the IDs exactly as given. Statements that fit no group may be returned as single-member groups.
For each group give a short label and one sentence describing the shared customer need. Do not include counts or numbers."""


class _Group(BaseModel):
    label: str
    customer_need: str
    member_ids: list[str]


class _Groups(BaseModel):
    groups: list[_Group]


def _llm_groups(session: Session, llm: LLMClient, need_code: str, statements: dict[str, str],
                job: AIProcessingJob, max_tokens: int) -> tuple[list[_Group], list[str]]:
    """Returns (validated groups, warnings). Invalid/unknown/duplicate IDs are dropped;
    statements the model left out become single-member groups."""
    from core.ai_models import AIApiCall

    payload = [{"id": k, "need": v} for k, v in statements.items()]
    user = (f"Need category: {NEED_NAMES[need_code]}\nGroup these need statements:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))
    warnings: list[str] = []
    try:
        response = llm.complete_json(system=_GROUP_SYSTEM, user=user, schema=_GROUP_SCHEMA, max_tokens=max_tokens)
    except LLMError as exc:
        warnings.append(f"LLM grouping failed for {need_code}: {exc.kind}; fell back to exact grouping")
        return [], warnings
    cost = estimate_cost_usd(response.model, response.usage)
    session.add(AIApiCall(job_id=job.job_id, batch_index=None, attempt=1, item_ids_json=json.dumps(list(statements)),
                          status="succeeded" if response.stop_reason != "refusal" else "refused", model=response.model,
                          request_id=response.request_id, stop_reason=response.stop_reason,
                          input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
                          cache_read_tokens=response.usage.cache_read_tokens,
                          cache_write_tokens=response.usage.cache_write_tokens, estimated_cost_usd=cost,
                          latency_ms=response.latency_ms, created_at=_now()))
    job.api_calls += 1
    job.input_tokens += response.usage.input_tokens
    job.output_tokens += response.usage.output_tokens
    job.estimated_cost_usd = round(job.estimated_cost_usd + cost, 6)
    try:
        parsed = _Groups.model_validate_json(response.text or "")
    except Exception:  # noqa: BLE001 - invalid output -> deterministic fallback
        warnings.append(f"Invalid grouping output for {need_code}; fell back to exact grouping")
        return [], warnings
    used: set[str] = set()
    groups: list[_Group] = []
    for g in parsed.groups:
        members = [m for m in g.member_ids if m in statements and m not in used]
        dropped = [m for m in g.member_ids if m not in statements]
        if dropped:
            warnings.append(f"Dropped {len(dropped)} unknown IDs returned by the model ({need_code})")
        used.update(members)
        if members:
            groups.append(_Group(label=g.label, customer_need=g.customer_need, member_ids=members))
    for sid, text in statements.items():
        if sid not in used:
            groups.append(_Group(label=text[:80], customer_need=text, member_ids=[sid]))
    return groups, warnings


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_clusters(session: Session, vehicle_code: str = "G60", *, llm: LLMClient | None = None,
                   max_tokens: int = 16000) -> dict:
    """Rebuild all clusters for the vehicle. Deterministic unless ``llm`` is
    given (then extracted-need statements are grouped semantically)."""
    run_id = f"CLU-{_now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    old = select(CustomerNeedCluster.cluster_id).where(CustomerNeedCluster.vehicle_code == vehicle_code)
    session.execute(delete(ClusterFeedbackLink).where(ClusterFeedbackLink.cluster_id.in_(old)))
    session.execute(delete(CustomerNeedCluster).where(CustomerNeedCluster.vehicle_code == vehicle_code))
    session.flush()

    mappings = session.scalars(select(VfcNeedMapping)).all()
    if not mappings:
        raise RuntimeError("VFC mappings missing - run the taxonomy setup first")
    by_need: dict[str, list[VfcNeedMapping]] = {}
    for m in mappings:
        by_need.setdefault(m.need_code, []).append(m)
    present = set(session.scalars(
        select(vfc_coalesced()).where(FeedbackClassification.vehicle_code == vehicle_code).distinct()))


    def link_by_vfc(cluster_id: str, names: list[str]) -> None:
        sub = (select(literal(cluster_id), Feedback.record_id, Feedback.original_feedback_id,
                      literal("vfc_classification"), literal(None))
               .join(FeedbackClassification, FeedbackClassification.feedback_record_id == Feedback.record_id)
               .where(Feedback.vehicle_code == vehicle_code, vfc_coalesced().in_(names)).distinct())
        session.execute(insert(ClusterFeedbackLink).from_select(
            ["cluster_id", "feedback_record_id", "original_feedback_id", "link_basis", "extracted_need_id"], sub))

    need_cluster_ids: dict[str, str] = {}
    summary = {"need_category": 0, "vfc_category": 0, "extracted_need": 0, "warnings": []}

    # Levels 1 and 2 (deterministic, from original VFC classes)
    for need in sorted(by_need, key=lambda c: _ORDER[c]):
        names = sorted(m.vfc_level2_name for m in by_need[need] if m.vfc_level2_name in present)
        if not names:
            continue
        review = [m for m in by_need[need] if m.vfc_level2_name in present and m.mapping_status == "review_required"]
        parent = CustomerNeedCluster(
            cluster_id=need_cluster_id(need), vehicle_code=vehicle_code, cluster_level="need_category", need_code=need,
            label=NEED_NAMES[need], method="deterministic_vfc_mapping", unique_feedback_count=0,
            classification_count=0, requires_review=bool(review) or need in ("OTHER", "OVERALL"),
            review_note=(f"{len(review)} of {len(names)} VFC classes mapped with review_required" if review else None),
            run_id=run_id, created_at=_now())
        session.add(parent)
        session.flush()
        link_by_vfc(parent.cluster_id, names)
        _refresh_counts(session, parent, names)
        need_cluster_ids[need] = parent.cluster_id
        summary["need_category"] += 1
        for m in sorted((m for m in by_need[need] if m.vfc_level2_name in present), key=lambda m: m.vfc_level2_name):
            child = CustomerNeedCluster(
                cluster_id=vfc_cluster_id(m.vfc_level2_name), vehicle_code=vehicle_code, cluster_level="vfc_category",
                need_code=need,
                vfc_level2_name=m.vfc_level2_name, parent_cluster_id=parent.cluster_id, label=m.vfc_level2_name,
                method="deterministic_vfc", unique_feedback_count=0, classification_count=0,
                requires_review=m.mapping_status == "review_required", review_note=m.rationale, run_id=run_id,
                created_at=_now())
            session.add(child)
            session.flush()
            link_by_vfc(child.cluster_id, [m.vfc_level2_name])
            _refresh_counts(session, child, [m.vfc_level2_name])
            summary["vfc_category"] += 1

    # Level 3 (extracted needs)
    needs = session.scalars(
        select(ExtractedCustomerNeed).join(AIFeedbackExtraction, AIFeedbackExtraction.cache_key == ExtractedCustomerNeed.cache_key)
        .where(ExtractedCustomerNeed.vehicle_code == vehicle_code, AIFeedbackExtraction.status == "succeeded")
        .order_by(ExtractedCustomerNeed.original_feedback_id, ExtractedCustomerNeed.need_index)
    ).all()
    job = None
    if llm is not None and needs:
        job = AIProcessingJob(job_id=f"JOB-{run_id}", job_type="cluster_grouping", status="running",
                              vehicle_code=vehicle_code, model=llm.model, prompt_version=GROUPING_PROMPT_VERSION,
                              selection_json=json.dumps([n.need_id for n in needs]), batch_size=0,
                              total_items=len(needs), processed_items=0, cached_items=0, failed_items=0, api_calls=0,
                              input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
                              estimated_cost_usd=0.0, created_at=_now(), updated_at=_now())
        session.add(job)

    by_code: dict[str, list[ExtractedCustomerNeed]] = {}
    for n in needs:
        by_code.setdefault(n.need_code, []).append(n)
    for need_code in sorted(by_code, key=lambda c: _ORDER[c]):
        items = by_code[need_code]
        groups: list[_Group] = []
        method = "deterministic_exact_statement"
        if job is not None:
            groups, warnings = _llm_groups(session, llm, need_code, {n.need_id: n.customer_need for n in items}, job, max_tokens)
            summary["warnings"].extend(warnings)
            method = "llm_semantic_grouping" if groups else method
        if not groups:
            exact: dict[str, list[ExtractedCustomerNeed]] = {}
            for n in items:
                exact.setdefault(normalize_need(n.customer_need), []).append(n)
            groups = [_Group(label=members[0].customer_need[:80], customer_need=members[0].customer_need,
                             member_ids=[m.need_id for m in members]) for members in exact.values()]
        need_by_id = {n.need_id: n for n in items}
        scope = [m.vfc_level2_name for m in by_need.get(need_code, [])]
        for g in sorted(groups, key=lambda g: (-len(g.member_ids), g.label)):
            members = [need_by_id[i] for i in g.member_ids]
            if method == "deterministic_exact_statement":
                key = normalize_need(g.customer_need)
            else:  # semantic groups are identified by their (stable) member need IDs
                key = "|".join(sorted(g.member_ids))
            cluster = CustomerNeedCluster(
                cluster_id=extracted_cluster_id(need_code, key), vehicle_code=vehicle_code,
                cluster_level="extracted_need", need_code=need_code,
                parent_cluster_id=need_cluster_ids.get(need_code), label=g.label, customer_need=g.customer_need,
                method=method, unique_feedback_count=0, classification_count=0,
                requires_review=method == "llm_semantic_grouping" or all(m.confidence == "low" for m in members),
                review_note="Semantic grouping by LLM; verify members." if method == "llm_semantic_grouping" else None,
                run_id=run_id, created_at=_now())
            session.add(cluster)
            session.flush()
            session.add_all(ClusterFeedbackLink(cluster_id=cluster.cluster_id, feedback_record_id=m.feedback_record_id,
                                                original_feedback_id=m.original_feedback_id, link_basis="extracted_need",
                                                extracted_need_id=m.need_id) for m in members)
            _refresh_counts(session, cluster, scope)
            summary["extracted_need"] += 1
    if job is not None:
        job.status, job.processed_items, job.updated_at = "completed", len(needs), _now()
    session.flush()
    summary["run_id"] = run_id
    return summary


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


class ClusterView(BaseModel):
    cluster_id: str
    cluster_level: str
    need_code: str
    label: str
    customer_need: str | None
    vfc_level2_name: str | None
    parent_cluster_id: str | None
    method: str
    unique_feedback_count: int
    classification_count: int
    extracted_need_count: int
    requires_review: bool
    review_note: str | None
    feedback_ids: list[str]


def clusters_for_need(session: Session, need_code: str, vehicle_code: str = "G60",
                      levels: tuple[str, ...] = ("need_category", "vfc_category", "extracted_need")) -> list[ClusterView]:
    rows = session.scalars(select(CustomerNeedCluster).where(
        CustomerNeedCluster.vehicle_code == vehicle_code, CustomerNeedCluster.need_code == need_code,
        CustomerNeedCluster.cluster_level.in_(levels)).order_by(CustomerNeedCluster.cluster_id)).all()
    out = []
    for c in rows:
        ids = sorted(set(session.scalars(select(ClusterFeedbackLink.original_feedback_id)
                                         .where(ClusterFeedbackLink.cluster_id == c.cluster_id))))
        out.append(ClusterView(
            cluster_id=c.cluster_id, cluster_level=c.cluster_level, need_code=c.need_code, label=c.label,
            customer_need=c.customer_need, vfc_level2_name=c.vfc_level2_name, parent_cluster_id=c.parent_cluster_id,
            method=c.method, unique_feedback_count=c.unique_feedback_count, classification_count=c.classification_count,
            extracted_need_count=c.extracted_need_count, requires_review=c.requires_review, review_note=c.review_note,
            feedback_ids=ids))
    return out
