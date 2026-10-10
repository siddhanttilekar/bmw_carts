"""Customer insights per validated need cluster.

Facts (counts, markets, feedback IDs, evidence IDs, limitations) are assembled
by code from SQL queries. Text fields (title, problem, need, evidence summary)
come from either:

* ``deterministic_template`` - built only from stored values (VFC labels,
  AI-extracted statements quoted with their feedback ID), or
* ``llm`` - the model receives the facts and verbatim comments and returns
  ONLY the four text fields. The text is rejected (template used instead) if it
  contains a number, feedback ID or public ID that is not in the facts.

Every stored insight is re-validated against the database (``validate_insight``).
No prioritisation and no vehicle requirements are produced here.
"""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from core.ai_models import (
    AIApiCall,
    AIFeedbackExtraction,
    AIProcessingJob,
    ClusterFeedbackLink,
    CrossSourceEvidenceLink,
    CustomerNeedCluster,
    ExtractedCustomerNeed,
    GeneratedCustomerInsight,
    VfcNeedMapping,
)
from core.models import CustomerStudy, Feedback, FeedbackClassification, PublicFeedback
from services.cross_source_mapping import vfc_coalesced
from services.llm_client import LLMClient, LLMError, estimate_cost_usd
from services.need_taxonomy import NEED_NAMES

INSIGHT_PROMPT_VERSION = "insight-v1"
FEEDBACK_ID_RE = re.compile(r"\b[A-Z]\d{2}-\d{4}\b")
PUBLIC_ID_RE = re.compile(r"\bP\d{3}\b")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def insight_id_for(cluster_id: str) -> str:
    """One insight per cluster; the ID follows the stable cluster ID."""
    return f"INS-{cluster_id}"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class InsightFacts(BaseModel):
    cluster_id: str
    cluster_level: str
    need_code: str
    need_name: str
    cluster_label: str
    vfc_level2_name: str | None
    unique_feedback_count: int
    classification_count: int
    markets: list[dict]  # [{"country": "US", "unique_feedback_count": n}]
    vfc_level3_labels: list[dict]  # [{"label": ..., "unique_feedback_count": n}]
    internal_evidence_ids: list[str]
    extracted_needs: list[dict]  # [{"feedback_id", "customer_need", "problem", "polarity", "quote_verified"}]
    ai_coverage: int  # member comments with a succeeded AI extraction
    public_evidence_ids: list[str]
    study_evidence_ids: list[str]
    contextual_evidence_ids: list[str]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Facts (SQL only)
# ---------------------------------------------------------------------------


def build_facts(session: Session, cluster: CustomerNeedCluster) -> InsightFacts:
    member_rids = select(ClusterFeedbackLink.feedback_record_id).where(ClusterFeedbackLink.cluster_id == cluster.cluster_id)
    ids = sorted(set(session.scalars(select(ClusterFeedbackLink.original_feedback_id)
                                     .where(ClusterFeedbackLink.cluster_id == cluster.cluster_id))))
    markets = [
        {"country": c or "(missing)", "unique_feedback_count": n}
        for c, n in session.execute(
            select(Feedback.country, func.count(func.distinct(Feedback.record_id)))
            .where(Feedback.record_id.in_(member_rids)).group_by(Feedback.country)
            .order_by(func.count(func.distinct(Feedback.record_id)).desc(), Feedback.country))
    ]
    if cluster.vfc_level2_name is not None:
        scope = [cluster.vfc_level2_name]
    else:
        scope = list(session.scalars(select(VfcNeedMapping.vfc_level2_name).where(VfcNeedMapping.need_code == cluster.need_code)))
    level3 = [
        {"label": l or "(no level-3 label)", "unique_feedback_count": n}
        for l, n in session.execute(
            select(FeedbackClassification.vfc_level3_name, func.count(func.distinct(FeedbackClassification.feedback_record_id)))
            .where(FeedbackClassification.feedback_record_id.in_(member_rids), vfc_coalesced().in_(scope))
            .group_by(FeedbackClassification.vfc_level3_name)
            .order_by(func.count(func.distinct(FeedbackClassification.feedback_record_id)).desc(),
                      FeedbackClassification.vfc_level3_name))
    ]
    need_stmt = (select(ExtractedCustomerNeed)
                 .join(AIFeedbackExtraction, AIFeedbackExtraction.cache_key == ExtractedCustomerNeed.cache_key)
                 .where(AIFeedbackExtraction.status == "succeeded", ExtractedCustomerNeed.feedback_record_id.in_(member_rids)))
    if cluster.cluster_level == "extracted_need":
        need_stmt = need_stmt.where(ExtractedCustomerNeed.need_id.in_(
            select(ClusterFeedbackLink.extracted_need_id).where(ClusterFeedbackLink.cluster_id == cluster.cluster_id)))
    else:
        need_stmt = need_stmt.where(ExtractedCustomerNeed.need_code == cluster.need_code)
    extracted = [
        {"feedback_id": n.original_feedback_id, "customer_need": n.customer_need, "problem": n.problem,
         "polarity": n.polarity, "quote_verified": n.quote_verified}
        for n in session.scalars(need_stmt.order_by(ExtractedCustomerNeed.original_feedback_id, ExtractedCustomerNeed.need_index))
    ]
    coverage = session.scalar(select(func.count(func.distinct(AIFeedbackExtraction.feedback_record_id))).where(
        AIFeedbackExtraction.status == "succeeded", AIFeedbackExtraction.feedback_record_id.in_(member_rids))) or 0

    links = session.scalars(select(CrossSourceEvidenceLink).where(CrossSourceEvidenceLink.need_code == cluster.need_code)
                            .order_by(CrossSourceEvidenceLink.link_id)).all()
    public = [l.source_key for l in links if l.evidence_tier == "supporting" and l.source_table != "customer_studies"]
    studies = [l.source_key for l in links if l.source_table == "customer_studies"]
    context = [l.source_key for l in links if l.evidence_tier == "contextual"]
    review_links = [l.source_key for l in links if l.requires_review and l.evidence_tier != "direct"]

    limitations = ["Counts are distinct comments in the supplied G60 export, not market prevalence."]
    if cluster.classification_count != cluster.unique_feedback_count:
        limitations.append("Classification rows exceed comments because one comment can carry several VFC labels; "
                           "use unique_feedback_count for customer volume.")
    total = sum(m["unique_feedback_count"] for m in markets) or 1
    if markets and markets[0]["unique_feedback_count"] / total >= 0.8:
        limitations.append(f"Feedback is concentrated in one market ({markets[0]['country']}); other markets are thinly represented.")
    if cluster.requires_review:
        limitations.append("Cluster or its VFC mapping is marked requires_review" + (f": {cluster.review_note}" if cluster.review_note else "."))
    limitations.append(f"AI extraction covers {coverage} of {cluster.unique_feedback_count} comments in this cluster."
                       if coverage else "No AI extraction yet for these comments; need statements derive from VFC labels only.")
    if public:
        limitations.append("Public observations are curated and not representative; several rows can come from one review.")
    else:
        limitations.append("No public observation is mapped to this need.")
    if studies:
        limitations.append("Survey items have no sample sizes; US values are answer proportions; CN/EU mean scale not stated; "
                           "the China survey has no G60.")
    if context:
        limitations.append("Competitor observations are context only and are not evidence about BMW customers.")
    if review_links:
        limitations.append(f"{len(review_links)} cross-source mappings for this need are marked requires_review.")
    return InsightFacts(
        cluster_id=cluster.cluster_id, cluster_level=cluster.cluster_level, need_code=cluster.need_code,
        need_name=NEED_NAMES[cluster.need_code], cluster_label=cluster.label, vfc_level2_name=cluster.vfc_level2_name,
        unique_feedback_count=cluster.unique_feedback_count, classification_count=cluster.classification_count,
        markets=markets, vfc_level3_labels=level3, internal_evidence_ids=ids, extracted_needs=extracted,
        ai_coverage=coverage, public_evidence_ids=public, study_evidence_ids=studies, contextual_evidence_ids=context,
        limitations=limitations,
    )


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class InsightText(BaseModel):
    title: str
    customer_problem: str
    customer_need: str
    evidence_summary: str


_TEXT_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("title", "customer_problem", "customer_need", "evidence_summary")},
    "required": ["title", "customer_problem", "customer_need", "evidence_summary"],
    "additionalProperties": False,
}

_INSIGHT_SYSTEM = """You write concise customer insights for BMW product managers about the BMW 5 Series (G60).
You receive facts computed from a database and verbatim customer comments. Write:
- title: a short insight title (max 10 words)
- customer_problem: the problem customers describe, in 1-2 sentences, grounded in the comments
- customer_need: the underlying customer need, in one sentence
- evidence_summary: 1-3 sentences on what evidence exists, using only the numbers and IDs given in the facts
Rules: do not invent numbers, percentages, IDs, markets or sources. Do not recommend solutions, requirements or priorities. Do not describe competitor observations as evidence about BMW customers. If the comments do not clearly state a problem, say so."""


def template_text(facts: InsightFacts) -> InsightText:
    labels = ", ".join(f"'{l['label']}' ({l['unique_feedback_count']} comments)" for l in facts.vfc_level3_labels[:3])
    problems = [e for e in facts.extracted_needs if e["problem"] and e["polarity"] in ("negative", "mixed")]
    needs = Counter(e["customer_need"] for e in facts.extracted_needs)
    if problems:
        p = problems[0]
        problem = f"AI-extracted example ({p['feedback_id']}): {p['problem']}"
    else:
        problem = f"Original BMW VFC level-3 labels for these comments: {labels or 'none'}."
    if needs:
        top_need, _ = sorted(needs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        need = f"{top_need} (most frequent AI-extracted need statement in this cluster)"
    else:
        need = "Not yet extracted - run AI extraction for these comments."
    top_level3 = facts.vfc_level3_labels[0]["label"] if facts.vfc_level3_labels else facts.need_name
    title = f"{facts.cluster_label}: {top_level3}" if facts.cluster_level != "need_category" else facts.need_name
    markets = ", ".join(f"{m['country']} {m['unique_feedback_count']}" for m in facts.markets[:4])
    summary = (f"{facts.unique_feedback_count} distinct G60 comments ({facts.classification_count} classification rows); "
               f"markets: {markets}. Supporting references: {len(facts.public_evidence_ids)} public, "
               f"{len(facts.study_evidence_ids)} survey items; contextual: {len(facts.contextual_evidence_ids)} competitor rows.")
    return InsightText(title=title, customer_problem=problem, customer_need=need, evidence_summary=summary)


def check_llm_text(text: InsightText, facts: InsightFacts) -> list[str]:
    """Reject numbers/IDs not present in the facts."""
    allowed_numbers = {str(facts.unique_feedback_count), str(facts.classification_count), str(facts.ai_coverage),
                       str(len(facts.public_evidence_ids)), str(len(facts.study_evidence_ids)),
                       str(len(facts.contextual_evidence_ids))}
    allowed_numbers |= {str(m["unique_feedback_count"]) for m in facts.markets}
    allowed_numbers |= {str(l["unique_feedback_count"]) for l in facts.vfc_level3_labels}
    allowed_numbers |= set(NUMBER_RE.findall(" ".join([facts.cluster_label, facts.need_name] +
                                                       [e["customer_need"] for e in facts.extracted_needs])))
    joined = " ".join(text.model_dump().values())
    stripped = PUBLIC_ID_RE.sub(" ", FEEDBACK_ID_RE.sub(" ", joined))
    errors = [f"number not in facts: {n}" for n in sorted(set(NUMBER_RE.findall(stripped)) - allowed_numbers)]
    errors += [f"feedback ID not in cluster: {i}" for i in sorted(set(FEEDBACK_ID_RE.findall(joined)) - set(facts.internal_evidence_ids))]
    public_ids = {k for k in facts.public_evidence_ids if PUBLIC_ID_RE.fullmatch(k)}
    errors += [f"public ID not linked: {i}" for i in sorted(set(PUBLIC_ID_RE.findall(joined)) - public_ids)]
    return errors


def _llm_text(session: Session, llm: LLMClient, facts: InsightFacts, job: AIProcessingJob,
              max_examples: int, max_tokens: int) -> tuple[InsightText | None, list[str]]:
    examples = list(session.execute(
        select(Feedback.original_feedback_id, Feedback.customer_feedback)
        .where(Feedback.original_feedback_id.in_(facts.internal_evidence_ids[: max_examples * 5]),
               Feedback.vehicle_code == "G60")
        .order_by(Feedback.original_feedback_id).limit(max_examples)))
    payload = {"facts": facts.model_dump(exclude={"internal_evidence_ids"}),
               "example_comments": [{"feedback_id": i, "comment": t} for i, t in examples]}
    try:
        response = llm.complete_json(system=_INSIGHT_SYSTEM, user=json.dumps(payload, ensure_ascii=False, indent=1),
                                     schema=_TEXT_SCHEMA, max_tokens=max_tokens)
    except LLMError as exc:
        return None, [f"LLM error: {exc.kind}"]
    cost = estimate_cost_usd(response.model, response.usage)
    session.add(AIApiCall(job_id=job.job_id, batch_index=None, attempt=1, item_ids_json=json.dumps([facts.cluster_id]),
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
    if response.stop_reason == "refusal":
        return None, ["model declined"]
    try:
        text = InsightText.model_validate_json(response.text or "")
    except Exception:  # noqa: BLE001
        return None, ["invalid LLM output"]
    errors = check_llm_text(text, facts)
    return (None, errors) if errors else (text, [])


# ---------------------------------------------------------------------------
# Validation and generation
# ---------------------------------------------------------------------------


def validate_insight(session: Session, insight: GeneratedCustomerInsight) -> list[str]:
    """Every count and ID must match the database."""
    errors: list[str] = []
    cluster = session.get(CustomerNeedCluster, insight.cluster_id)
    if cluster is None:
        return [f"cluster {insight.cluster_id} not found"]
    members = set(session.scalars(select(ClusterFeedbackLink.original_feedback_id)
                                  .where(ClusterFeedbackLink.cluster_id == cluster.cluster_id)))
    internal = json.loads(insight.internal_evidence_ids_json)
    if set(internal) != members:
        errors.append("internal_evidence_ids differ from cluster members")
    existing = set(session.scalars(select(Feedback.original_feedback_id).where(
        Feedback.original_feedback_id.in_(internal), Feedback.vehicle_code == insight.vehicle_code)))
    if missing := set(internal) - existing:
        errors.append(f"{len(missing)} internal IDs not found for {insight.vehicle_code}")
    if insight.unique_feedback_count != len(members):
        errors.append("unique_feedback_count mismatch")
    if insight.classification_count != cluster.classification_count:
        errors.append("classification_count mismatch")
    public_keys = set(session.scalars(select(CrossSourceEvidenceLink.source_key).where(
        CrossSourceEvidenceLink.need_code == insight.need_code, CrossSourceEvidenceLink.evidence_tier == "supporting")))
    for key in json.loads(insight.public_evidence_ids_json) + json.loads(insight.study_evidence_ids_json):
        if key not in public_keys:
            errors.append(f"evidence key not linked to need: {key}")
    pids = [k for k in json.loads(insight.public_evidence_ids_json) if PUBLIC_ID_RE.fullmatch(k)]
    found = set(session.scalars(select(PublicFeedback.original_public_id).where(PublicFeedback.original_public_id.in_(pids))))
    errors += [f"public ID not in database: {p}" for p in pids if p not in found]
    for key in json.loads(insight.study_evidence_ids_json):
        study, attribute = key.split("::", 1)
        if not session.scalar(select(func.count()).select_from(CustomerStudy).where(
                CustomerStudy.study_name == study, CustomerStudy.attribute == attribute)):
            errors.append(f"study attribute not in database: {key}")
    context_keys = set(session.scalars(select(CrossSourceEvidenceLink.source_key).where(
        CrossSourceEvidenceLink.need_code == insight.need_code, CrossSourceEvidenceLink.evidence_tier == "contextual")))
    errors += [f"contextual key not linked: {k}" for k in json.loads(insight.contextual_evidence_ids_json) if k not in context_keys]
    return errors


def generate_insights(session: Session, vehicle_code: str = "G60", *, levels: tuple[str, ...] = ("vfc_category", "extracted_need"),
                      need_code: str | None = None, min_unique_feedback: int = 10, llm: LLMClient | None = None,
                      max_examples: int = 8, max_tokens: int = 4000) -> dict:
    """(Re)generate insights for clusters meeting the minimum comment count.
    The OTHER category (no usable classification) is never turned into an insight."""
    run_id = f"INS-{_now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    stmt = select(CustomerNeedCluster).where(
        CustomerNeedCluster.vehicle_code == vehicle_code, CustomerNeedCluster.cluster_level.in_(levels),
        CustomerNeedCluster.need_code != "OTHER", CustomerNeedCluster.unique_feedback_count >= min_unique_feedback)
    if need_code:
        stmt = stmt.where(CustomerNeedCluster.need_code == need_code)
    clusters = session.scalars(stmt.order_by(CustomerNeedCluster.unique_feedback_count.desc(), CustomerNeedCluster.cluster_id)).all()
    old = delete(GeneratedCustomerInsight).where(GeneratedCustomerInsight.vehicle_code == vehicle_code)
    if need_code:
        old = old.where(GeneratedCustomerInsight.need_code == need_code)
    session.execute(old)

    job = None
    if llm is not None and clusters:
        job = AIProcessingJob(job_id=f"JOB-{run_id}", job_type="insights", status="running", vehicle_code=vehicle_code,
                              model=llm.model, prompt_version=INSIGHT_PROMPT_VERSION,
                              selection_json=json.dumps([c.cluster_id for c in clusters]), batch_size=1,
                              total_items=len(clusters), processed_items=0, cached_items=0, failed_items=0, api_calls=0,
                              input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
                              estimated_cost_usd=0.0, created_at=_now(), updated_at=_now())
        session.add(job)

    stats = {"insights": 0, "llm_text": 0, "template_text": 0, "llm_rejected": 0, "invalid": 0, "run_id": run_id}
    for cluster in clusters:
        facts = build_facts(session, cluster)
        text, text_errors, source = None, [], "deterministic_template"
        if job is not None:
            text, text_errors = _llm_text(session, llm, facts, job, max_examples, max_tokens)
            job.processed_items += 1
            if text is not None:
                source = "llm"
            else:
                stats["llm_rejected"] += 1
        text = text or template_text(facts)
        insight = GeneratedCustomerInsight(
            insight_id=insight_id_for(cluster.cluster_id), cluster_id=cluster.cluster_id, vehicle_code=vehicle_code,
            need_code=cluster.need_code, title=text.title, customer_problem=text.customer_problem,
            customer_need=text.customer_need, evidence_summary=text.evidence_summary,
            unique_feedback_count=facts.unique_feedback_count, classification_count=facts.classification_count,
            markets_json=json.dumps(facts.markets), internal_evidence_ids_json=json.dumps(facts.internal_evidence_ids),
            public_evidence_ids_json=json.dumps(facts.public_evidence_ids),
            study_evidence_ids_json=json.dumps(facts.study_evidence_ids),
            contextual_evidence_ids_json=json.dumps(facts.contextual_evidence_ids),
            limitations_json=json.dumps(facts.limitations), text_source=source,
            model=llm.model if source == "llm" else None,
            prompt_version=INSIGHT_PROMPT_VERSION if source == "llm" else None,
            validated=False, validation_errors_json="[]", run_id=run_id, created_at=_now())
        session.add(insight)
        session.flush()
        errors = validate_insight(session, insight)
        insight.validated = not errors
        insight.validation_errors_json = json.dumps(errors + [f"LLM text rejected: {e}" for e in text_errors])
        stats["insights"] += 1
        stats["llm_text" if source == "llm" else "template_text"] += 1
        stats["invalid"] += bool(errors)
    if job is not None:
        job.status, job.updated_at = "completed", _now()
    session.flush()
    return stats


class InsightView(BaseModel):
    insight_id: str
    cluster_id: str
    need_code: str
    title: str
    customer_problem: str
    customer_need: str
    evidence_summary: str
    unique_feedback_count: int
    classification_count: int
    markets: list[dict]
    internal_evidence_ids: list[str]
    public_evidence_ids: list[str]
    study_evidence_ids: list[str]
    contextual_evidence_ids: list[str]
    limitations: list[str]
    text_source: str
    validated: bool
    validation_errors: list[str]


def insights_for_need(session: Session, need_code: str, vehicle_code: str = "G60") -> list[InsightView]:
    rows = session.scalars(select(GeneratedCustomerInsight).where(
        GeneratedCustomerInsight.vehicle_code == vehicle_code, GeneratedCustomerInsight.need_code == need_code)
        .order_by(GeneratedCustomerInsight.unique_feedback_count.desc(), GeneratedCustomerInsight.insight_id)).all()
    return [InsightView(
        insight_id=r.insight_id, cluster_id=r.cluster_id, need_code=r.need_code, title=r.title,
        customer_problem=r.customer_problem, customer_need=r.customer_need, evidence_summary=r.evidence_summary,
        unique_feedback_count=r.unique_feedback_count, classification_count=r.classification_count,
        markets=json.loads(r.markets_json), internal_evidence_ids=json.loads(r.internal_evidence_ids_json),
        public_evidence_ids=json.loads(r.public_evidence_ids_json), study_evidence_ids=json.loads(r.study_evidence_ids_json),
        contextual_evidence_ids=json.loads(r.contextual_evidence_ids_json), limitations=json.loads(r.limitations_json),
        text_source=r.text_source, validated=r.validated, validation_errors=json.loads(r.validation_errors_json),
    ) for r in rows]


# ---------------------------------------------------------------------------
# Generation status (for display)
# ---------------------------------------------------------------------------

STATUS_TEMPLATE = "TEMPLATE"        # text built from original VFC classifications only
STATUS_AI_ENRICHED = "AI_ENRICHED"  # uses actual LLM extraction results or validated LLM text
STATUS_PENDING = "PENDING"          # failed validation or no usable content yet - needs further analysis

STATUS_DESCRIPTIONS = {
    STATUS_TEMPLATE: "Generated from existing BMW VFC classifications by a deterministic template. Not AI-generated.",
    STATUS_AI_ENRICHED: "Uses actual LLM extraction results for comments in this cluster (validated against the database).",
    STATUS_PENDING: "Requires further analysis: the insight failed validation or has no usable content yet.",
}


def insight_status(session: Session, insight: GeneratedCustomerInsight) -> str:
    """Derived from stored facts, never assumed. AI_ENRICHED requires real
    extraction rows (or validated LLM text) for this cluster's comments."""
    if not insight.validated:
        return STATUS_PENDING
    if insight.text_source == "llm":
        return STATUS_AI_ENRICHED
    member_rids = select(ClusterFeedbackLink.feedback_record_id).where(ClusterFeedbackLink.cluster_id == insight.cluster_id)
    extracted = session.scalar(select(func.count()).select_from(ExtractedCustomerNeed)
                               .join(AIFeedbackExtraction, AIFeedbackExtraction.cache_key == ExtractedCustomerNeed.cache_key)
                               .where(AIFeedbackExtraction.status == "succeeded",
                                      ExtractedCustomerNeed.feedback_record_id.in_(member_rids),
                                      ExtractedCustomerNeed.need_code == insight.need_code)) or 0
    return STATUS_AI_ENRICHED if extracted else STATUS_TEMPLATE


class InsightCard(InsightView):
    status: str
    cluster_level: str
    cluster_label: str
    vfc_level2_name: str | None
    cluster_requires_review: bool


def all_insight_cards(session: Session, vehicle_code: str = "G60") -> list[InsightCard]:
    """Every stored insight with its derived status and cluster context."""
    cards = []
    for need_code in sorted(set(session.scalars(select(GeneratedCustomerInsight.need_code)
                                                .where(GeneratedCustomerInsight.vehicle_code == vehicle_code)))):
        for view in insights_for_need(session, need_code, vehicle_code):
            row = session.get(GeneratedCustomerInsight, view.insight_id)
            cluster = session.get(CustomerNeedCluster, view.cluster_id)
            cards.append(InsightCard(
                **view.model_dump(), status=insight_status(session, row),
                cluster_level=cluster.cluster_level if cluster else "missing",
                cluster_label=cluster.label if cluster else "(cluster missing)",
                vfc_level2_name=cluster.vfc_level2_name if cluster else None,
                cluster_requires_review=bool(cluster and cluster.requires_review)))
    return sorted(cards, key=lambda c: (-c.unique_feedback_count, c.insight_id))
