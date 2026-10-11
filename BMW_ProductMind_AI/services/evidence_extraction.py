"""Controlled observation extraction (ProductMind AI 2.0): sample first,
explicit approval for every paid batch, budget ceiling, full run only after a
completed sample and an explicit confirmation.

Nothing here runs automatically: the UI and CLI call these functions only on
an explicit user action, and tests use a fake LLM client.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.ai_models import AIFeedbackExtraction, AIProcessingJob
from core.config import LLMSettings
from core.evidence_models import AIFeedbackObservation
from services import ai_feedback_analysis as ai
from services.llm_budget import (
    SAMPLE_SIZE,
    ApprovalRequired,
    CostEstimate,
    batch_guard,
    check_approval,
    estimate_cost,
)


class SampleRequired(ApprovalRequired):
    pass


@dataclass
class ExtractionPlan:
    feedback_ids: list[str]
    description: str
    estimate: CostEstimate
    full_run: bool


def plan_sample(session: Session, settings: LLMSettings, size: int = SAMPLE_SIZE) -> ExtractionPlan:
    ids = ai.select_representative_sample(session, size)
    spec = ai.observation_spec(settings)
    return ExtractionPlan(ids, f"representative sample n={size} (round-robin across need categories)",
                          estimate_cost(session, ai.load_comments(session, ids), spec, settings), False)


def plan_full(session: Session, settings: LLMSettings) -> ExtractionPlan:
    ids = ai.all_feedback_ids(session)
    spec = ai.observation_spec(settings)
    return ExtractionPlan(ids, f"all G60 comments ({len(ids)})",
                          estimate_cost(session, ai.load_comments(session, ids), spec, settings), True)


def completed_sample_job(session: Session, settings: LLMSettings) -> AIProcessingJob | None:
    return session.scalar(select(AIProcessingJob).where(
        AIProcessingJob.job_type == ai.OBSERVATION_JOB_TYPE, AIProcessingJob.model == settings.model,
        AIProcessingJob.prompt_version == ai.OBSERVATION_PROMPT_VERSION,
        AIProcessingJob.status.in_(("completed", "completed_with_failures")),
        AIProcessingJob.total_items <= ai.FULL_RUN_THRESHOLD,
    ).order_by(AIProcessingJob.updated_at.desc()).limit(1))


def run_observation_extraction(session: Session, llm, settings: LLMSettings, plan: ExtractionPlan, *,
                               approved_max_usd: float | None, confirm_full_run: bool = False,
                               progress=None, sleep=None) -> AIProcessingJob:
    """Execute an approved plan. Raises ApprovalRequired / BudgetExceeded /
    SampleRequired / FullRunNotConfirmed before any request is sent."""
    spec = ai.observation_spec(settings)
    if plan.full_run or len(plan.feedback_ids) > ai.FULL_RUN_THRESHOLD:
        if not confirm_full_run:
            raise ai.FullRunNotConfirmed("Full extraction requires explicit confirmation.")
        if completed_sample_job(session, settings) is None:
            raise SampleRequired("Run and review the representative sample before full extraction.")
    # re-estimate now (cache state may have changed since the plan was shown)
    estimate = estimate_cost(session, ai.load_comments(session, plan.feedback_ids), spec, settings, llm.model)
    check_approval(session, estimate, settings, approved_max_usd)
    kwargs = {"progress": progress}
    if sleep is not None:
        kwargs["sleep"] = sleep
    return ai.run_extraction(
        session, llm, settings, plan.feedback_ids, selection_description=plan.description,
        allow_full_run=confirm_full_run, spec=spec,
        before_batch=batch_guard(settings, approved_max_usd or 0.0, spec, llm.model) if estimate.to_send else None,
        **kwargs)


def extraction_status(session: Session, settings: LLMSettings) -> dict:
    version_id = ai.current_observation_version_id(settings)
    total = len(ai.all_feedback_ids(session))
    processed = ai.observed_feedback_ids(session, version_id)
    by_type = dict(session.execute(
        select(AIFeedbackObservation.observation_type, func.count())
        .where(AIFeedbackObservation.version_id == version_id).group_by(AIFeedbackObservation.observation_type)).all())
    comments_by_type = dict(session.execute(
        select(AIFeedbackObservation.observation_type, func.count(func.distinct(AIFeedbackObservation.original_feedback_id)))
        .where(AIFeedbackObservation.version_id == version_id).group_by(AIFeedbackObservation.observation_type)).all())
    rejected = 0
    for raw in session.scalars(select(AIFeedbackExtraction.raw_json).where(
            AIFeedbackExtraction.observation_version_id == version_id, AIFeedbackExtraction.status == "succeeded")):
        rejected += len(json.loads(raw or "{}").get("rejected", []))
    failed = session.scalar(select(func.count()).select_from(AIFeedbackExtraction).where(
        AIFeedbackExtraction.prompt_version == ai.OBSERVATION_PROMPT_VERSION, AIFeedbackExtraction.status == "failed",
        AIFeedbackExtraction.model == settings.model)) or 0
    jobs = [{"job_id": j.job_id, "status": j.status, "total": j.total_items, "processed": j.processed_items,
             "cached": j.cached_items, "failed": j.failed_items, "api_calls": j.api_calls,
             "cost_usd": round(j.estimated_cost_usd, 4), "description": j.selection_description,
             "last_error": j.last_error, "updated_at": j.updated_at}
            for j in session.scalars(select(AIProcessingJob).where(AIProcessingJob.job_type == ai.OBSERVATION_JOB_TYPE)
                                     .order_by(AIProcessingJob.created_at.desc()))]
    return {"observation_version_id": version_id, "model": settings.model,
            "model_version": settings.model_version or settings.model,
            "prompt_version": ai.OBSERVATION_PROMPT_VERSION, "taxonomy_version": ai.TAXONOMY_VERSION,
            "g60_comments": total, "processed_comments": len(processed), "failed_comments": failed,
            "validated_observations": sum(by_type.values()), "observations_by_type": by_type,
            "comments_by_type": comments_by_type, "rejected_observations": rejected,
            "sample_completed": completed_sample_job(session, settings) is not None, "jobs": jobs}


def counts_by_category(session: Session, settings: LLMSettings) -> dict[str, dict[str, int]]:
    """{need_code: {observation_type: distinct comments}} for the current version."""
    version_id = ai.current_observation_version_id(settings)
    out: dict[str, dict[str, int]] = {}
    for code, otype, n in session.execute(
        select(AIFeedbackObservation.taxonomy_category, AIFeedbackObservation.observation_type,
               func.count(func.distinct(AIFeedbackObservation.original_feedback_id)))
        .where(AIFeedbackObservation.version_id == version_id)
        .group_by(AIFeedbackObservation.taxonomy_category, AIFeedbackObservation.observation_type)
    ):
        out.setdefault(code, {})[otype] = n
    return out


def need_summary(session: Session, settings: LLMSettings, need_code: str, limit: int = 8) -> dict:
    """AI-identified needs in one category: distinct-comment counts per type,
    the most frequent need phrases, and example observations with the original
    comment text (all from validated observations)."""
    from core.models import Feedback

    version_id = ai.current_observation_version_id(settings)
    obs = list(session.scalars(select(AIFeedbackObservation).where(
        AIFeedbackObservation.version_id == version_id, AIFeedbackObservation.taxonomy_category == need_code)
        .order_by(AIFeedbackObservation.original_feedback_id, AIFeedbackObservation.observation_index)))
    by_type: dict[str, set[str]] = {}
    phrases: dict[str, set[str]] = {}
    for o in obs:
        by_type.setdefault(o.observation_type, set()).add(o.original_feedback_id)
        key = " ".join(o.customer_need.lower().split())
        phrases.setdefault(key, set()).add(o.original_feedback_id)
    top = sorted(phrases.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]
    examples = [o for o in obs if o.observation_type in ("COMPLAINT", "IMPROVEMENT_REQUEST")][:limit]
    texts = dict(session.execute(select(Feedback.original_feedback_id, Feedback.customer_feedback).where(
        Feedback.original_feedback_id.in_({o.original_feedback_id for o in examples}), Feedback.vehicle_code == "G60")).all())
    return {"comments_by_type": {k: len(v) for k, v in by_type.items()},
            "top_needs": [{"customer_need": k, "comments": len(v)} for k, v in top],
            "examples": [{"feedback_id": o.original_feedback_id, "observation_type": o.observation_type,
                          "customer_need": o.customer_need, "evidence_excerpt": o.evidence_excerpt,
                          "original_comment": texts.get(o.original_feedback_id)} for o in examples]}


def sample_review_rows(session: Session, job_id: str) -> list[dict]:
    """Observations produced for one job's comments (for extraction-quality review)."""
    job = session.get(AIProcessingJob, job_id)
    if job is None:
        return []
    ids = json.loads(job.selection_json)
    rows = []
    for o in ai.observations_for(session, ids):
        rows.append({"feedback_id": o.original_feedback_id, "type": o.observation_type, "category": o.taxonomy_category,
                     "customer_need": o.customer_need, "excerpt": o.evidence_excerpt, "confidence": o.confidence})
    return rows


def has_credentials(settings: LLMSettings | None = None) -> bool:
    """Credential present for the CONFIGURED provider only (no fallback)."""
    from core.config import get_llm_settings
    from services.llm_client import credential_present

    return credential_present(settings or get_llm_settings())


def default_client(settings: LLMSettings):
    """Claude client for approval-gated paid actions started from the UI. The
    UI never builds a client itself; every paid entry point (sample
    extraction, semantic matching) re-checks authorisation, approval and the
    budget before the first request."""
    if not settings.external_calls_authorized:
        raise ApprovalRequired("External LLM processing is not authorised (set PRODUCTMIND_ALLOW_EXTERNAL_LLM=1).")
    from services.llm_client import ProviderNotConfigured, make_llm_client

    try:
        return make_llm_client(settings)
    except ProviderNotConfigured as exc:
        raise ApprovalRequired(str(exc)) from exc
