"""Application LLM budget and cost control (ProductMind AI 2.0).

* Budget: ``LLMSettings.budget_usd`` (default $26, ``PRODUCTMIND_LLM_BUDGET_USD``).
* Spend: the sum of ``ai_api_calls.estimated_cost_usd`` - every attempt is
  logged, including retries, refusals and invalid outputs, so retry cost is
  visible. Costs are list-price estimates from token usage, not an invoice.
* Before a paid batch: ``estimate_observation_cost`` (cached comments cost
  nothing). Once a sample has run, the estimate uses the measured tokens per
  comment for the same prompt version instead of the character heuristic.
* Approval: a paid run needs an explicit ``approved_max_usd``; it must cover
  the estimate and fit in the remaining budget (the upper bound with retries
  is shown; the per-batch guard below is the hard ceiling).
* Ceiling: ``batch_guard`` re-checks before every batch and stops the job
  (resumable, completed batches kept) if the next batch could exceed the
  approved amount or the application budget.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.ai_models import AIApiCall, AIProcessingJob
from core.config import LLMSettings
from services.ai_feedback_analysis import CommentInput, ExtractionSpec, ExtractionStopped, _user_message, is_cached
from services.llm_client import LLMUsage, estimate_cost_usd

RETRY_MARGIN = 1.5  # upper bound covers retries / splits
HEURISTIC_OUTPUT_TOKENS_PER_COMMENT = 220  # ~2 observations of ~80 tokens + JSON overhead (no thinking)
SAMPLE_SIZE = 20


class BudgetExceeded(ExtractionStopped):
    pass


class ApprovalRequired(ExtractionStopped):
    pass


@dataclass
class BudgetStatus:
    budget_usd: float
    spent_usd: float
    remaining_usd: float
    api_calls: int
    retry_or_failed_calls: int
    retry_or_failed_cost_usd: float


@dataclass
class CostEstimate:
    comments: int
    cached: int
    to_send: int
    batches: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_usd: float
    upper_bound_usd: float
    basis: str  # "measured from N comments" or "character heuristic"


def budget_status(session: Session, settings: LLMSettings) -> BudgetStatus:
    spent = float(session.scalar(select(func.coalesce(func.sum(AIApiCall.estimated_cost_usd), 0.0))) or 0.0)
    calls = session.scalar(select(func.count()).select_from(AIApiCall)) or 0
    retry_filter = (AIApiCall.attempt > 1) | (AIApiCall.status != "succeeded")
    retry_calls = session.scalar(select(func.count()).select_from(AIApiCall).where(retry_filter)) or 0
    retry_cost = float(session.scalar(select(func.coalesce(func.sum(AIApiCall.estimated_cost_usd), 0.0))
                                      .where(retry_filter)) or 0.0)
    return BudgetStatus(budget_usd=round(settings.budget_usd, 2), spent_usd=round(spent, 4),
                        remaining_usd=round(settings.budget_usd - spent, 4), api_calls=calls,
                        retry_or_failed_calls=retry_calls, retry_or_failed_cost_usd=round(retry_cost, 4))


def measured_tokens_per_comment(session: Session, prompt_version: str, model: str) -> tuple[float, float, int] | None:
    """(input, output) tokens per successfully processed comment, measured from
    earlier jobs with the same prompt version and model; None if none ran."""
    row = session.execute(
        select(func.sum(AIProcessingJob.input_tokens + AIProcessingJob.cache_write_tokens),
               func.sum(AIProcessingJob.output_tokens),
               func.sum(AIProcessingJob.processed_items - AIProcessingJob.cached_items))
        .where(AIProcessingJob.prompt_version == prompt_version, AIProcessingJob.model == model)
    ).one()
    inp, out, n = row
    if not n or n <= 0:
        return None
    return inp / n, out / n, int(n)


def estimate_cost(session: Session, comments: list[CommentInput], spec: ExtractionSpec, settings: LLMSettings,
                  model: str | None = None) -> CostEstimate:
    model = model or settings.model
    todo = [c for c in comments if not is_cached(session, c, model, spec)]
    batches = -(-len(todo) // max(1, settings.batch_size)) if todo else 0
    system_tokens = len(spec.build_system()) // 4
    measured = measured_tokens_per_comment(session, spec.prompt_version, model)
    if measured:
        per_in, per_out, n = measured
        inp, out = int(per_in * len(todo)), int(per_out * len(todo))
        usage = LLMUsage(input_tokens=inp, output_tokens=out)
        basis = f"measured from {n} processed comments"
    else:
        inp = sum(len(_user_message([c], spec.instruction)) for c in todo) // 4 + batches * 60
        usage = LLMUsage(input_tokens=inp, output_tokens=len(todo) * HEURISTIC_OUTPUT_TOKENS_PER_COMMENT,
                         cache_write_tokens=system_tokens if batches else 0,
                         cache_read_tokens=system_tokens * max(0, batches - 1))
        basis = "character heuristic (no sample measured yet)"
    cost = estimate_cost_usd(model, usage)
    return CostEstimate(comments=len(comments), cached=len(comments) - len(todo), to_send=len(todo), batches=batches,
                        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                        cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
                        estimated_usd=round(cost, 4), upper_bound_usd=round(cost * RETRY_MARGIN, 4), basis=basis)


def check_approval(session: Session, estimate: CostEstimate, settings: LLMSettings, approved_max_usd: float | None) -> None:
    if estimate.to_send == 0:
        return
    if not settings.external_calls_authorized:
        raise ApprovalRequired("External LLM processing is not authorised (set PRODUCTMIND_ALLOW_EXTERNAL_LLM=1).")
    if approved_max_usd is None:
        raise ApprovalRequired(
            f"Paid batch needs explicit approval: estimate ${estimate.estimated_usd:.2f} "
            f"(up to ${estimate.upper_bound_usd:.2f} with retries) for {estimate.to_send} comments.")
    if approved_max_usd < estimate.estimated_usd:
        raise ApprovalRequired(
            f"Approved ${approved_max_usd:.2f} is below the estimate ${estimate.estimated_usd:.2f}.")
    status = budget_status(session, settings)
    if approved_max_usd > status.remaining_usd:
        raise BudgetExceeded(
            f"Approved ${approved_max_usd:.2f} exceeds the remaining budget ${status.remaining_usd:.2f} "
            f"(budget ${status.budget_usd:.2f}, spent ${status.spent_usd:.2f}).")


def batch_guard(settings: LLMSettings, approved_max_usd: float, spec: ExtractionSpec, model: str):
    """``before_batch`` hook for ``run_extraction``: stops before a batch whose
    upper-bound cost would exceed the approval or the application budget."""
    def guard(session: Session, job: AIProcessingJob, batch: list[CommentInput]) -> None:
        est = estimate_cost(session, batch, spec, settings, model)
        job_spent = job.estimated_cost_usd
        if job_spent + est.upper_bound_usd > approved_max_usd:
            raise BudgetExceeded(f"Stopped: next batch (up to ${est.upper_bound_usd:.2f}) would exceed the approved "
                                 f"${approved_max_usd:.2f} (job spent ${job_spent:.2f}). Resume with a new approval.")
        status = budget_status(session, settings)
        if est.upper_bound_usd > status.remaining_usd:
            raise BudgetExceeded(f"Stopped: next batch (up to ${est.upper_bound_usd:.2f}) would exceed the remaining "
                                 f"application budget ${status.remaining_usd:.2f}.")
    return guard
