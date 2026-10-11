"""One-request provider connection test (Stage 1 - OpenAI).

Sends a single minimal structured-output request that contains NO BMW data,
validates the reply with Pydantic, and logs the attempt in ``ai_api_calls``
under a ``connection_test`` job so it counts against the application budget.
Nothing is faked: success is reported only when the API answered with valid
structured output.
"""
from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from core.ai_models import AIProcessingJob
from core.config import LLMSettings
from services import ai_feedback_analysis as ai
from services.llm_budget import budget_status
from services.llm_client import LLMClient, LLMError, LLMUsage, estimate_cost_usd

PROMPT_VERSION = "connection-test-v1"
SYSTEM = "You are a connection test for a software application. Reply only with the requested JSON."
SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["ok"]}, "echo": {"type": "string"}},
    "required": ["status", "echo"],
    "additionalProperties": False,
}
ESTIMATE_USAGE = LLMUsage(input_tokens=120, output_tokens=300)  # generous; includes possible reasoning tokens
MAX_OUTPUT_TOKENS = 500


class _Reply(BaseModel):
    status: str
    echo: str


@dataclass
class ConnectionResult:
    success: bool
    provider: str
    model_requested: str
    model_reported: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    estimated_cost_usd: float = 0.0
    request_id: str | None = None
    job_id: str | None = None
    error: str | None = None


def estimated_test_cost(settings: LLMSettings) -> float:
    return estimate_cost_usd(settings.model, ESTIMATE_USAGE)


def run_connection_test(session: Session, settings: LLMSettings, llm: LLMClient) -> ConnectionResult:
    """Caller must have obtained explicit confirmation. Checks the budget,
    sends ONE request, validates it and logs usage/cost."""
    estimate = estimated_test_cost(settings)
    remaining = budget_status(session, settings).remaining_usd
    if estimate > remaining:
        return ConnectionResult(False, settings.provider, settings.model,
                                error=f"Estimated ${estimate:.4f} exceeds the remaining budget ${remaining:.4f}.")
    nonce = f"PING-{secrets.token_hex(4).upper()}"
    job = AIProcessingJob(
        job_id=f"JOB-{ai._now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}", job_type="connection_test", status="running",
        vehicle_code="-", model=settings.model, prompt_version=PROMPT_VERSION, selection_json="[]",
        selection_description=f"{settings.provider} connection test (no BMW data)", batch_size=1, total_items=1,
        processed_items=0, cached_items=0, failed_items=0, api_calls=0, input_tokens=0, output_tokens=0,
        cache_read_tokens=0, cache_write_tokens=0, estimated_cost_usd=0.0, created_at=ai._now(), updated_at=ai._now())
    session.add(job)
    session.flush()
    result = ConnectionResult(False, settings.provider, settings.model, job_id=job.job_id)
    try:
        resp = llm.complete_json(system=SYSTEM, user=f'Return status "ok" and echo exactly: {nonce}',
                                 schema=SCHEMA, max_tokens=MAX_OUTPUT_TOKENS)
    except LLMError as exc:
        ai._log_call(session, job, 0, 1, [], "rate_limited" if exc.kind == "rate_limited" else "api_error",
                     settings.model, exc.usage, error=str(exc)[:500])
        result.error = f"{exc.kind}: {exc}"
    else:
        result.model_reported, result.usage, result.request_id = resp.model, resp.usage, resp.request_id
        status, error = "succeeded", None
        try:
            reply = _Reply.model_validate_json(resp.text or "")
            if resp.stop_reason != "end_turn":
                error = f"Response stopped with {resp.stop_reason}."
            elif reply.status != "ok" or reply.echo.strip() != nonce:
                error = "Response did not match the requested content."
        except ValidationError:
            error = f"Response was not valid structured JSON (stop_reason={resp.stop_reason})."
        if error:
            status = "invalid_output"
        result.estimated_cost_usd = ai._log_call(session, job, 0, 1, [], status, resp.model, resp.usage,
                                                 request_id=resp.request_id, stop_reason=resp.stop_reason,
                                                 latency_ms=resp.latency_ms, error=error)
        result.success, result.error = error is None, error
    job.status = "completed" if result.success else "failed"
    job.processed_items, job.failed_items = (1, 0) if result.success else (0, 1)
    job.last_error, job.updated_at = result.error, ai._now()
    session.flush()
    return result


def schema_is_strict(schema: dict) -> list[str]:
    """Problems that OpenAI strict structured outputs would reject: every
    object needs additionalProperties=false and all properties required."""
    problems: list[str] = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "object":
                props = node.get("properties", {})
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: additionalProperties must be false")
                if set(node.get("required", [])) != set(props):
                    problems.append(f"{path}: all properties must be required")
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(schema, "$")
    if schema.get("type") != "object":
        problems.append("$: root must be an object")
    return problems


def describe(result: ConnectionResult) -> str:
    u = result.usage
    return json.dumps({"success": result.success, "provider": result.provider, "model_requested": result.model_requested,
                       "model_reported": result.model_reported, "input_tokens_uncached": u.input_tokens,
                       "cached_input_tokens": u.cache_read_tokens, "output_tokens": u.output_tokens,
                       "reasoning_tokens_in_output": u.reasoning_tokens,
                       "estimated_cost_usd": result.estimated_cost_usd, "error": result.error}, indent=1)
