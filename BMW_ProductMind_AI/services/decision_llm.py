"""Optional LLM wording layer for decision answers (Part 6) - DISABLED by default.

The assistant always produces a validated, structured offline answer first.
An LLM may only *rephrase* that answer. It runs only when all hold:
external processing authorised (PRODUCTMIND_ALLOW_EXTERNAL_LLM), credentials
available (or a registered local model), a visible cost estimate, a
user-initiated action and explicit confirmation. Server-side fallback to
another model is forced off. Rephrased text containing numbers or IDs that are
not in the structured answer is rejected.

``DecisionLanguageModel`` is a small protocol so a local model can be plugged in
later via ``register_local_model`` (no data leaves the machine in that case,
but confirmation is still required).
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from pydantic import BaseModel

from core.config import LLMSettings, get_llm_settings
from services.decision_grounding import AssistantResponse

ID_RE = re.compile(r"\b(?:G60|F70|G70)-\d{4}\b|\bP\d{3}\b|\bCOMP-R\d+\b|\b(?:REQ|RUN|AGT|ASM|PRF|CHL|SCN)-[A-Z0-9-]+\b")
NUM_RE = re.compile(r"(?<![A-Za-z0-9.,-])\d+(?:[.,]\d+)?")


class DecisionLanguageModel(Protocol):
    name: str
    is_local: bool

    def rephrase(self, system: str, payload: str, max_tokens: int) -> str: ...


_LOCAL_MODEL: DecisionLanguageModel | None = None


def register_local_model(model: DecisionLanguageModel | None) -> None:
    """Hook for a future on-premise model."""
    global _LOCAL_MODEL
    _LOCAL_MODEL = model


class LLMModeStatus(BaseModel):
    enabled: bool
    provider: str | None
    reasons: list[str]


def llm_mode_status(settings: LLMSettings | None = None) -> LLMModeStatus:
    settings = settings or get_llm_settings()
    if _LOCAL_MODEL is not None:
        return LLMModeStatus(enabled=True, provider=f"local:{_LOCAL_MODEL.name}", reasons=[])
    reasons = []
    if not settings.external_calls_authorized:
        reasons.append("External data processing is not authorised (PRODUCTMIND_ALLOW_EXTERNAL_LLM is not set). "
                       "Internal BMW feedback stays local.")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        reasons.append("No Anthropic API credentials configured (a ChatGPT subscription does not provide Anthropic API "
                       "credits).")
    return LLMModeStatus(enabled=not reasons, provider=None if reasons else f"anthropic:{settings.model}", reasons=reasons)


_SYSTEM = ("Rephrase the structured decision explanation for a BMW product manager in clear prose. Use ONLY the "
           "statements provided. Keep every number and ID exactly as given; add no numbers, IDs, sources, claims or "
           "recommendations. Keep the FACT/ASSESSMENT/ASSUMPTION/LIMITATION/SIMULATION distinction visible.")


def _payload(resp: AssistantResponse) -> str:
    return "\n".join(f"[{i.kind}] {i.text}" for i in resp.items)


def estimate_cost(resp: AssistantResponse, settings: LLMSettings | None = None) -> dict:
    from services.llm_client import LLMUsage, estimate_cost_usd

    settings = settings or get_llm_settings()
    usage = LLMUsage(input_tokens=(len(_SYSTEM) + len(_payload(resp))) // 4, output_tokens=800)
    return {"model": settings.model, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "estimated_cost_usd": estimate_cost_usd(settings.model, usage)}


def check_rephrasing(text: str, resp: AssistantResponse) -> list[str]:
    source = _payload(resp) + " " + resp.title + " " + " ".join(r["key"] for r in resp.evidence_refs)
    allowed_nums, allowed_ids = set(NUM_RE.findall(source)), set(ID_RE.findall(source))
    errors = [f"number not in the answer: {n}" for n in sorted(set(NUM_RE.findall(ID_RE.sub(" ", text))) - allowed_nums)]
    errors += [f"ID not in the answer: {i}" for i in sorted(set(ID_RE.findall(text)) - allowed_ids)]
    return errors


class _AnthropicRephraser:
    is_local = False

    def __init__(self, settings: LLMSettings):
        from services.llm_client import AnthropicLLMClient

        self._client = AnthropicLLMClient(settings.model_copy(update={"use_fallbacks": False}))
        self.name = settings.model

    def rephrase(self, system: str, payload: str, max_tokens: int) -> str:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
                  "additionalProperties": False}
        import json

        r = self._client.complete_json(system=system, user=payload, schema=schema, max_tokens=max_tokens)
        if r.stop_reason == "refusal" or not r.text:
            raise RuntimeError("The model returned no usable text.")
        return json.loads(r.text)["text"]


def rephrase(resp: AssistantResponse, *, confirmed: bool, cost_acknowledged: dict | None,
             model: DecisionLanguageModel | None = None, settings: LLMSettings | None = None) -> dict:
    """Returns {"text", "provider", "errors"}; raises PermissionError if not allowed."""
    settings = (settings or get_llm_settings()).model_copy(update={"use_fallbacks": False})
    status = llm_mode_status(settings)
    model = model or _LOCAL_MODEL
    if model is None:
        if not status.enabled:
            raise PermissionError("; ".join(status.reasons))
        model = _AnthropicRephraser(settings)
    elif not model.is_local and not settings.external_calls_authorized:
        raise PermissionError("External data processing is not authorised.")
    if not confirmed or cost_acknowledged is None:
        raise PermissionError("Review the cost estimate and confirm before generating.")
    text = model.rephrase(_SYSTEM, _payload(resp), 2000)
    errors = check_rephrasing(text, resp)
    return {"text": None if errors else text, "provider": model.name, "errors": errors}
