"""Thin LLM client used by the AI services.

``LLMClient`` is a small protocol so the pipeline can be tested with a fake
client (no network, no cost). ``AnthropicLLMClient`` calls the Claude API with
structured JSON output (``output_config.format``), a cached system prompt, and
the server-side refusal fallback. Errors are mapped to ``LLMError`` with a
``retryable`` flag; the batch runner owns retry/backoff decisions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.config import LLM_PRICING_PER_MTOK, LLMSettings


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMResponse:
    text: str | None
    stop_reason: str | None
    model: str
    request_id: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: int | None = None


class LLMError(Exception):
    def __init__(self, message: str, *, kind: str, retryable: bool, retry_after_s: float | None = None,
                 usage: LLMUsage | None = None):
        super().__init__(message)
        self.kind = kind  # rate_limited | api_error | connection | bad_request | auth
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.usage = usage or LLMUsage()


class LLMClient(Protocol):
    model: str

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> LLMResponse: ...


def estimate_cost_usd(model: str, usage: LLMUsage) -> float:
    """List-price estimate; unknown models fall back to the configured default's price row."""
    price = LLM_PRICING_PER_MTOK.get(model) or LLM_PRICING_PER_MTOK["claude-opus-5-5"]
    return round(
        (usage.input_tokens * price["input"] + usage.output_tokens * price["output"]
         + usage.cache_read_tokens * price["cache_read"] + usage.cache_write_tokens * price["cache_write"]) / 1_000_000,
        6,
    )


class AnthropicLLMClient:
    """Claude API client. Credentials are resolved by the SDK (env or profile)."""

    def __init__(self, settings: LLMSettings, client: Any | None = None):
        import anthropic  # imported lazily so tests and offline commands never need it

        self._anthropic = anthropic
        self.settings = settings
        self.model = settings.model
        # SDK-level retries off: the batch runner handles retries and logs each attempt.
        self._client = client or anthropic.Anthropic(max_retries=0)

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> LLMResponse:
        a = self._anthropic
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Stable instructions first and cached; per-batch comments go in the user turn.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.settings.effort, "format": {"type": "json_schema", "schema": schema}},
        }
        started = time.monotonic()
        try:
            if self.settings.use_fallbacks:
                response = self._client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                response = self._client.messages.create(**params)
        except a.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise LLMError(str(exc), kind="rate_limited", retryable=True,
                           retry_after_s=float(retry_after) if retry_after else None) from exc
        except (a.AuthenticationError, a.PermissionDeniedError) as exc:
            raise LLMError(str(exc), kind="auth", retryable=False) from exc
        except a.APIStatusError as exc:
            retryable = exc.status_code >= 500 or exc.status_code in (408, 409)
            raise LLMError(str(exc), kind="api_error" if retryable else "bad_request", retryable=retryable) from exc
        except a.APIConnectionError as exc:  # includes timeouts
            raise LLMError(str(exc), kind="connection", retryable=True) from exc

        usage = response.usage
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return LLMResponse(
            text=text or None,
            stop_reason=response.stop_reason,
            model=getattr(response, "model", self.model) or self.model,
            request_id=getattr(response, "_request_id", None),
            usage=LLMUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
