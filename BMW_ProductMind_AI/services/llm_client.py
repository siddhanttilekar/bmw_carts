"""Thin LLM client used by the AI services.

``LLMClient`` is a small provider-neutral protocol (``complete_json``) so the
pipeline can be tested with a fake client (no network, no cost) and any
feature - customer intelligence, custom AI agents, requirement generation,
decision assistant, agent debate - can use the same interface.

* ``OpenAILLMClient`` (default provider) - OpenAI Responses API with strict
  JSON-schema structured output and configurable reasoning effort.
* ``AnthropicLLMClient`` - kept, used ONLY when PRODUCTMIND_LLM_PROVIDER=anthropic.

``make_llm_client`` builds exactly the configured provider; there is never a
fallback to the other provider. Errors are mapped to ``LLMError`` with a
``retryable`` flag; the batch runner owns retry/backoff decisions. SDK-level
retries are disabled so every attempt is logged and costed.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.config import CREDENTIAL_ENV, LLM_PRICING_PER_MTOK, LLM_PROVIDERS, UNKNOWN_MODEL_PRICE, LLMSettings


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0  # informational (already included in output_tokens for OpenAI)


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


def price_row(model: str) -> dict[str, float]:
    """Exact model ID, else the longest priced ID the (dated snapshot) name
    starts with, else the most expensive row (conservative)."""
    if model in LLM_PRICING_PER_MTOK:
        return LLM_PRICING_PER_MTOK[model]
    prefixes = [k for k in LLM_PRICING_PER_MTOK if model.startswith(k + "-")]
    return LLM_PRICING_PER_MTOK[max(prefixes, key=len)] if prefixes else LLM_PRICING_PER_MTOK[UNKNOWN_MODEL_PRICE]


def estimate_cost_usd(model: str, usage: LLMUsage) -> float:
    """List-price ESTIMATE from token usage (not provider billing)."""
    price = price_row(model)
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


class OpenAILLMClient:
    """OpenAI Responses API client. The key is read by the SDK from
    OPENAI_API_KEY and never stored or logged here.

    Usage mapping: OpenAI ``input_tokens`` include cached tokens, so
    ``LLMUsage.input_tokens`` = uncached input and ``cache_read_tokens`` =
    cached input; ``output_tokens`` include reasoning tokens (billed as output).
    """

    provider = "openai"

    def __init__(self, settings: LLMSettings, client: Any | None = None):
        try:
            import openai  # imported lazily so tests and offline commands never need it
        except ImportError as exc:
            raise ProviderNotConfigured(
                "The 'openai' package is not installed in this Python environment. Use the project's virtual "
                "environment (.venv/bin/python -m ...) or run: pip install -r requirements.txt") from exc

        self._openai = openai
        self.settings = settings
        self.model = settings.model
        self._client = client or openai.OpenAI(max_retries=0)

    def _params(self, system: str, user: str, schema: dict[str, Any], max_tokens: int,
                name: str = "structured_output") -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "instructions": system,  # stable prefix -> OpenAI automatic prompt caching
            "input": user,
            "max_output_tokens": max_tokens,
            "store": False,  # do not retain BMW data in OpenAI response storage
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        }
        if self.settings.effort:
            params["reasoning"] = {"effort": self.settings.effort}
        return params

    def verify_model(self) -> str:
        """Model lookup (GET /v1/models/{id}; not a billed generation request).
        Raises LLMError if the model is unavailable to this key."""
        o = self._openai
        try:
            return self._client.models.retrieve(self.model).id
        except o.NotFoundError as exc:
            raise LLMError(f"Model {self.model!r} is not available to this API key.", kind="bad_request",
                           retryable=False) from exc
        except (o.AuthenticationError, o.PermissionDeniedError) as exc:
            raise LLMError(_safe(exc), kind="auth", retryable=False) from exc
        except o.APIConnectionError as exc:
            raise LLMError(_safe(exc), kind="connection", retryable=True) from exc
        except o.APIStatusError as exc:
            raise LLMError(_safe(exc), kind="api_error", retryable=exc.status_code >= 500) from exc

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> LLMResponse:
        o = self._openai
        started = time.monotonic()
        try:
            response = self._client.responses.create(**self._params(system, user, schema, max_tokens))
        except o.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if getattr(exc, "response", None) is not None else None
            raise LLMError(_safe(exc), kind="rate_limited", retryable=True,
                           retry_after_s=float(retry_after) if retry_after else None) from exc
        except (o.AuthenticationError, o.PermissionDeniedError) as exc:
            raise LLMError(_safe(exc), kind="auth", retryable=False) from exc
        except o.APIStatusError as exc:
            retryable = exc.status_code >= 500 or exc.status_code in (408, 409)
            raise LLMError(_safe(exc), kind="api_error" if retryable else "bad_request", retryable=retryable) from exc
        except o.APIConnectionError as exc:  # includes timeouts
            raise LLMError(_safe(exc), kind="connection", retryable=True) from exc
        return _openai_response(response, self.model, int((time.monotonic() - started) * 1000))


def _safe(exc: Exception) -> str:
    """Error text without request bodies (no customer feedback in logs)."""
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or exc.__class__.__name__
    return f"{exc.__class__.__name__}{f' {status}' if status else ''}: {str(message)[:300]}"


def _openai_response(response: Any, model: str, latency_ms: int | None) -> LLMResponse:
    usage = getattr(response, "usage", None)
    total_in = getattr(usage, "input_tokens", 0) or 0
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    text_parts, refused = [], False
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            kind = getattr(part, "type", None)
            if kind == "output_text":
                text_parts.append(part.text)
            elif kind == "refusal":
                refused = True
    status = getattr(response, "status", None)
    reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
    if refused:
        stop = "refusal"
    elif status == "incomplete" and reason == "max_output_tokens":
        stop = "max_tokens"
    elif status == "incomplete" and reason == "content_filter":
        stop = "refusal"
    else:
        stop = "end_turn" if status in (None, "completed") else str(status)
    return LLMResponse(
        text="".join(text_parts) or None, stop_reason=stop, model=getattr(response, "model", None) or model,
        request_id=getattr(response, "_request_id", None) or getattr(response, "id", None),
        usage=LLMUsage(input_tokens=max(total_in - cached, 0), output_tokens=getattr(usage, "output_tokens", 0) or 0,
                       cache_read_tokens=cached, cache_write_tokens=0,
                       reasoning_tokens=getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Provider selection (single provider, no cross-provider fallback)
# ---------------------------------------------------------------------------


class ProviderNotConfigured(RuntimeError):
    pass


def credential_present(settings: LLMSettings) -> bool:
    if settings.provider == "anthropic":
        from pathlib import Path

        return any(os.getenv(v) for v in CREDENTIAL_ENV["anthropic"]) or (Path.home() / ".config" / "anthropic").exists()
    return any(os.getenv(v) for v in CREDENTIAL_ENV.get(settings.provider, ()))


def setup_instructions(settings: LLMSettings) -> str:
    var = CREDENTIAL_ENV.get(settings.provider, ("OPENAI_API_KEY",))[0]
    return (f"No API key for provider '{settings.provider}'. Set it in your shell (never in code or committed files):\n"
            f"  export {var}=...\n  export PRODUCTMIND_LLM_PROVIDER={settings.provider}\n"
            "  export PRODUCTMIND_ALLOW_EXTERNAL_LLM=1\n  export PRODUCTMIND_MAX_API_SPEND_USD=26")


def make_llm_client(settings: LLMSettings) -> LLMClient:
    """Client for the configured provider only. Raises ProviderNotConfigured
    instead of switching providers."""
    if settings.provider not in LLM_PROVIDERS:
        raise ProviderNotConfigured(f"Unknown LLM provider {settings.provider!r}; choose one of {', '.join(LLM_PROVIDERS)}.")
    if not credential_present(settings):
        raise ProviderNotConfigured(setup_instructions(settings))
    if settings.provider == "openai":
        return OpenAILLMClient(settings)
    return AnthropicLLMClient(settings)
