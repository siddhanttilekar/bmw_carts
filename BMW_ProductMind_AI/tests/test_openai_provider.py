"""Stage 1 - OpenAI provider tests. The OpenAI SDK client is ALWAYS replaced by
a local fake (no network, no cost); real ``openai`` exception classes are used
so error mapping is tested against the SDK."""
from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import httpx2 as httpx  # the transport used by openai>=3
import openai
import pytest
from sqlalchemy import select

from core.ai_models import AIApiCall
from core.config import LLM_PRICING_PER_MTOK, LLMSettings, get_llm_settings
from core.database import get_engine, session_scope
from services import ai_evidence_selector as sel
from services import ai_feedback_analysis as ai
from services import evidence_extraction as ee
from services import llm_budget as budget
from services import llm_connection as conn
from services import need_clustering, requirement_generator, insight_generator
from services import requirement_review as rr
from services.agent_scoring import log_normalize
from services.llm_client import (
    AnthropicLLMClient,
    LLMError,
    LLMUsage,
    OpenAILLMClient,
    ProviderNotConfigured,
    estimate_cost_usd,
    make_llm_client,
)
from tests.test_evidence_first import USER, base_db, category_map, kind  # noqa: F401  (fixture re-export)

MODEL = "gpt-5.6-luna"
REQ = httpx.Request("POST", "https://api.openai.com/v1/responses")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "PRODUCTMIND_LLM_PROVIDER",
                "PRODUCTMIND_LLM_MODEL", "PRODUCTMIND_LLM_EFFORT", "PRODUCTMIND_MAX_API_SPEND_USD",
                "PRODUCTMIND_LLM_BUDGET_USD", "PRODUCTMIND_ALLOW_EXTERNAL_LLM", "PRODUCTMIND_LLM_MODEL_VERSION"):
        monkeypatch.delenv(var, raising=False)

    def forbid(*a, **k):
        raise AssertionError("a real SDK client must never be created in tests")

    monkeypatch.setattr("openai.OpenAI.__init__", forbid)
    monkeypatch.setattr("anthropic.Anthropic.__init__", forbid, raising=False)


def settings(**kw) -> LLMSettings:
    return LLMSettings(**{**dict(provider="openai", model=MODEL, model_version=MODEL, effort="low", batch_size=10,
                                 max_attempts=3, external_calls_authorized=True, budget_usd=26.0), **kw})


def status_error(cls, code, headers=None):
    return cls("boom", response=httpx.Response(code, request=REQ, headers=headers or {}), body=None)


def fake_response(text, *, status="completed", reason=None, refusal=False, inp=1000, cached=600, out=300, reasoning=40):
    content = [SimpleNamespace(type="refusal", refusal="no")] if refusal else [SimpleNamespace(type="output_text", text=text)]
    return SimpleNamespace(
        id="resp_test", model=MODEL + "-2026-09-01", status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(type="message", content=content)],
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                              input_tokens_details=SimpleNamespace(cached_tokens=cached),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)))


class FakeSDK:
    """Stand-in for ``openai.OpenAI``: records requests, replays scripted outcomes."""

    def __init__(self, outcomes=None, builder=None):
        self.requests: list[dict] = []
        self.outcomes = list(outcomes or [])
        self.builder = builder
        self.responses = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=lambda model: SimpleNamespace(id=model))

    def _create(self, **params):
        self.requests.append(params)
        if self.outcomes:
            o = self.outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return o
        return self.builder(params)


def observation_builder(category_for):
    """Valid observe-v1 output built from the comments in the request."""
    def build(params):
        payload = json.loads(params["input"].split("\n\n", 1)[1])
        results = []
        for p in payload:
            text, fid = p["comment"], p["feedback_id"]
            obs = [{"observation_type": "COMPLAINT" if kind(fid) in (0, 1) else "POSITIVE",
                    "customer_need": "need", "vehicle_system": "system",
                    "evidence_excerpt": text[: max(1, min(30, len(text) // 2))], "confidence": "high",
                    "taxonomy_category": category_for(fid)}]
            if kind(fid) in (1, 3):
                obs.append({"observation_type": "IMPROVEMENT_REQUEST", "customer_need": "change",
                            "vehicle_system": "system", "evidence_excerpt": text[len(text) // 2:],
                            "confidence": "medium", "taxonomy_category": category_for(fid)})
            results.append({"feedback_id": fid, "observations": obs})
        return fake_response(json.dumps({"results": results}))
    return build


@pytest.fixture()
def s(base_db, tmp_path):  # noqa: F811
    path = tmp_path / "o.db"
    shutil.copy2(base_db, path)
    engine = get_engine(path)
    with session_scope(engine) as session:
        yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# configuration and provider selection
# ---------------------------------------------------------------------------


def test_openai_is_default_provider_and_reads_env(monkeypatch):
    st = get_llm_settings()
    assert (st.provider, st.model, st.effort, st.budget_usd) == ("openai", "gpt-5.6-luna", "low", 26.0)
    assert st.external_calls_authorized is False and st.use_fallbacks is False
    monkeypatch.setenv("PRODUCTMIND_MAX_API_SPEND_USD", "12.5")
    monkeypatch.setenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("PRODUCTMIND_LLM_PROVIDER", "openai")
    st = get_llm_settings()
    assert st.budget_usd == 12.5 and st.external_calls_authorized
    assert LLM_PRICING_PER_MTOK[MODEL] == {"input": 0.20, "output": 1.20, "cache_read": 0.02, "cache_write": 0.20}


def test_missing_api_key_gives_setup_instructions(monkeypatch, capsys):
    with pytest.raises(ProviderNotConfigured) as exc:
        make_llm_client(settings())
    assert "export OPENAI_API_KEY=" in str(exc.value)
    from scripts.customer_intelligence import main

    with pytest.raises(SystemExit) as ex:
        main(["test-openai"])
    assert "OPENAI_API_KEY" in str(ex.value.code)
    assert "MISSING" in capsys.readouterr().out
    with pytest.raises(budget.ApprovalRequired):
        ee.default_client(settings())


def test_provider_selection_without_anthropic_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-used")
    with pytest.raises(ProviderNotConfigured):  # an Anthropic key never rescues a missing OpenAI key
        make_llm_client(settings())
    monkeypatch.setattr("services.llm_client.AnthropicLLMClient.__init__", lambda self, st: None)
    assert isinstance(make_llm_client(settings(provider="anthropic", model="claude-opus-5-5")), AnthropicLLMClient)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    monkeypatch.setattr("services.llm_client.OpenAILLMClient.__init__", lambda self, st: setattr(self, "model", st.model))
    assert isinstance(make_llm_client(settings()), OpenAILLMClient)
    with pytest.raises(ProviderNotConfigured):
        make_llm_client(settings(provider="other"))


# ---------------------------------------------------------------------------
# request shape, structured output parsing, usage, errors
# ---------------------------------------------------------------------------


def test_structured_output_request_and_parsing():
    sdk = FakeSDK(builder=observation_builder(lambda f: "CONTROLS"))
    llm = OpenAILLMClient(settings(), client=sdk)
    c = ai.CommentInput("rec", "123", "G60", "The touchscreen is slow and I wish it had buttons.", None, [])
    resp = llm.complete_json(system="sys", user=ai._user_message([c], ai.OBSERVATION_INSTRUCTION),
                             schema=ai.OBSERVATION_SCHEMA, max_tokens=900)
    req = sdk.requests[0]
    assert req["model"] == MODEL and req["instructions"] == "sys" and req["max_output_tokens"] == 900
    assert req["text"]["format"] == {"type": "json_schema", "name": "structured_output",
                                     "schema": ai.OBSERVATION_SCHEMA, "strict": True}
    assert req["reasoning"] == {"effort": "low"} and req["store"] is False
    parsed = ai.BatchObservationResult.model_validate_json(resp.text)
    assert parsed.results[0].feedback_id == "123"
    assert resp.stop_reason == "end_turn" and resp.model.startswith(MODEL)
    assert (resp.usage.input_tokens, resp.usage.cache_read_tokens, resp.usage.output_tokens,
            resp.usage.reasoning_tokens) == (400, 600, 300, 40)


def test_all_existing_schemas_are_strict_compatible():
    from services import decision_llm  # noqa: F401  (imports cleanly with the new client module)

    for schema in (ai.RESPONSE_SCHEMA, ai.OBSERVATION_SCHEMA, sel.SEMANTIC_SCHEMA, need_clustering._GROUP_SCHEMA,
                   requirement_generator._LLM_SCHEMA, insight_generator._TEXT_SCHEMA, conn.SCHEMA):
        assert conn.schema_is_strict(schema) == []


def test_refusal_and_truncation_mapping():
    llm = OpenAILLMClient(settings(), client=FakeSDK([fake_response("", refusal=True),
                                                      fake_response('{"res', status="incomplete", reason="max_output_tokens"),
                                                      fake_response("", status="incomplete", reason="content_filter")]))
    stops = [llm.complete_json(system="s", user="u", schema=conn.SCHEMA, max_tokens=10).stop_reason for _ in range(3)]
    assert stops == ["refusal", "max_tokens", "refusal"]


def test_api_error_mapping_never_leaks_request_content():
    cases = [
        (status_error(openai.AuthenticationError, 401), "auth", False),
        (status_error(openai.PermissionDeniedError, 403), "auth", False),
        (status_error(openai.RateLimitError, 429, {"retry-after": "7"}), "rate_limited", True),
        (status_error(openai.InternalServerError, 500), "api_error", True),
        (status_error(openai.BadRequestError, 400), "bad_request", False),
        (openai.APIConnectionError(request=REQ), "connection", True),
        (openai.APITimeoutError(request=REQ), "connection", True),
    ]
    for exc, kind_, retryable in cases:
        llm = OpenAILLMClient(settings(), client=FakeSDK([exc]))
        with pytest.raises(LLMError) as err:
            llm.complete_json(system="s", user="SECRET CUSTOMER COMMENT", schema=conn.SCHEMA, max_tokens=10)
        assert (err.value.kind, err.value.retryable) == (kind_, retryable)
        assert "SECRET CUSTOMER COMMENT" not in str(err.value)
    llm = OpenAILLMClient(settings(), client=FakeSDK([status_error(openai.RateLimitError, 429, {"retry-after": "7"})]))
    with pytest.raises(LLMError) as err:
        llm.complete_json(system="s", user="u", schema=conn.SCHEMA, max_tokens=10)
    assert err.value.retry_after_s == 7.0


def test_openai_pricing_and_snapshot_names():
    usage = LLMUsage(input_tokens=1_000_000, cache_read_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost_usd(MODEL, usage) == pytest.approx(0.20 + 0.02 + 1.20)
    assert estimate_cost_usd(MODEL + "-2026-09-01", usage) == estimate_cost_usd(MODEL, usage)
    assert estimate_cost_usd("unknown-model", usage) > estimate_cost_usd(MODEL, usage)  # conservative fallback


# ---------------------------------------------------------------------------
# pipeline compatibility: extraction, retries, budget, selection, scoring
# ---------------------------------------------------------------------------


def test_extraction_retry_and_budget_accounting_with_openai(s):
    cats = category_map(s)
    ids = sorted(cats)[:12]
    sdk = FakeSDK([status_error(openai.InternalServerError, 500), status_error(openai.RateLimitError, 429)],
                  builder=observation_builder(lambda f: cats[f]))
    st = settings()
    job = ai.run_extraction(s, OpenAILLMClient(st, client=sdk), st, ids, spec=ai.observation_spec(st),
                            sleep=lambda _: None)
    assert job.status == "completed" and job.processed_items == 12
    calls = list(s.scalars(select(AIApiCall).where(AIApiCall.job_id == job.job_id).order_by(AIApiCall.call_id)))
    assert [c.status for c in calls][:2] == ["api_error", "rate_limited"]
    ok = [c for c in calls if c.status == "succeeded"]
    assert len(ok) == 2  # 12 comments, batch size 10
    one = estimate_cost_usd(MODEL, LLMUsage(input_tokens=400, cache_read_tokens=600, output_tokens=300))
    assert all(c.estimated_cost_usd == pytest.approx(one) for c in ok)
    b = budget.budget_status(s, st)
    assert b.spent_usd == pytest.approx(round(2 * one, 4), abs=1e-4)
    assert b.retry_or_failed_calls >= 2 and b.remaining_usd == pytest.approx(26 - b.spent_usd)
    # cache reuse: a second run sends nothing
    sent = len(sdk.requests)
    ai.run_extraction(s, OpenAILLMClient(st, client=sdk), st, ids, spec=ai.observation_spec(st), sleep=lambda _: None)
    assert len(sdk.requests) == sent
    # the estimate now uses OpenAI pricing and measured tokens
    est = budget.estimate_cost(s, ai.load_comments(s, sorted(cats)[12:32]), ai.observation_spec(st), st)
    assert est.basis.startswith("measured") and est.estimated_usd < 0.05


def test_openai_auth_failure_stops_without_fallback(s, monkeypatch):
    monkeypatch.setattr("services.llm_client.AnthropicLLMClient.__init__",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no Anthropic fallback")))
    cats = category_map(s)
    st = settings()
    sdk = FakeSDK([status_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMError):
        ai.run_extraction(s, OpenAILLMClient(st, client=sdk), st, sorted(cats)[:5], spec=ai.observation_spec(st),
                          sleep=lambda _: None)
    assert len(sdk.requests) == 1


def test_scoring_unchanged_with_openai_extraction(s, monkeypatch):
    from services import agent_configuration as ac
    from services import dynamic_prioritization as dp
    from services import prioritization_profiles as pp

    cats = category_map(s)
    st = settings(batch_size=50)
    ai.run_extraction(s, OpenAILLMClient(st, client=FakeSDK(builder=observation_builder(lambda f: cats.get(f, "OTHER")))),
                      st, sorted(cats), spec=ai.observation_spec(st), allow_full_run=True, sleep=lambda _: None)
    version = ai.current_observation_version_id(get_llm_settings())  # env default = openai / gpt-5.6-luna
    assert version == ai.current_observation_version_id(st)
    for r in rr.list_requirements(s):
        sel.run_selection(s, r["requirement_id"], version, USER)
    agent, _ = ac.create_agent(s, ac.AgentSpec(name="Demand", objective="o", execution_method="AI_ANALYSIS",
                                               scoring_method="COMBINED_CUSTOMER_DEMAND",
                                               resources=["RES_AI_OBSERVATIONS", "RES_REQUIREMENTS"],
                                               weight_percent=100), USER)
    prof = pp.create_profile(s, "OpenAI profile", {agent.agent_id: 100}, USER)
    run = dp.run_prioritization(s, prof.profile_id, USER)
    for row in dp.ranking(s, run.run_id):
        res = dp.agent_results(s, run.run_id, row.requirement_id)[0]
        n = len({f for f, c in cats.items() if c == rr.get_requirement(s, row.requirement_id).need_code})
        assert res["status"] == "SCORED" and res["inputs"]["N"] == 4000
        assert res["score"] == log_normalize(res["inputs"]["n"]) and 0 < res["inputs"]["n"] <= n
    assert dp.verify_reproducibility(s, run.run_id)["reproducible"]
    snap = next(iter(dp.evidence_snapshot(s, run.run_id).values()))
    assert snap["observation_version"]["model"] == MODEL


# ---------------------------------------------------------------------------
# connection test
# ---------------------------------------------------------------------------


def test_connection_test_success_logs_cost(s):
    def echo(params):
        nonce = params["input"].rsplit(" ", 1)[1]
        return fake_response(json.dumps({"status": "ok", "echo": nonce}), inp=60, cached=0, out=25, reasoning=10)

    sdk = FakeSDK(builder=echo)
    result = conn.run_connection_test(s, settings(), OpenAILLMClient(settings(), client=sdk))
    assert result.success and result.model_reported.startswith(MODEL) and result.usage.output_tokens == 25
    assert len(sdk.requests) == 1 and "G60" not in json.dumps(sdk.requests[0]) and "BMW" not in sdk.requests[0]["input"]
    assert result.estimated_cost_usd == pytest.approx(estimate_cost_usd(MODEL, LLMUsage(input_tokens=60, output_tokens=25)))
    assert budget.budget_status(s, settings()).spent_usd == pytest.approx(round(result.estimated_cost_usd, 4), abs=1e-4)


def test_connection_test_reports_failures_honestly(s):
    bad = OpenAILLMClient(settings(), client=FakeSDK([fake_response(json.dumps({"status": "ok", "echo": "WRONG"}))]))
    r = conn.run_connection_test(s, settings(), bad)
    assert not r.success and "did not match" in r.error
    err = OpenAILLMClient(settings(), client=FakeSDK([status_error(openai.AuthenticationError, 401)]))
    r = conn.run_connection_test(s, settings(), err)
    assert not r.success and r.error.startswith("auth")
    sdk = FakeSDK()
    r = conn.run_connection_test(s, settings(budget_usd=0.0), OpenAILLMClient(settings(), client=sdk))
    assert not r.success and "exceeds the remaining budget" in r.error and sdk.requests == []


def test_cli_requires_confirmation_before_paid_request(s, monkeypatch, capsys):
    from scripts import customer_intelligence as cli

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    monkeypatch.setenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", "1")
    sdk = FakeSDK()
    monkeypatch.setattr("services.llm_client.OpenAILLMClient.__init__",
                        lambda self, st, client=None: (setattr(self, "settings", st), setattr(self, "model", st.model),
                                                       setattr(self, "_client", sdk), setattr(self, "_openai", openai))
                        and None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit) as ex:
        cli.main(["--db", str(s.get_bind().url.database), "test-openai"])
    assert "Not confirmed" in str(ex.value.code) and sdk.requests == []
    assert "Model available" in capsys.readouterr().out
