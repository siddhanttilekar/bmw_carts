"""BMW_CAST Bot: branding, inline debate, owner-authorised allowance.
The LLM is always a local fake - no paid calls."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from sqlalchemy import func, select

from core.ai_models import AIApiCall, AIProcessingJob
from core.database import get_engine, session_scope
from services import argue_chat
from services import dynamic_prioritization as dp
from services import llm_budget as budget
from services import product_workflow as pw
from services.llm_client import LLMResponse, LLMUsage
from tests.test_evidence_first import (  # noqa: F401  (fixture re-export)
    MODEL, USER, base_db, extract_all_direct, requirements, settings,
)
from tests.test_product_workflow import FakeChat, coverage_agent, data_agent

ROOT = Path(__file__).resolve().parent.parent
UI_FILES = [ROOT / "ui/pages/product_prioritization.py", ROOT / "ui/pages/dashboard.py"]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("PRODUCTMIND_LLM_PROVIDER", "openai")
    monkeypatch.setenv("PRODUCTMIND_LLM_MODEL", MODEL)
    for var in ("PRODUCTMIND_LLM_MODEL_VERSION", "OPENAI_API_KEY", "PRODUCTMIND_ALLOW_EXTERNAL_LLM",
                "PRODUCTMIND_DEBATE_ALLOWANCE_USD"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def world(base_db, tmp_path):  # noqa: F811
    path = tmp_path / "b.db"
    shutil.copy2(base_db, path)
    engine = get_engine(path)
    with session_scope(engine) as s:
        extract_all_direct(s)
        d, h = data_agent(s), coverage_agent(s)
        rids = [r["requirement_id"] for r in requirements(s)]
        run_id = pw.execute(s, settings(), rids, {d: 60, h: 40}, USER)
    engine.dispose()
    return {"path": path, "rids": rids, "run": run_id, "data": d, "human": h}


def test_01_02_branding_is_consistent_and_never_chatgpt():
    assert argue_chat.BOT_NAME == "BMW_CAST Bot"
    assert argue_chat.SYSTEM.startswith("You are BMW_CAST Bot") and "not an official BMW product" in argue_chat.SYSTEM
    for f in UI_FILES:
        src = f.read_text()
        for banned in ("ChatGPT", "Claude", "GPT Assistant", "Chatbot", "chatbot"):
            assert banned not in src, (f.name, banned)
    page = (ROOT / "ui/pages/product_prioritization.py").read_text()
    for phrase in ("Ask {BOT}", "{BOT} is analyzing your argument...", "Welcome to {BOT}.",
                   "could not find enough evidence"):
        assert phrase in page


def test_08_09_allowance_lets_messages_through_and_is_tracked(world, monkeypatch):
    engine = get_engine(world["path"])
    with session_scope(engine) as s:
        rid = world["rids"][0]
        with pytest.raises(budget.ApprovalRequired):  # no allowance -> approval still required
            argue_chat.ask(s, FakeChat(), settings(), world["run"], rid, [], "Why?", USER)
        st = settings(debate_allowance_usd=0.5)
        status = argue_chat.allowance_status(s, st)
        assert status["authorized"] and status["remaining_usd"] == 0.5
        chat = FakeChat()
        spent_before = budget.budget_status(s, st).spent_usd
        for q in ("Why is this first?", "I disagree", "What contradicts it?"):  # no per-message approval
            argue_chat.ask(s, chat, st, world["run"], rid, [], q, USER)
        assert len(chat.messages) == 3
        after = argue_chat.allowance_status(s, st)
        assert after["spent_usd"] > 0 and after["remaining_usd"] == pytest.approx(0.5 - after["spent_usd"], abs=1e-4)
        assert budget.budget_status(s, st).spent_usd > spent_before  # same application ledger
        calls = s.scalar(select(func.count()).select_from(AIApiCall).join(
            AIProcessingJob, AIProcessingJob.job_id == AIApiCall.job_id).where(AIProcessingJob.job_type == "argue_chat"))
        assert calls == 3
        # exhausted allowance stops the bot before any call
        tiny = settings(debate_allowance_usd=after["spent_usd"] + 0.000001)
        with pytest.raises(budget.BudgetExceeded):
            argue_chat.ask(s, chat, tiny, world["run"], rid, [], "One more", USER)
        # the allowance can never exceed the application budget
        capped = settings(debate_allowance_usd=20.0, budget_usd=budget.budget_status(s, st).spent_usd + 0.0001)
        assert argue_chat.allowance_status(s, capped)["remaining_usd"] <= 0.0001
        with pytest.raises(budget.BudgetExceeded):
            argue_chat.ask(s, chat, capped, world["run"], rid, [], "Again", USER)
        assert len(chat.messages) == 3
        # requires the external-call authorisation as well
        assert not argue_chat.allowance_status(s, settings(debate_allowance_usd=1, external_calls_authorized=False))["authorized"]
    engine.dispose()


class ScenarioChat(FakeChat):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def complete_json(self, *, system, user, schema, max_tokens):
        self.messages.append(user)
        return LLMResponse(text=json.dumps({"answer": "A higher weight on customer demand would fit the evidence.",
                                            "stance": "challenges_ranking", "evidence_ids": [],
                                            "suggested_weights": self.weights}),
                           stop_reason="end_turn", model=MODEL, usage=LLMUsage(input_tokens=1000, output_tokens=100))


def test_weight_scenarios_are_validated_in_python_and_never_applied(world):
    engine = get_engine(world["path"])
    with session_scope(engine) as s:
        st = settings(debate_allowance_usd=1)
        rid = world["rids"][0]
        names = {a["agent_id"]: a["name"] for a in pw.available_agents(s)}
        good = [{"agent": names[world["data"]], "weight_percent": 70}, {"agent": names[world["human"]], "weight_percent": 30}]
        out = argue_chat.ask(s, ScenarioChat(good), st, world["run"], rid, [], "More weight on demand?", USER)
        assert out["scenario"] == {world["data"]: 70.0, world["human"]: 30.0}
        for bad in ([{"agent": names[world["data"]], "weight_percent": 80}, {"agent": names[world["human"]], "weight_percent": 30}],
                    [{"agent": "Invented Agent", "weight_percent": 100}],
                    [{"agent": names[world["data"]], "weight_percent": 100}]):
            o = argue_chat.ask(s, ScenarioChat(bad), st, world["run"], rid, [], "?", USER)
            assert o["scenario"] is None and any("not shown" in n for n in o["notes"])
        before = [(r.requirement_id, r.final_score, r.rank) for r in dp.ranking(s, world["run"])]
        assert before == [(r.requirement_id, r.final_score, r.rank) for r in dp.ranking(s, world["run"])]
        assert dp.verify_reproducibility(s, world["run"])["reproducible"]  # 12: history intact
    engine.dispose()


def _app(path, monkeypatch, *, enabled):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRODUCTMIND_DB_PATH", str(path))
    if enabled:
        monkeypatch.setenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-NOT-A-REAL-KEY-0123456789")
        monkeypatch.setenv("PRODUCTMIND_DEBATE_ALLOWANCE_USD", "1")
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_03_to_07_inline_panels_separate_chats_and_api_only_on_send(world, monkeypatch):
    calls = []

    class SharedClientFake(FakeChat):
        def complete_json(self, **kw):
            calls.append(kw["user"])
            return super().complete_json(**kw)

    monkeypatch.setattr("services.evidence_extraction.default_client", lambda st: SharedClientFake())
    monkeypatch.setattr("services.llm_client.OpenAILLMClient.__init__",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("real client")))
    app = _app(world["path"], monkeypatch, enabled=True)
    app.session_state["workflow_actor"] = USER
    app.session_state["wiz_step"] = 4
    app.session_state["wiz_run_id"] = world["run"]
    app.switch_page("ui/pages/product_prioritization.py").run()
    assert not app.exception
    labels = [b.label for b in app.button]
    assert labels.count("WHY?") == 2 and labels.count("ARGUE") == 2  # 3: every ranked requirement
    argue = [b for b in app.button if b.label == "ARGUE"]
    argue[0].click().run()
    html = " ".join(m.value for m in app.markdown)
    assert "#### BMW_CAST Bot" in html and "Customer evidence" in html and "Agent perspectives" in html
    assert any("Welcome to BMW_CAST Bot." in i.value for i in app.info)
    assert calls == []  # 6: opening makes no API call
    assert not any("Allow paid replies" in c.label for c in app.checkbox)  # 8: no approval form with allowance
    key0 = f"{world['run']}|{world['rids'][0]}" if f"argue_q_{world['run']}|{world['rids'][0]}" in \
        [t.key for t in app.text_area] else f"{world['run']}|{world['rids'][1]}"
    app.text_area(key=f"argue_q_{key0}").input("Why is this ranked here?")
    next(b for b in app.button if b.label == "SEND").click().run()
    assert not [e.value for e in app.error], [e.value for e in app.error]
    assert not app.exception and len(calls) == 1  # 7: shared client called once on SEND
    chats = app.session_state["wiz_chats"]
    assert len(chats[key0]) == 2 and chats[key0][1]["role"] == "assistant"
    other = [b for b in app.button if b.label == "ARGUE"]
    other[1 if key0.endswith(world["rids"][0]) else 0].click().run()
    chats = app.session_state["wiz_chats"]
    assert sum(len(v) for k, v in chats.items() if k != key0) == 0  # 5: the other requirement has its own chat
    assert len(calls) == 1
    page_text = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
    assert "sk-test-NOT-A-REAL-KEY" not in page_text  # 10: key never shown
    app.switch_page("ui/pages/application_settings.py").run()
    settings_text = " ".join(str(x.value) for x in app.markdown) + " ".join(str(x.value) for x in app.caption)
    assert "sk-test-NOT-A-REAL-KEY" not in settings_text and "BMW_CAST Bot debate allowance" in settings_text
    assert re.search(r"API provider: openai", settings_text)


def test_11_scoring_unchanged(world):
    engine = get_engine(world["path"])
    with session_scope(engine) as s:
        for row in dp.ranking(s, world["run"]):
            res = {r["agent_id"]: r for r in dp.agent_results(s, world["run"], row.requirement_id)}
            assert row.final_score == round(res[world["data"]]["score"] * 0.6 + res[world["human"]]["score"] * 0.4, 2)
    engine.dispose()


def test_no_payment_prompts_in_main_ui_and_costs_still_tracked(world, monkeypatch):
    """No approval fields, credit limits or confirmations in the wizard: AI agents
    and BMW_CAST Bot use the configured key automatically; the backend still
    enforces the application budget and logs every call."""
    from tests.test_agent_analysis import FakeJudge
    from tests.test_product_workflow import demand_ai

    calls = []

    class Shared(FakeChat):
        def complete_json(self, **kw):
            calls.append("chat")
            return super().complete_json(**kw)

    judge = FakeJudge()

    engine = get_engine(world["path"])
    with session_scope(engine) as s:
        ai_agent = demand_ai(s)
    engine.dispose()
    monkeypatch.setattr("services.evidence_extraction.default_client", lambda st: judge if not judge.payloads else Shared())
    monkeypatch.setattr("services.llm_client.OpenAILLMClient.__init__",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("real client")))
    app = _app(world["path"], monkeypatch, enabled=True)
    monkeypatch.delenv("PRODUCTMIND_DEBATE_ALLOWANCE_USD")  # no owner allowance: still no prompt
    app.session_state["workflow_actor"] = USER
    app.session_state["wiz_step"] = 3
    app.session_state["wiz_reqs"] = world["rids"]
    app.session_state["wiz_agents"] = [ai_agent]
    app.switch_page("ui/pages/product_prioritization.py").run()
    assert not app.exception
    app.number_input(key=f"w_wt_{ai_agent}").set_value(100).run()
    labels = " ".join([n.label for n in app.number_input] + [c.label for c in app.checkbox]
                      + [m.value for m in app.markdown] + [c.value for c in app.caption])
    for banned in ("approve", "Approve", "USD", "Allow paid", "spending up to", "estimated $"):
        assert banned not in labels, banned
    run_btn = next(b for b in app.button if b.label == "RUN PRIORITIZATION")
    assert not run_btn.disabled
    run_btn.click().run()
    assert not app.exception and app.session_state["wiz_step"] == 4 and judge.payloads  # AI ran without approval
    assert not any("Allow paid" in c.label for c in app.checkbox) and not [e for e in app.expander
                                                                           if "Enable" in e.label]
    next(b for b in app.button if b.label == "ARGUE").click().run()
    key = next(t.key for t in app.text_area if t.key.startswith("argue_q_"))
    app.text_area(key=key).input("Why this priority?")
    send = next(b for b in app.button if b.label == "SEND")
    assert not send.disabled
    send.click().run()
    assert not app.exception and calls == ["chat"]  # sent without any approval prompt
    engine = get_engine(world["path"])
    with session_scope(engine) as s:
        jobs = dict(s.execute(select(AIProcessingJob.job_type, func.count()).group_by(AIProcessingJob.job_type)).all())
        assert jobs.get("agent_analysis") == 2 and jobs.get("argue_chat") == 1  # every call still in the ledger
        assert budget.budget_status(s, settings()).spent_usd > 0
    engine.dispose()
