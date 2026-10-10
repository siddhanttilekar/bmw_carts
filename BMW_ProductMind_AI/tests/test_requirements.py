"""Part 4 tests: evidence packages, rule-based generation, validation, review
workflow, persistence and the Vehicle Requirements page. Real G60 data; no
network and no paid LLM calls."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from core import config
from core.config import LLMSettings
from core.database import ensure_ai_schema, get_engine, reset_schema, session_scope
from core.requirement_models import RequirementReviewEvent, RequirementVersion, VehicleRequirement
from services import requirement_generator as gen
from services import requirement_review as rr
from services.cross_source_mapping import build_cross_source_links
from services.insight_generator import generate_insights, insight_id_for
from services.llm_client import LLMResponse, LLMUsage
from services.need_clustering import build_clusters, vfc_cluster_id
from services.need_taxonomy import MISSING_VFC, sync_taxonomy, vfc_mapping
from services.requirement_evidence import build_evidence_package, label_group
from services.requirement_validator import validate_requirement

ROOT = Path(__file__).resolve().parent.parent
TOUCH = "Touch screen, operation"


@pytest.fixture(scope="module")
def req_template(built, tmp_path_factory):
    path = tmp_path_factory.mktemp("req") / "template.db"
    shutil.copy2(built["db_path"], path)
    engine = get_engine(path)
    ensure_ai_schema(engine)
    with session_scope(engine) as s:
        sync_taxonomy(s)
        build_cross_source_links(s)
        build_clusters(s)
        generate_insights(s, min_unique_feedback=10)
    engine.dispose()
    return path


@pytest.fixture()
def db(req_template, tmp_path):
    path = tmp_path / "req.db"
    shutil.copy2(req_template, path)
    return path


@pytest.fixture()
def s(db):
    engine = get_engine(db)
    with session_scope(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture(scope="module")
def raw(data_dir):
    df = pd.read_excel(next(data_dir.rglob(config.FEEDBACK_FILES["G60"])), sheet_name=config.FEEDBACK_SHEET, dtype=object)
    mapping = {n: code for n, (code, _, _) in vfc_mapping().items()}
    df["vfc_key"] = df["Vfc level2 Name"].map(lambda v: MISSING_VFC if pd.isna(v) else v)
    df["need"] = df["vfc_key"].map(mapping)
    return df


def as_dict(draft: gen.RequirementDraft) -> dict:
    return {**draft.model_dump(), "evidence": [e.model_dump() for e in draft.evidence],
            "acceptance_criteria": [c.model_dump() for c in draft.acceptance_criteria]}


def draft_for(s, vfc_name=None, need=None):
    pkg = build_evidence_package(s, cluster_id=vfc_cluster_id(vfc_name)) if vfc_name else build_evidence_package(s, need_code=need)
    return pkg, gen.generate_rule_based(pkg)


# --- evidence ------------------------------------------------------------------


def test_evidence_package_counts_match_source(s, raw):
    pkg, _ = draft_for(s, TOUCH)
    rows = raw[raw["vfc_key"] == TOUCH]
    assert pkg.unique_feedback_count == rows["ID"].nunique() and pkg.classification_count == len(rows)
    assert sorted(pkg.direct_feedback_ids) == sorted(rows["ID"].unique())
    groups = rows.assign(g=rows["Vfc level3 Name"].map(lambda v: label_group(None if pd.isna(v) else v)))
    for g in ("issue", "request", "positive"):
        assert pkg.label_groups[g] == groups[groups["g"] == g]["ID"].nunique()
    need = build_evidence_package(s, need_code="DRIVER_ASSISTANCE")
    assert need.unique_feedback_count == raw[raw["need"] == "DRIVER_ASSISTANCE"]["ID"].nunique()


def test_evidence_package_uses_real_ids_and_separates_tiers(s, raw):
    pkg, _ = draft_for(s, "Electric Range")
    texts = raw.drop_duplicates("ID").set_index("ID")["Customer Feedback"].to_dict()
    for d in pkg.direct_evidence:
        assert d.text == texts[d.feedback_id] and d.source_file == config.FEEDBACK_FILES["G60"] and d.source_row
    assert all(i.tier == "SUPPORTING" and i.kind in ("public_owner_feedback", "public_bmw_observation", "customer_study")
               for i in pkg.supporting_evidence)
    assert all(i.tier == "CONTEXTUAL" and i.kind == "competitor_observation" for i in pkg.contextual_evidence)
    assert pkg.insight_id == insight_id_for(pkg.cluster_id)


def test_category_level_evidence_is_never_scoped_as_individual_complaints(s):
    _, draft = draft_for(s, TOUCH)
    for e in draft.evidence:
        assert (e.evidence_scope == "cluster_member") == (e.tier == "DIRECT")
    tampered = as_dict(draft)
    tampered["evidence"][-1]["evidence_scope"] = "cluster_member"
    assert any("category-level" in i.message for i in validate_requirement(s, tampered).errors)


def test_equipment_context_comes_from_option_list(s):
    pkg = build_evidence_package(s, need_code="DRIVER_ASSISTANCE")
    assert {e.option_code for e in pkg.equipment_context} >= {"5AT", "5AU"}
    assert all(e.source_file == "G60_OptionList.PDF" and e.source_page for e in pkg.equipment_context)


# --- generation --------------------------------------------------------------------


def test_rule_based_generation_needs_no_api_key(s, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", raising=False)
    pkg, draft = draft_for(s, TOUCH)
    assert draft.generation_method == "RULE_BASED" and draft.framing == "improve"
    assert draft.supporting_evidence_ids()["direct"] == pkg.direct_feedback_ids
    assert "2030" not in json.dumps(draft.model_dump()) and "next-generation" in draft.target_generation.lower()
    result = validate_requirement(s, as_dict(draft))
    assert result.passed, result.errors


def test_all_cluster_drafts_validate(s):
    from core.ai_models import CustomerNeedCluster

    framings = {}
    for c in s.scalars(select(CustomerNeedCluster).where(CustomerNeedCluster.need_code != "OTHER")):
        draft = gen.generate_rule_based(build_evidence_package(s, cluster_id=c.cluster_id))
        framings[draft.framing] = framings.get(draft.framing, 0) + 1
        assert validate_requirement(s, as_dict(draft)).passed, c.label
    assert set(framings) == {"improve", "retain", "refine"}


def test_retain_framing_for_predominantly_positive_feedback(s):
    _, draft = draft_for(s, "Exterior design")
    assert draft.framing == "retain" and "no regression" in draft.proposed_requirement.lower()


@pytest.mark.parametrize("selector", [{"need": "DRIVER_ASSISTANCE"}, {"vfc_name": "Body equipment, overall technology"},
                                      {"vfc_name": "Bidirectional charging"}])
def test_broad_or_thin_needs_get_problem_oriented_drafts(s, selector):
    _, draft = draft_for(s, **selector)
    assert draft.framing == "refine" and draft.requires_refinement
    assert draft.refinement_notes and "No technical solution is proposed" in draft.proposed_requirement
    assert draft.requirement_type == "needs_refinement"
    req = rr.create_requirement(s, draft, "pm.alice")
    rr.submit_for_review(s, req.requirement_id, "pm.alice")
    with pytest.raises(rr.InvalidTransition):
        rr.approve(s, req.requirement_id, "lead.bob", "looks fine")


def test_other_category_is_not_generatable(s):
    with pytest.raises(gen.NotGeneratable):
        gen.generate_rule_based(build_evidence_package(s, need_code="OTHER"))


# --- validation ---------------------------------------------------------------------


def test_validator_rejects_invented_facts_specs_years_and_foreign_ids(s):
    _, draft = draft_for(s, "Electric Range")
    bad = as_dict(draft)
    bad["proposed_requirement"] = "Most customers want 450 km of range by 2030; studies show 30% fewer complaints."
    bad["evidence"] += [{"tier": "DIRECT", "source_table": "feedback", "source_key": "F70-0001", "evidence_scope": "cluster_member"},
                        {"tier": "DIRECT", "source_table": "feedback", "source_key": "G60-9999", "evidence_scope": "cluster_member"},
                        {"tier": "SUPPORTING", "source_table": "public_feedback", "source_key": "P099", "evidence_scope": "need_category"}]
    codes = {i.code for i in validate_requirement(s, bad).errors}
    assert codes >= {"VEHICLE_SCOPE", "EVIDENCE_VALID", "UNSUPPORTED_CLAIMS", "INVENTED_STATISTICS", "UNSUPPORTED_SPEC"}
    messages = " ".join(i.message for i in validate_requirement(s, bad).errors)
    assert "F70-0001 is F70" in messages and "G60-9999 does not exist" in messages and "P099" in messages


def test_validator_rejects_copied_complaint(s, raw):
    pkg, draft = draft_for(s, TOUCH)
    copied = as_dict(draft)
    copied["proposed_requirement"] = pkg.direct_evidence[0].text
    assert any(i.code == "NOT_COPIED_COMPLAINT" for i in validate_requirement(s, copied).errors)


def test_validator_requires_assumptions_and_need(s):
    _, draft = draft_for(s, TOUCH)
    empty = {**as_dict(draft), "assumptions": [], "customer_need": "UI"}
    codes = {i.code for i in validate_requirement(s, empty).errors}
    assert {"ASSUMPTIONS", "NEED_CLEAR"} <= codes


# --- review workflow -------------------------------------------------------------------


def test_edit_creates_versions_and_preserves_original(s):
    _, draft = draft_for(s, TOUCH)
    req = rr.create_requirement(s, draft, "pm.alice")
    rid = req.requirement_id
    rr.edit_requirement(s, rid, {"title": "Low-effort touchscreen operation", "customer_need":
                                 "Drivers need to complete frequent touchscreen tasks with low effort."}, "pm.alice", "clarify")
    rr.edit_requirement(s, rid, {"assumptions": ["Single assumption kept."]}, "pm.carol")
    versions = rr.versions(s, rid)
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert versions[0]["change_type"] == "generated" and versions[0]["snapshot"]["title"] == draft.title
    assert versions[0]["snapshot"]["proposed_requirement"] == draft.proposed_requirement
    assert versions[1]["changed_fields"] == ["title", "customer_need"] and versions[2]["changed_by"] == "pm.carol"
    assert rr.get_requirement(s, rid).title == "Low-effort touchscreen operation"


def test_edits_before_first_save_keep_generated_version(s):
    _, draft = draft_for(s, TOUCH)
    req = rr.create_requirement(s, draft, "pm.alice", edited={"title": "Edited title before save"})
    versions = rr.versions(s, req.requirement_id)
    assert versions[0]["snapshot"]["title"] == draft.title and versions[1]["snapshot"]["title"] == "Edited title before save"


def test_review_decisions_persist_and_rejected_cannot_jump_to_approved(db):
    engine = get_engine(db)
    with session_scope(engine) as s:
        _, draft = draft_for(s, TOUCH)
        rid = rr.create_requirement(s, draft, "pm.alice").requirement_id
        rr.submit_for_review(s, rid, "pm.alice")
        rr.reject(s, rid, "lead.bob", "Criteria need scenario definitions")
    with session_scope(engine) as s:  # new session: decisions persisted
        assert rr.get_requirement(s, rid).review_status == "REJECTED"
        with pytest.raises(rr.InvalidTransition):
            rr.approve(s, rid, "lead.bob", "changed my mind")
        assert rr.get_requirement(s, rid).review_status == "REJECTED"
        with pytest.raises(rr.InvalidTransition):
            rr.reopen(s, rid, "pm.alice", "")  # reason required
        rr.reopen(s, rid, "pm.alice", "Added scenario definitions")
        rr.submit_for_review(s, rid, "pm.alice")
        with pytest.raises(rr.InvalidTransition, match="Four-eyes"):
            rr.approve(s, rid, "pm.alice", "self approval")
        rr.approve(s, rid, "lead.bob", "Evidence and criteria acceptable")
    with session_scope(engine) as s:
        events = s.scalars(select(RequirementReviewEvent).where(RequirementReviewEvent.requirement_id == rid)
                           .order_by(RequirementReviewEvent.event_id)).all()
        transitions = [(e.action, e.previous_status, e.new_status, e.reviewer_id) for e in events]
        assert transitions == [
            ("create", None, "DRAFT", "pm.alice"), ("submit", "DRAFT", "PENDING_REVIEW", "pm.alice"),
            ("reject", "PENDING_REVIEW", "REJECTED", "lead.bob"), ("reopen", "REJECTED", "DRAFT", "pm.alice"),
            ("submit", "DRAFT", "PENDING_REVIEW", "pm.alice"), ("approve", "PENDING_REVIEW", "APPROVED", "lead.bob")]
        assert all(e.created_at for e in events) and events[2].reason == "Criteria need scenario definitions"
    engine.dispose()


def test_editing_an_approved_requirement_returns_it_to_draft(s):
    _, draft = draft_for(s, TOUCH)
    rid = rr.create_requirement(s, draft, "pm.alice").requirement_id
    rr.submit_for_review(s, rid, "pm.alice")
    rr.approve(s, rid, "lead.bob", "ok")
    rr.edit_requirement(s, rid, {"title": "Changed after approval"}, "pm.alice")
    assert rr.get_requirement(s, rid).review_status == "DRAFT"
    assert rr.history(s, rid)[-1].previous_status == "APPROVED"


def test_requirements_are_never_auto_approved(s):
    _, draft = draft_for(s, TOUCH)
    req = rr.create_requirement(s, draft, "pm.alice")
    assert req.review_status == "DRAFT"
    with pytest.raises(rr.InvalidTransition):
        rr.approve(s, req.requirement_id, "lead.bob", "skip review")


# --- persistence / IDs ---------------------------------------------------------------------


def test_requirement_ids_survive_rebuilds(db):
    engine = get_engine(db)
    with session_scope(engine) as s:
        _, draft = draft_for(s, TOUCH)
        rid = rr.create_requirement(s, draft, "pm.alice").requirement_id
    assert rid.startswith("REQ-") and len(rid) == 16
    with session_scope(engine) as s:
        build_clusters(s)
        generate_insights(s, min_unique_feedback=10)
    reset_schema(engine)  # ingestion rebuild drops only ingestion tables
    with session_scope(engine) as s:
        req = s.get(VehicleRequirement, rid)
        assert req is not None and req.source_cluster_id == vfc_cluster_id(TOUCH)
        assert s.scalar(select(RequirementVersion.version_number).where(RequirementVersion.requirement_id == rid)) == 1
    engine.dispose()


# --- LLM interface (gated) -------------------------------------------------------------------


class FakeLLM:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def complete_json(self, *, system, user, schema, max_tokens):
        self.calls += 1
        return LLMResponse(text=json.dumps({
            "title": "Low-effort touchscreen operation", "customer_need": "Customers need low-effort touchscreen operation.",
            "customer_problem": "Customers describe touchscreen usability issues.",
            "proposed_requirement": "The next-generation BMW 5 Series shall offer low-effort touchscreen operation.",
            "assumptions": ["Wording drafted from VFC labels."],
            "acceptance_criteria": [{"criterion": "Usability test with target customers", "measurement": "task success rate"}],
        }), stop_reason="end_turn", model=self.model, usage=LLMUsage(input_tokens=10, output_tokens=10))


def test_llm_mode_is_disabled_by_default(s, monkeypatch):
    monkeypatch.delenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    assert not gen.llm_generation_status().enabled
    pkg, _ = draft_for(s, TOUCH)
    with pytest.raises(gen.LLMNotAuthorized):
        gen.generate_llm_assisted(pkg, confirmed=True, cost_acknowledged=gen.estimate_llm_cost(pkg))


def test_llm_mode_requires_confirmation_and_never_falls_back(s, monkeypatch):
    monkeypatch.setenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    pkg, _ = draft_for(s, TOUCH)
    fake = FakeLLM()
    with pytest.raises(gen.LLMNotAuthorized):
        gen.generate_llm_assisted(pkg, confirmed=False, cost_acknowledged=gen.estimate_llm_cost(pkg), client=fake)
    with pytest.raises(gen.LLMNotAuthorized):
        gen.generate_llm_assisted(pkg, confirmed=True, cost_acknowledged=None, client=fake)
    assert fake.calls == 0
    settings = LLMSettings(model="fake-model", use_fallbacks=True, external_calls_authorized=True)
    draft = gen.generate_llm_assisted(pkg, confirmed=True, cost_acknowledged=gen.estimate_llm_cost(pkg),
                                      settings=settings, client=fake)
    assert fake.calls == 1 and draft.generation_method == "LLM_ASSISTED"
    assert draft.evidence == gen.generate_rule_based(pkg).evidence  # evidence attached by code, not the model
    assert validate_requirement(s, as_dict(draft)).passed


def test_fallbacks_are_off_by_default(monkeypatch):
    monkeypatch.delenv("PRODUCTMIND_LLM_FALLBACKS", raising=False)
    assert LLMSettings().use_fallbacks is False


def test_anthropic_client_is_built_without_fallbacks(monkeypatch, s):
    captured = {}

    class Spy:
        def __init__(self, settings, client=None):
            captured["fallbacks"] = settings.use_fallbacks
            self.model = settings.model

        def complete_json(self, **kw):
            return FakeLLM().complete_json(**kw)

    monkeypatch.setattr("services.llm_client.AnthropicLLMClient", Spy)
    monkeypatch.setenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    monkeypatch.setenv("PRODUCTMIND_LLM_FALLBACKS", "1")
    pkg, _ = draft_for(s, TOUCH)
    gen.generate_llm_assisted(pkg, confirmed=True, cost_acknowledged=gen.estimate_llm_cost(pkg))
    assert captured["fallbacks"] is False


# --- Streamlit -------------------------------------------------------------------------------


@pytest.fixture()
def app(db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRODUCTMIND_DB_PATH", str(db))
    monkeypatch.delenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", raising=False)

    def forbid(*a, **k):
        raise AssertionError("No LLM client may be constructed")

    monkeypatch.setattr("services.llm_client.AnthropicLLMClient.__init__", forbid)
    monkeypatch.setattr("anthropic.Anthropic.__init__", forbid, raising=False)
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_all_pages_still_load(app):
    app.run()
    assert not app.exception
    for page in ("ui/pages/evidence_explorer.py", "ui/pages/customer_insights.py", "ui/pages/vehicle_requirements.py",
                 "ui/pages/data_quality.py"):
        app.switch_page(page).run()
        assert not app.exception, (page, [e.value for e in app.exception])


def test_page_flow_insight_to_saved_requirement(app, db):
    app.switch_page("ui/pages/customer_insights.py").run()
    button = next(b for b in app.button if b.label == "Draft Requirement")
    insight_id = button.key.removeprefix("req_")
    button.click().run()
    assert not app.exception
    assert app.session_state["req_insight"] == insight_id
    # AppTest re-runs the page last selected via its API; select the requirements page explicitly
    # (the insight selection is preserved in session state).
    app.switch_page("ui/pages/vehicle_requirements.py").run()
    assert app.selectbox(key="req_insight").value == insight_id
    next(b for b in app.button if b.label == "Generate Requirement Draft").click().run()
    text = " ".join(m.value for m in app.markdown)
    assert "Evidence-based draft — not AI-generated" in text
    app.text_input(key="req_actor").set_value("pm.alice").run()
    next(b for b in app.button if b.label == "Save Draft").click().run()
    assert not app.exception
    engine = get_engine(db)
    with session_scope(engine) as s:
        reqs = s.scalars(select(VehicleRequirement)).all()
        assert len(reqs) == 1 and reqs[0].source_insight_id == insight_id and reqs[0].review_status == "DRAFT"
        assert reqs[0].generation_method == "RULE_BASED" and reqs[0].created_by == "pm.alice"
    engine.dispose()


def test_page_shows_llm_mode_disabled(app):
    app.switch_page("ui/pages/vehicle_requirements.py").run()
    assert any("not authorised by default" in i.value for i in app.info)
    assert not [b for b in app.button if b.label == "Generate LLM-assisted draft"]
