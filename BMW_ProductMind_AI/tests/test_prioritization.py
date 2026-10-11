"""Part 5 tests: agents, resources, scoring, weights, profiles, runs,
explanations, critical flags, decision context and the Streamlit page.

All tests run on temporary database copies. Requirements are generated in the
copy from real G60 clusters (rule-based); human assessment scores used here
are SYNTHETIC TEST VALUES and never touch the production database."""
from __future__ import annotations

import json
import math
import shutil
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from core import config
from core.database import ensure_ai_schema, get_engine, session_scope
from core.prioritization_models import AgentConfigVersion, AgentExecutionResult, PrioritizationAgent
from core.requirement_models import RequirementEvidenceLink
from services import agent_configuration as ac
from services import critical_issues as ci
from services import decision_context as dc
from services import dynamic_prioritization as dp
from services import prioritization_profiles as pp
from services import requirement_review as rr
from services.agent_resources import RESOURCES, list_resources
from services.agent_scoring import AI_UNAVAILABLE, METHODS, check_compatibility
from services.agent_weights import WeightEntry, distribute_remaining, percent_to_bp, validate_weights, weighted_contribution
from services.cross_source_mapping import build_cross_source_links
from services.insight_generator import generate_insights
from services.need_clustering import build_clusters, vfc_cluster_id
from services.need_taxonomy import sync_taxonomy
from services.requirement_evidence import build_evidence_package
from services.requirement_generator import generate_rule_based

ROOT = Path(__file__).resolve().parent.parent
CLUSTERS = ["Touch screen, operation", "Electric Range", "Head-Up Display"]
USER = "test.pm"


@pytest.fixture(scope="module")
def base_db(built, tmp_path_factory):
    path = tmp_path_factory.mktemp("prio") / "base.db"
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


@pytest.fixture(scope="module")
def req_db(base_db, tmp_path_factory):
    """Base + three rule-based requirements generated from real G60 clusters."""
    path = tmp_path_factory.mktemp("prio") / "reqs.db"
    shutil.copy2(base_db, path)
    engine = get_engine(path)
    with session_scope(engine) as s:
        for name in CLUSTERS:
            rr.create_requirement(s, generate_rule_based(build_evidence_package(s, cluster_id=vfc_cluster_id(name))), USER)
    engine.dispose()
    return path


@pytest.fixture()
def s(req_db, tmp_path):
    path = tmp_path / "p.db"
    shutil.copy2(req_db, path)
    engine = get_engine(path)
    with session_scope(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture()
def empty_s(base_db, tmp_path):
    path = tmp_path / "empty.db"
    shutil.copy2(base_db, path)
    engine = get_engine(path)
    with session_scope(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture(scope="module")
def raw(data_dir):
    return pd.read_excel(next(data_dir.rglob(config.FEEDBACK_FILES["G60"])), sheet_name=config.FEEDBACK_SHEET, dtype=object)


def demand_spec(name="Customer Demand Agent", weight=0):
    return ac.AgentSpec(name=name, objective="Customer demand from unique G60 comments.", execution_method="DATA_ANALYSIS",
                        scoring_method="LOG_NORMALIZED_FEEDBACK_COUNT", resources=["RES_G60_FEEDBACK", "RES_REQUIREMENTS"],
                        weight_percent=weight)


def coverage_spec(name="Evidence Coverage Agent", resources=None):
    return ac.AgentSpec(name=name, objective="Traceable evidence available.", execution_method="DATA_ANALYSIS",
                        scoring_method="EVIDENCE_COVERAGE",
                        resources=resources or ["RES_REQUIREMENTS", "RES_G60_FEEDBACK", "RES_PUBLIC_FEEDBACK", "RES_CUSTOMER_STUDIES"])


def human_spec(name):
    return ac.AgentSpec(name=name, objective=f"{name} judged by a product manager.", execution_method="HUMAN_ASSESSMENT",
                        scoring_method="HUMAN_SCORE", resources=["RES_HUMAN_ASSESSMENTS", "RES_REQUIREMENTS"])


def req_ids(s):
    return [r["requirement_id"] for r in sorted(rr.list_requirements(s), key=lambda r: r["title"])]


def make_profile(s, specs_weights: list[tuple[ac.AgentSpec, str]], name="Test profile"):
    ids = {}
    for spec, w in specs_weights:
        agent, _ = ac.create_agent(s, spec, USER)
        ids[agent.agent_id] = w
    p = pp.create_profile(s, name, ids, USER)
    return p, list(ids)


def assess_all(s, agent_ids, scores):  # SYNTHETIC TEST VALUES
    for rid in req_ids(s):
        for aid, sc in zip(agent_ids, scores):
            if s.get(PrioritizationAgent, aid).scoring_method == "HUMAN_SCORE":
                dp.record_human_assessment(s, rid, aid, sc, "Synthetic test assessment", "test.assessor")


# --- 1-7 agent builder & resources ------------------------------------------------------


def test_create_edit_duplicate_deactivate_remove(s):
    agent, warnings = ac.create_agent(s, demand_spec(weight=30), USER)
    assert agent.agent_id.startswith("AGT-") and agent.configuration_version == 1 and agent.default_weight_bp == 3000
    spec = ac.current_spec(s, agent.agent_id)
    spec.objective = "Updated objective"
    ac.edit_agent(s, agent.agent_id, spec, USER)
    versions = ac.versions(s, agent.agent_id)
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["snapshot"]["objective"] != "Updated objective"  # v1 preserved
    dup = ac.duplicate_agent(s, agent.agent_id, USER)
    assert dup.agent_id != agent.agent_id and dup.name.endswith("(copy)") and dup.scoring_method == agent.scoring_method
    ac.set_enabled(s, dup.agent_id, False, USER)
    assert not s.get(PrioritizationAgent, dup.agent_id).enabled
    ac.remove_agent(s, dup.agent_id, USER)
    row = s.get(PrioritizationAgent, dup.agent_id)
    assert row is not None and row.deleted  # soft delete
    assert dup.agent_id not in {a["agent_id"] for a in ac.list_agents(s)}
    assert dup.agent_id in {a["agent_id"] for a in ac.list_agents(s, include_removed=True)}
    with pytest.raises(ac.AgentConfigError):
        ac.edit_agent(s, dup.agent_id, demand_spec(), USER)


def test_valid_and_incompatible_resources(s):
    ac.create_agent(s, coverage_spec(resources=["RES_REQUIREMENTS", "RES_COMPETITOR", "RES_EQUIPMENT"]), USER)
    bad = [
        demand_spec().model_copy(update={"resources": ["RES_SALES", "RES_REQUIREMENTS"]}),  # missing feedback; sales unsupported
        demand_spec().model_copy(update={"execution_method": "HUMAN_ASSESSMENT"}),  # method/execution mismatch
        coverage_spec(resources=["RES_REQUIREMENTS"]),  # no evidence resource
        human_spec("X").model_copy(update={"resources": ["RES_REQUIREMENTS"]}),  # missing assessments
        demand_spec().model_copy(update={"resources": ["RES_G60_FEEDBACK", "RES_REQUIREMENTS", "RES_NOT_REAL"]}),
        demand_spec().model_copy(update={"scoring_method": "MANUFACTURING_COST_ESTIMATE"}),  # unsupported method
    ]
    for spec in bad:
        with pytest.raises(ac.AgentConfigError):
            ac.create_agent(s, spec, USER)
    errors, warnings = check_compatibility("DATA_ANALYSIS", "EVIDENCE_COVERAGE", ["RES_REQUIREMENTS", "RES_G60_FEEDBACK"])
    assert not errors and any("not customer impact" in w for w in warnings)


def test_agent_name_does_not_determine_behaviour(s):
    agent, _ = ac.create_agent(s, demand_spec(name="Manufacturing Cost Agent"), USER)
    p = pp.create_profile(s, "Name test", {agent.agent_id: 100}, USER)
    run = dp.run_prioritization(s, p.profile_id, USER)
    result = dp.agent_results(s, run.run_id, req_ids(s)[0])[0]
    assert result["scoring_method"] == "LOG_NORMALIZED_FEEDBACK_COUNT"
    assert "unique_g60_feedback_ids" in result["inputs"] and "cost" not in json.dumps(result["inputs"]).lower()


def test_resource_registry_lists_all_resources_with_live_counts(s):
    resources = {r.resource_id: r for r in list_resources(s)}
    assert len(resources) == 10 and set(resources) == set(RESOURCES)  # 8 (Part 5) + 2 (2.0: AI observations, uploads)
    assert resources["RES_G60_FEEDBACK"].available_records == 4000
    assert resources["RES_REQUIREMENTS"].available_records == 3
    for r in resources.values():
        assert r.limitations and r.supported_analysis_methods and r.available_fields


# --- 8-11 execution, human assessments --------------------------------------------------------


def test_deterministic_demand_scoring_matches_formula(s, raw):
    p, _ = make_profile(s, [(demand_spec(), 100)])
    run1 = dp.run_prioritization(s, p.profile_id, USER)
    run2 = dp.run_prioritization(s, p.profile_id, USER)
    for rid in req_ids(s):
        r1, r2 = dp.agent_results(s, run1.run_id, rid)[0], dp.agent_results(s, run2.run_id, rid)[0]
        assert r1["score"] == r2["score"]
        n = r1["inputs"]["unique_g60_feedback_ids"]
        assert r1["inputs"]["reference_population"] == raw["ID"].nunique() == 4000
        assert r1["score"] == round(100 * math.log1p(n) / math.log1p(4000), 2)


def test_unique_comment_counting(s, raw):
    p, _ = make_profile(s, [(demand_spec(), 100)])
    run = dp.run_prioritization(s, p.profile_id, USER)
    touch = next(r for r in rr.list_requirements(s) if "Touch screen" in r["title"])["requirement_id"]
    result = dp.agent_results(s, run.run_id, touch)[0]
    rows = raw[raw["Vfc level2 Name"] == "Touch screen, operation"]
    assert result["inputs"]["unique_g60_feedback_ids"] == rows["ID"].nunique() == 70 < len(rows) == 72


def test_human_assessments_stored_and_validated(s):
    agent, _ = ac.create_agent(s, human_spec("Customer Impact Agent"), USER)
    rid = req_ids(s)[0]
    a = dp.record_human_assessment(s, rid, agent.agent_id, 80, "Synthetic test rationale", "test.assessor")
    assert a.score == 80 and a.assessor_id == "test.assessor" and a.assessment_timestamp
    for bad in (101, -1, 50.5, "abc"):
        with pytest.raises(dp.PrioritizationError):
            dp.record_human_assessment(s, rid, agent.agent_id, bad, "x", "test.assessor")
    with pytest.raises(dp.PrioritizationError, match="rationale"):
        dp.record_human_assessment(s, rid, agent.agent_id, 50, "  ", "test.assessor")
    with pytest.raises(dp.PrioritizationError, match="assessor"):
        dp.record_human_assessment(s, rid, agent.agent_id, 50, "ok", "")
    with pytest.raises(dp.PrioritizationError):
        dp.record_human_assessment(s, rid, agent.agent_id, 50, "ok", "x", evidence_refs=["G60-9999"])
    demand, _ = ac.create_agent(s, demand_spec(), USER)
    with pytest.raises(dp.PrioritizationError):  # data agents never take human scores
        dp.record_human_assessment(s, rid, demand.agent_id, 50, "ok", "x")


def test_human_scores_are_never_generated(s):
    p, ids = make_profile(s, [(human_spec("Impact"), 100)])
    run = dp.run_prioritization(s, p.profile_id, USER)
    for row in dp.ranking(s, run.run_id):
        assert row.priority_status == "INCOMPLETE" and row.final_score is None and row.rank is None


# --- 12-18 weights, agent counts, recalculation, incomplete -------------------------------------


def test_weight_validation_uses_exact_basis_points():
    assert percent_to_bp("33.33") + percent_to_bp("33.33") + percent_to_bp("33.34") == 10000
    for bad in ("-1", "100.01", "12.345", "abc"):
        with pytest.raises(ValueError):
            percent_to_bp(bad)
    ok = validate_weights([WeightEntry(agent_id="a", weight_bp=6000), WeightEntry(agent_id="b", weight_bp=4000)])
    assert ok.valid and ok.remaining_bp == 0
    low = validate_weights([WeightEntry(agent_id="a", weight_bp=8500)])
    assert not low.valid and low.remaining_bp == 1500
    disabled = validate_weights([WeightEntry(agent_id="a", weight_bp=10000, enabled=False)])
    assert not disabled.valid and disabled.active_agents == 0
    assert distribute_remaining({"a": 5000, "b": 3500}, ["a", "b"]) == {"a": 5750, "b": 4250}


def test_scoring_rejected_when_weights_do_not_total_100(s):
    p, _ = make_profile(s, [(demand_spec(), 85)])
    assert not pp.validate_profile(s, p.profile_id).valid
    with pytest.raises(dp.PrioritizationError, match="100%"):
        dp.run_prioritization(s, p.profile_id, USER)
    with pytest.raises(pp.ProfileError):
        pp.activate_profile(s, p.profile_id, USER)


def test_illustrative_weighted_sum_example():
    # Spec example (synthetic, not stored anywhere): 90/30, 80/25, 70/20, 85/15, 60/10 -> 79.75
    pairs = [(90, 3000), (80, 2500), (70, 2000), (85, 1500), (60, 1000)]
    assert sum(weighted_contribution(sc, bp) for sc, bp in pairs) == Decimal("79.75")


@pytest.mark.parametrize("n_agents, weights", [
    (3, ["40", "30", "30"]),
    (5, ["30", "25", "20", "15", "10"]),
    (7, ["20", "15", "15", "15", "15", "10", "10"]),
])
def test_profiles_with_3_5_7_agents(s, n_agents, weights):
    specs = [demand_spec(), coverage_spec()] + [human_spec(f"Human {i}") for i in range(n_agents - 2)]
    p, ids = make_profile(s, list(zip(specs, weights)), name=f"{n_agents} agents")
    assert pp.validate_profile(s, p.profile_id).valid and len(pp.profile_view(s, p.profile_id).agents) == n_agents
    synthetic = [0, 0] + [60 + 5 * i for i in range(n_agents - 2)]  # SYNTHETIC TEST VALUES
    assess_all(s, ids, synthetic)
    run = dp.run_prioritization(s, p.profile_id, USER)
    for row in dp.ranking(s, run.run_id):
        results = dp.agent_results(s, run.run_id, row.requirement_id)
        assert len(results) == n_agents and row.priority_status == "COMPLETE"
        expected = sum(Decimal(str(r["score"])) * Decimal(r["weight_bp"]) / 10000 for r in results)
        assert row.final_score == float(expected.quantize(Decimal("0.01")))
    assert sorted(r.rank for r in dp.ranking(s, run.run_id)) == [1, 2, 3]


def test_recalculation_after_weight_change_keeps_history(s):
    p, ids = make_profile(s, [(demand_spec(), "50"), (coverage_spec(), "50")])
    old = dp.run_prioritization(s, p.profile_id, USER)
    old_scores = {r.requirement_id: r.final_score for r in dp.ranking(s, old.run_id)}
    pp.edit_profile(s, p.profile_id, USER, weights={ids[0]: "80", ids[1]: "20"})
    new = dp.run_prioritization(s, p.profile_id, USER)
    new_scores = {r.requirement_id: r.final_score for r in dp.ranking(s, new.run_id)}
    assert new.profile_version == 2 and old.profile_version == 1
    assert new_scores != old_scores
    assert {r.requirement_id: r.final_score for r in dp.ranking(s, old.run_id)} == old_scores  # history unchanged


def test_incomplete_requirements_are_not_ranked_and_weight_not_redistributed(s):
    p, ids = make_profile(s, [(demand_spec(), "60"), (human_spec("Impact"), "40")])
    first = req_ids(s)[0]
    dp.record_human_assessment(s, first, ids[1], 70, "Synthetic test rationale", "test.assessor")
    run = dp.run_prioritization(s, p.profile_id, USER)
    rows = {r.requirement_id: r for r in dp.ranking(s, run.run_id)}
    assert rows[first].priority_status == "COMPLETE" and rows[first].rank == 1
    for rid, row in rows.items():
        if rid != first:
            assert row.priority_status == "INCOMPLETE" and row.final_score is None and row.rank is None
            assert row.missing_agents == ["Impact"]
            demand = dp.agent_results(s, run.run_id, rid)[0]
            assert demand["weighted_contribution"] == pytest.approx(demand["score"] * 0.6, abs=1e-4)  # not rescaled


# --- 19-21 history, versions, profiles -----------------------------------------------------------


def test_historical_results_survive_agent_edits_and_removal(s):
    p, ids = make_profile(s, [(demand_spec(), 100)])
    run = dp.run_prioritization(s, p.profile_id, USER)
    before = {r["agent_name"]: r["score"] for r in dp.agent_results(s, run.run_id, req_ids(s)[0])}
    spec = ac.current_spec(s, ids[0])
    spec.name = "Renamed agent"
    ac.edit_agent(s, ids[0], spec, USER)
    ac.remove_agent(s, ids[0], USER)
    after = {r["agent_name"]: r["score"] for r in dp.agent_results(s, run.run_id, req_ids(s)[0])}
    assert after == before and "Customer Demand Agent" in after
    snap = json.loads(s.get(type(run), run.run_id).profile_snapshot_json)
    assert snap["agents"][0]["configuration_version"] == 1 and snap["agents"][0]["name"] == "Customer Demand Agent"
    assert s.scalar(select(AgentConfigVersion).where(AgentConfigVersion.agent_id == ids[0], AgentConfigVersion.version_number == 1))
    assert not pp.validate_profile(s, p.profile_id).valid  # removed agent can no longer run


def test_profiles_save_restore_duplicate_activate_archive(s, req_db, tmp_path):
    p, ids = make_profile(s, [(demand_spec(), "70"), (coverage_spec(), "30")], name="Customer Experience")
    pp.activate_profile(s, p.profile_id, USER)
    dup = pp.duplicate_profile(s, p.profile_id, USER)
    s.commit()
    view = pp.profile_view(s, p.profile_id)
    assert view.status == "active" and view.total_percent == "100.00"
    assert [(a["name"], a["weight_percent"]) for a in view.agents] == [("Customer Demand Agent", "70.00"),
                                                                       ("Evidence Coverage Agent", "30.00")]
    pp.activate_profile(s, dup.profile_id, USER)
    assert pp.get_profile(s, p.profile_id).status == "inactive" and pp.active_profile_id(s) == dup.profile_id
    pp.archive_profile(s, p.profile_id, USER)
    with pytest.raises(pp.ProfileError):
        pp.activate_profile(s, p.profile_id, USER)
    assert pp.profile_view(s, p.profile_id, version=1).agents[0]["weight_percent"] == "70.00"


def test_illustrative_starter_set_is_labelled(s):
    ac.seed_starter_agents(s, USER)
    prof = pp.seed_illustrative_profile(s, USER)
    view = pp.profile_view(s, prof.profile_id)
    assert view.illustrative and pp.ILLUSTRATIVE_NOTE in view.description and view.valid
    assert {a["scoring_method"] for a in view.agents} <= set(METHODS)
    market = next(a for a in ac.list_agents(s) if a["scoring_method"] == "VALIDATED_MARKET_RELEVANCE")
    assert market["status"] == "INACTIVE" and market["agent_id"] not in {a["agent_id"] for a in view.agents}


# --- 22-23 evidence validation --------------------------------------------------------------


def _add_link(s, rid, key):
    s.add(RequirementEvidenceLink(requirement_id=rid, tier="DIRECT", source_table="feedback", source_key=key,
                                  source_record_id=None, evidence_scope="cluster_member"))
    s.flush()


@pytest.mark.parametrize("bad_key, field", [("G60-9999", "invalid_ids"), ("F70-0001", "non_g60_ids"),
                                           ("G70-0001", "non_g60_ids")])
def test_invalid_or_foreign_evidence_is_rejected(s, bad_key, field):
    rid = req_ids(s)[0]
    _add_link(s, rid, bad_key)
    p, _ = make_profile(s, [(demand_spec(), "50"), (coverage_spec(), "50")])
    run = dp.run_prioritization(s, p.profile_id, USER)
    results = {r["agent_name"]: r for r in dp.agent_results(s, run.run_id, rid)}
    assert results["Customer Demand Agent"]["status"] == "UNSCORED"
    assert bad_key in results["Customer Demand Agent"]["inputs"][field]
    assert results["Evidence Coverage Agent"]["status"] == "UNSCORED"
    row = next(r for r in dp.ranking(s, run.run_id) if r.requirement_id == rid)
    assert row.priority_status == "INCOMPLETE" and row.rank is None


def test_market_relevance_and_ai_agents_return_unscored(s):
    market, _ = ac.create_agent(s, ac.AgentSpec(name="Market Relevance Agent", objective="Market relevance",
                                                execution_method="DATA_ANALYSIS", scoring_method="VALIDATED_MARKET_RELEVANCE",
                                                resources=["RES_G60_FEEDBACK", "RES_SALES", "RES_REQUIREMENTS"]), USER)
    ai, warnings = ac.create_agent(s, ac.AgentSpec(name="AI Agent", objective="Future", execution_method="AI_ANALYSIS",
                                                   scoring_method="AI_ASSESSMENT", resources=["RES_REQUIREMENTS"]), USER)
    assert AI_UNAVAILABLE in warnings
    p = pp.create_profile(s, "Unscored", {market.agent_id: 50, ai.agent_id: 50}, USER)
    run = dp.run_prioritization(s, p.profile_id, USER)
    results = {r["agent_name"]: r for r in dp.agent_results(s, run.run_id, req_ids(s)[0])}
    assert results["Market Relevance Agent"]["status"] == "UNSCORED" and results["Market Relevance Agent"]["score"] is None
    assert "not validated" in results["Market Relevance Agent"]["rationale"]
    assert results["AI Agent"]["rationale"] == AI_UNAVAILABLE


# --- 25-27 critical issues, decision context, zero requirements ---------------------------------------


def test_critical_flags_are_human_only_and_independent_of_rank(s):
    p, ids = make_profile(s, [(demand_spec(), 100)])
    run = dp.run_prioritization(s, p.profile_id, USER)
    lowest = max(dp.ranking(s, run.run_id), key=lambda r: r.rank)
    with pytest.raises(ci.FlagError):
        ci.raise_flag(s, lowest.requirement_id, "SAFETY", "x", USER, evidence_refs=["G60-9999"])
    key = s.scalar(select(RequirementEvidenceLink.source_key).where(RequirementEvidenceLink.requirement_id == lowest.requirement_id))
    flag = ci.raise_flag(s, lowest.requirement_id, "SAFETY", "Synthetic test flag", USER, evidence_refs=[key])
    assert flag.status == "PROPOSED"
    with pytest.raises(ci.FlagError):
        ci.decide_flag(s, flag.flag_id, "CONFIRMED", "test.lead", "")
    ci.decide_flag(s, flag.flag_id, "CONFIRMED", "test.lead", "Confirmed in synthetic test")
    run2 = dp.run_prioritization(s, p.profile_id, USER)
    row = next(r for r in dp.ranking(s, run2.run_id) if r.requirement_id == lowest.requirement_id)
    assert row.rank == lowest.rank and row.final_score == lowest.final_score  # score unchanged, never set to 100
    assert row.critical_flags[0]["status"] == "CONFIRMED"
    assert rr.get_requirement(s, lowest.requirement_id).review_status == "DRAFT"  # no auto-approval


def test_decision_context_matches_stored_run(s):
    p, ids = make_profile(s, [(demand_spec(), "50"), (coverage_spec(), "20"), (human_spec("Impact"), "30")])
    assess_all(s, ids, [0, 0, 75])  # SYNTHETIC TEST VALUE
    run = dp.run_prioritization(s, p.profile_id, USER)
    top = dp.ranking(s, run.run_id)[0]
    ctx = dc.get_decision_context(s, top.requirement_id)
    assert {"requirement_id", "profile_id", "run_id", "final_score", "rank", "agent_contributions", "evidence_references",
            "human_assessments", "assumptions", "limitations", "critical_flags"} <= set(ctx)
    assert ctx["run_id"] == run.run_id and ctx["final_score"] == top.final_score and ctx["rank"] == 1
    assert ctx["top_contributor"] == max(ctx["agent_contributions"], key=lambda c: c["weighted_contribution"])["agent_name"]
    assert sum(c["weighted_contribution"] for c in ctx["agent_contributions"]) == pytest.approx(top.final_score, abs=0.01)
    assert ctx["human_assessments"][0]["score"] == 75 and ctx["human_assessments"][0]["assessor_id"] == "test.assessor"
    kinds = {i["kind"] for c in ctx["agent_contributions"] for i in c["explanation"]["items"]}
    assert {"FACT", "ASSESSMENT", "ASSUMPTION", "LIMITATION"} <= kinds
    what_if = dc.what_if_weights(s, run.run_id, {ids[0]: 100, ids[1]: 0, ids[2]: 0})
    demand_only = {r["requirement_id"]: r["final_score"] for r in what_if["results"]}
    assert demand_only[top.requirement_id] == dp.agent_results(s, run.run_id, top.requirement_id)[0]["score"]
    assert dp.ranking(s, run.run_id)[0].final_score == top.final_score  # what-if did not write
    others = [r.requirement_id for r in dp.ranking(s, run.run_id)]
    cmp = dc.compare_requirements(s, run.run_id, others[0], others[1])
    assert cmp["final_score_a"] >= cmp["final_score_b"] and len(cmp["agents"]) == 3


def test_decision_context_without_run(s):
    ctx = dc.get_decision_context(s, req_ids(s)[0])
    assert ctx["run_id"] is None and ctx["final_score"] is None and ctx["agent_contributions"] == []


def test_zero_saved_requirements(empty_s):
    agent, _ = ac.create_agent(empty_s, demand_spec(), USER)
    p = pp.create_profile(empty_s, "Empty", {agent.agent_id: 100}, USER)
    with pytest.raises(dp.NoSavedRequirements, match="No saved requirements yet"):
        dp.run_prioritization(empty_s, p.profile_id, USER)


def test_priority_scores_do_not_touch_review_status(s):
    before = {r["requirement_id"]: r["status"] for r in rr.list_requirements(s)}
    p, _ = make_profile(s, [(demand_spec(), 100)])
    dp.run_prioritization(s, p.profile_id, USER)
    assert {r["requirement_id"]: r["status"] for r in rr.list_requirements(s)} == before


# --- 28-29 credentials, no LLM calls ---------------------------------------------------------------


@pytest.fixture()
def no_llm(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "PRODUCTMIND_ALLOW_EXTERNAL_LLM"):
        monkeypatch.delenv(var, raising=False)

    def forbid(*a, **k):
        raise AssertionError("No LLM client may be constructed in Part 5")

    monkeypatch.setattr("services.llm_client.AnthropicLLMClient.__init__", forbid)
    monkeypatch.setattr("anthropic.Anthropic.__init__", forbid, raising=False)


def test_full_flow_without_credentials_or_llm(s, no_llm):
    ac.seed_starter_agents(s, USER)
    prof = pp.seed_illustrative_profile(s, USER)
    pp.activate_profile(s, prof.profile_id, USER)
    run = dp.run_prioritization(s, prof.profile_id, USER)
    assert len(dp.ranking(s, run.run_id)) == 3
    assert s.scalar(select(AgentExecutionResult.result_id).where(AgentExecutionResult.run_id == run.run_id))


# --- Streamlit ---------------------------------------------------------------------------------


def _app(db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRODUCTMIND_DB_PATH", str(db))
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_page_handles_zero_requirements(base_db, tmp_path, monkeypatch, no_llm):
    path = tmp_path / "ui_empty.db"
    shutil.copy2(base_db, path)
    app = _app(path, monkeypatch)
    app.switch_page("ui/pages/agent_prioritization.py").run()
    assert not app.exception
    texts = [i.value for i in app.info]
    assert any("No saved requirements yet. Create a requirement in the Vehicle Requirements page." in t for t in texts)
    assert any("No agents yet" in t for t in texts)


def test_page_renders_runs_rankings_and_explanations(req_db, tmp_path, monkeypatch, no_llm):
    path = tmp_path / "ui_full.db"
    shutil.copy2(req_db, path)
    engine = get_engine(path)
    with session_scope(engine) as s:
        p, ids = make_profile(s, [(demand_spec(), "60"), (coverage_spec(), "40")])
        pp.activate_profile(s, p.profile_id, USER)
        run = dp.run_prioritization(s, p.profile_id, USER)
        rid = dp.ranking(s, run.run_id)[0].requirement_id
    engine.dispose()
    app = _app(path, monkeypatch)
    app.switch_page("ui/pages/agent_prioritization.py").run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "Ranked (complete)" in html and "Weighted contribution" in html
    assert any("TOTAL WEIGHT: 100.00%" in x.value for x in app.success)
    assert len(app.get("plotly_chart")) >= 3  # ranking + contribution + score comparison


def test_navigation_from_requirement_preserves_id(req_db, tmp_path, monkeypatch, no_llm):
    path = tmp_path / "ui_nav.db"
    shutil.copy2(req_db, path)
    engine = get_engine(path)
    with session_scope(engine) as s:
        agent, _ = ac.create_agent(s, demand_spec(), USER)
        pp.activate_profile(s, pp.create_profile(s, "P", {agent.agent_id: 100}, USER).profile_id, USER)
        rid = req_ids(s)[1]
    engine.dispose()
    app = _app(path, monkeypatch)
    app.session_state["prio_requirement"] = rid
    app.switch_page("ui/pages/agent_prioritization.py").run()
    assert not app.exception and app.selectbox(key="x_req").value == rid


def test_all_pages_load(req_db, tmp_path, monkeypatch, no_llm):
    path = tmp_path / "ui_all.db"
    shutil.copy2(req_db, path)
    app = _app(path, monkeypatch)
    app.run()
    for page in ("ui/pages/evidence_explorer.py", "ui/pages/customer_insights.py", "ui/pages/vehicle_requirements.py",
                 "ui/pages/agent_prioritization.py", "ui/pages/data_quality.py"):
        app.switch_page(page).run()
        assert not app.exception, (page, [e.value for e in app.exception])
