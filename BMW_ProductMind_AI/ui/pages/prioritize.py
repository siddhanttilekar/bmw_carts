"""Guided workflow screen 3: Prioritize requirements."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from core.prioritization_models import EXECUTION_METHODS
from services import agent_configuration as ac
from services import agent_documents as adocs
from services import ai_evidence_selector as sel
from services import ai_feedback_analysis as ai
from services import custom_metrics as cmx
from services import dynamic_prioritization as dp
from services import prioritization_profiles as pp
from services import requirement_review as rr
from services.agent_resources import RESOURCES, list_resources
from services.agent_scoring import METHODS, check_compatibility, methods_for
from services.agent_weights import BP_TOTAL, bp_to_percent, percent_to_bp
from ui import data
from ui.evidence_ui import llm_settings, show_calculation
from ui.theme import apply_theme, badge, card, chip, esc, header, kpi, workflow

apply_theme()
workflow("PRIORITIZE")
header("Prioritize", "Evaluate saved requirements with configurable analysis agents.")

actor = st.text_input("Your user ID", value=st.session_state.get("workflow_actor", os.getenv("USER", "")),
                      key="workflow_actor")


def act(fn, success: str | None = None):
    try:
        with data.write_session() as s:
            out = fn(s)
        st.cache_data.clear()
        if success:
            st.success(success)
        return out if out is not None else True
    except (ValueError, PermissionError) as exc:
        st.error(str(exc))
        return None


with data.write_session() as s:
    agents = ac.list_agents(s)
    profiles = pp.list_profiles(s)
    requirements = rr.list_requirements(s)
    runs = dp.list_runs(s)
    resources = list_resources(s)

if not requirements:
    st.info("No requirements yet. Start by selecting a customer need.")
    if st.button("Go to Discover"):
        st.switch_page("ui/pages/discover.py")
    st.stop()

active_agents = [a for a in agents if a["status"] == "ACTIVE"]
valid_profiles = [p for p in profiles if p.valid and p.status != "archived"]
active_profile = next((p for p in profiles if p.status == "active"), None)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, len(requirements), "Requirements ready", "saved requirement drafts")
kpi(c2, len(active_agents), "Active agents", f"{len(agents)} configured")
kpi(c3, active_profile.total_percent + "%" if active_profile else "-", "Active weight total", "must equal 100%")
kpi(c4, len(runs), "Analyses run", "saved prioritization runs")

st.markdown("### Requirement Ranking")
if runs:
    run_ids = [r["run_id"] for r in runs]
    default_run = st.session_state.get("workflow_run_id")
    run_id = st.selectbox("Prioritization run", run_ids, index=run_ids.index(default_run) if default_run in run_ids else 0,
                          format_func=lambda rid: next(
                              f"{r['profile_name']} · {r['complete']} complete / {r['incomplete']} incomplete · {rid}"
                              for r in runs if r["run_id"] == rid),
                          key="prio_run_select")
    st.session_state["workflow_run_id"] = run_id
    with data.write_session() as s:
        rows = dp.ranking(s, run_id)
    complete = [r for r in rows if r.priority_status == "COMPLETE"]
    incomplete = [r for r in rows if r.priority_status != "COMPLETE"]
    if complete:
        st.dataframe(pd.DataFrame([{
            "Rank": r.rank,
            "Requirement name": r.title,
            "Priority score": r.final_score,
            "Evidence status": "Complete",
            "Review status": r.review_status,
            "Requirement ID": r.requirement_id,
        } for r in complete]), hide_index=True, width="stretch")
    else:
        st.info("No complete assessments in this run, so no formal ranking is shown.")
    if incomplete:
        with st.expander(f"Incomplete assessments ({len(incomplete)})"):
            st.dataframe(pd.DataFrame([{
                "Requirement name": r.title,
                "Missing information": ", ".join(r.missing_agents),
                "Review status": r.review_status,
                "Requirement ID": r.requirement_id,
            } for r in incomplete]), hide_index=True, width="stretch")
    st.markdown("#### Score calculation")
    calc_ids = [r.requirement_id for r in rows]
    calc_titles = {r.requirement_id: r.title for r in rows}
    calc_rid = st.selectbox("Show the calculation for", calc_ids, format_func=lambda x: f"{calc_titles[x]} · {x}",
                            key="prio_calc_req")
    with data.write_session() as s:
        calc_results = dp.agent_results(s, run_id, calc_rid)
    calc_row = next(r for r in rows if r.requirement_id == calc_rid)
    for res in calc_results:
        with st.container(border=True):
            st.markdown(f"**{esc(res['agent_name'])}** · {esc(METHODS[res['scoring_method']].label)} "
                        f"v{esc(res['scoring_version'])} · {res['status']}", unsafe_allow_html=True)
            show_calculation(res)
    if calc_row.final_score is not None:
        st.success(f"Priority score = sum of weighted contributions = {calc_row.final_score} (calculated by Python; "
                   f"rank {calc_row.rank}).")
    else:
        st.warning("INCOMPLETE - missing: " + ", ".join(calc_row.missing_agents) + ". Missing weight is not redistributed.")
else:
    st.info("Run your agents to generate an evidence-backed ranking.")

st.markdown("### Active Agents")
if not agents:
    st.info("No prioritization agents configured.")
    if st.button("Load Starter Configuration", type="primary"):
        act(lambda s: (ac.seed_starter_agents(s, actor), pp.seed_illustrative_profile(s, actor)),
            "Starter agents and illustrative weights loaded.")
        st.rerun()
else:
    resource_names = {r.resource_id: r.name for r in resources}
    profile_for_cards = active_profile or (profiles[0] if profiles else None)
    weights = {a["agent_id"]: a["weight_percent"] for a in profile_for_cards.agents} if profile_for_cards else {}
    for row in range(0, len(active_agents), 3):
        cols = st.columns(3)
        for col, agent in zip(cols, active_agents[row:row + 3]):
            col.markdown(card(agent["name"], agent["objective"],
                              f"Method: {agent['execution_method'].replace('_', ' ').title()}",
                              f"Scoring: {METHODS[agent['scoring_method']].label}",
                              f"Weight: {weights.get(agent['agent_id'], agent['default_weight_percent'])}%",
                              f"Resources: {len(agent['resources'])}"),
                         unsafe_allow_html=True)
            with col.expander("Details"):
                st.write(agent["description"] or agent["objective"])
                st.caption("Resources: " + ", ".join(resource_names.get(r, r) for r in agent["resources"]))
                st.caption("Scoring: " + agent["scoring_method"].replace("_", " ").title())
                if agent.get("context"):
                    st.caption("Context: " + agent["context"])
                if agent.get("instructions"):
                    st.caption("Custom instructions (scope notes, never executed): " + agent["instructions"][:300])
                if agent.get("documents"):
                    st.caption("Uploaded documents: " + ", ".join(d["file_name"] for d in agent["documents"]))

# ---------------------------------------------------------------------------
# Step 2: OpenAI-powered evidence analysis for AI agents
# ---------------------------------------------------------------------------
ai_agents_all = [a for a in active_agents if a["execution_method"] == "AI_ANALYSIS"]
if ai_agents_all:
    from services import agent_analysis as aa
    from services.agent_scoring import run_method
    from ui.evidence_ui import paid_call_blocker, paid_client

    st.markdown("### Run AI Analysis")
    st.caption("AI agents read only their assigned resources. The AI identifies relevant evidence; "
               "Python counts unique comments and calculates every score.")
    blocker = paid_call_blocker()
    if blocker:
        st.warning("AI unavailable — configure OpenAI API in application settings.")
    ai_req_ids = [r["requirement_id"] for r in requirements]
    ai_rid = st.selectbox("Requirement to evaluate", ai_req_ids,
                          format_func=lambda i: next(r["title"] for r in requirements if r["requirement_id"] == i),
                          key="ai_req_select")
    plans, plan_errors = [], {}
    with data.write_session() as s:
        for a in ai_agents_all:
            try:
                plans.append(aa.plan_analysis(s, llm_settings(), a["agent_id"], ai_rid, actor or "system"))
            except aa.AnalysisError as exc:
                plan_errors[a["name"]] = str(exc)
    for name, err in plan_errors.items():
        st.info(f"{name}: {err}")
    to_run = [p for p in plans if p.cached is None and p.requests]
    total = sum(p.estimate.estimated_usd for p in to_run)
    upper = sum(p.estimate.upper_bound_usd for p in to_run)
    if plans:
        reused = [p.agent["name"] for p in plans if p.cached is not None]
        st.write(f"Estimated cost: ${total:.4f} (up to ${upper:.4f}) for {len(to_run)} agent(s)"
                 + (f" · saved analysis reused at no cost: {', '.join(reused)}" if reused else ""))
        approve = st.number_input("Approve up to (USD)", 0.0, 5.0, float(round(upper + 0.005, 2)), 0.01,
                                  key="ai_approve")
        if st.button("Run AI Analysis", type="primary",
                     disabled=bool(blocker) or not to_run or not actor.strip()):
            try:
                llm = paid_client()
                with st.spinner("Analysing evidence"), data.write_session() as s:
                    remaining = approve
                    for p in to_run:
                        p = aa.plan_analysis(s, llm_settings(), p.agent["agent_id"], ai_rid, actor)
                        row = aa.run_analysis(s, llm, llm_settings(), p, actor, approved_max_usd=remaining)
                        remaining -= row.cost_usd
                        if row.status != "completed":
                            st.error(f"{p.agent['name']}: {row.error}")
                st.rerun()
            except (aa.ApprovalRequired, aa.BudgetExceeded, aa.AnalysisError) as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - never surface request content or keys
                st.error(f"AI analysis stopped: {type(exc).__name__}. Nothing was double-charged; retry later.")
        weights_now = {a["agent_id"]: float(a["weight_percent"]) for a in active_profile.agents} if active_profile else {}
        with data.write_session() as s:
            req_dict = rr.requirement_as_dict(s, rr.get_requirement(s, ai_rid))
            for p in plans:
                agent = p.agent
                res = run_method(s, req_dict, agent)
                inp = res.inputs
                counts = inp.get("evidence_counts") or {}
                weight = weights_now.get(agent["agent_id"])
                contribution = (round(res.score * weight / 100, 2)
                                if res.score is not None and weight is not None else None)
                with st.container(border=True):
                    st.markdown(f"**{esc(agent['name'])}** · {esc(p.requirement_title)}", unsafe_allow_html=True)
                    basis = inp.get("evidence_basis")
                    used = ("This agent's AI analysis" if basis == "agent_ai_analysis" else
                            "Taxonomy matching only - run AI Analysis for this agent" if basis else "No evidence yet")
                    docs = inp.get("document_refs") or []
                    c = st.columns(5)
                    c[0].metric("Unique complaints", counts.get("complaints", "-"))
                    c[1].metric("Unique improvement requests", counts.get("improvement_requests", "-"))
                    c[2].metric("Agent score", res.score if res.score is not None else "UNSCORED")
                    c[3].metric("Weight", f"{weight:g}%" if weight is not None else "not in profile")
                    c[4].metric("Weighted contribution", contribution if contribution is not None else "-")
                    st.caption(f"Evidence used: {used}" + (f" · {len(docs)} document passages" if docs else "")
                               + f" · Scoring method: {METHODS[agent['scoring_method']].label}")
                    if res.status != "SCORED":
                        st.warning(res.rationale)
                    for m in inp.get("missing_information", []):
                        st.caption(f"Missing information: {m}")
                    with st.expander("Evidence details and calculation"):
                        show_calculation({"status": res.status, "score": res.score, "inputs": inp,
                                          "rationale": res.rationale, "weight_percent": weight,
                                          "weighted_contribution": contribution})
                        row = aa.current_analysis(s, agent, req_dict)
                        if row is not None:
                            result = json.loads(row.result_json)
                            st.dataframe(pd.DataFrame([{
                                "Comment": o["feedback_id"], "Type": o["observation_type"].replace("_", " ").title(),
                                "Excerpt": o["evidence_excerpt"], "AI judgement": o["relevance"],
                                "Reason": o["reason"]} for o in result["observations"]]), hide_index=True, width="stretch")
                            if result["documents"]:
                                st.dataframe(pd.DataFrame([{"Location": d["location"], "Passage": d["excerpt"],
                                                            "AI judgement": d["relevance"], "Reason": d["reason"]}
                                                           for d in result["documents"]]),
                                             hide_index=True, width="stretch")
                            st.caption(f"Analysis {row.analysis_id} · {row.model} · cost ${row.cost_usd:.4f} (estimate) "
                                       f"· {row.created_at:%Y-%m-%d %H:%M}")

st.markdown("### Run Analysis")
if not active_agents:
    if st.button("Create Starter Agents"):
        act(lambda s: (ac.seed_starter_agents(s, actor), pp.seed_illustrative_profile(s, actor)),
            "Starter agents and illustrative profile created.")
        st.rerun()
elif not valid_profiles:
    st.warning("No executable profile yet. Agent weights must total exactly 100%.")
    if st.button("Load Starter Configuration"):
        act(lambda s: (ac.seed_starter_agents(s, actor), pp.seed_illustrative_profile(s, actor)),
            "Starter profile created.")
        st.rerun()
else:
    req_ids = [r["requirement_id"] for r in requirements]
    selected_req = st.session_state.get("workflow_requirement_id") or st.session_state.get("prio_requirement")
    rid = st.selectbox("Requirement", req_ids,
                       index=req_ids.index(selected_req) if selected_req in req_ids else 0,
                       format_func=lambda i: next(f"{r['title']} · {r['status']} · {i}" for r in requirements
                                                  if r["requirement_id"] == i),
                       key="prio_req_select")
    st.session_state["workflow_requirement_id"] = rid
    default_profile = active_profile.profile_id if active_profile and active_profile.valid else valid_profiles[0].profile_id
    profile_id = st.selectbox("Profile", [p.profile_id for p in valid_profiles],
                              index=[p.profile_id for p in valid_profiles].index(default_profile),
                              format_func=lambda pid: next(f"{p.profile_name} · {p.total_percent}% · {p.status}"
                                                           for p in valid_profiles if p.profile_id == pid),
                              key="prio_profile_select")
    profile = next(p for p in valid_profiles if p.profile_id == profile_id)
    st.success(f"Total active weight: {profile.total_percent}%")

    ai_agents = [a for a in profile.agents if a["scoring_method"] in ("COMPLAINT_FREQUENCY", "CUSTOMER_REQUEST_FREQUENCY",
                                                                       "COMBINED_CUSTOMER_DEMAND")]
    if ai_agents:
        version_id = ai.current_observation_version_id(llm_settings())
        with data.write_session() as s:
            ready = []
            for r in requirements:
                cur_sel = sel.latest_selection(s, r["requirement_id"], version_id,
                                               requirement_version=r["version"])
                ready.append({"Requirement": r["title"], "Evidence selection": cur_sel.selection_id if cur_sel else "missing",
                              "Coverage": (f"{cur_sel.coverage['extracted_comments']} / {cur_sel.coverage['direct_comments']}"
                                           if cur_sel else "-"),
                              "Unique supporting comments": cur_sel.counts()["combined_demand"] if cur_sel else None})
        with st.expander("AI evidence readiness (AI analysis agents need a complete selection)"):
            st.dataframe(pd.DataFrame(ready), hide_index=True, width="stretch")
            if st.button("Select evidence for all requirements (no API cost)", disabled=not actor.strip()):
                with data.write_session() as s:
                    for r in requirements:
                        sel.run_selection(s, r["requirement_id"], version_id, actor)
                st.rerun()

    human_agents = [a for a in profile.agents if a["scoring_method"] == "HUMAN_SCORE"]
    if human_agents:
        with st.expander("Human assessments required"):
            with data.write_session() as s:
                refs = [e["source_key"] for e in rr.requirement_as_dict(s, rr.get_requirement(s, rid))["evidence"]]
            for agent in human_agents:
                st.markdown(f"**{esc(agent['name'])}**")
                score = st.number_input("Score", 0, 100, None, key=f"prio_hs_{rid}_{agent['agent_id']}")
                rationale = st.text_area("Rationale", key=f"prio_hr_{rid}_{agent['agent_id']}")
                ev = st.multiselect("Evidence references", refs, key=f"prio_he_{rid}_{agent['agent_id']}")
                if st.button("Save assessment", key=f"prio_hb_{rid}_{agent['agent_id']}", disabled=score is None):
                    act(lambda s, a=agent: dp.record_human_assessment(s, rid, a["agent_id"], score, rationale, actor, ev),
                        f"Assessment saved for {agent['name']}.")

    b1, b2, b3 = st.columns(3)
    if b1.button("Run Analysis", type="primary"):
        run_id = act(lambda s: dp.run_prioritization(s, profile_id, actor, [rid]).run_id, "Analysis complete.")
        if run_id:
            st.session_state["workflow_run_id"] = run_id
            st.rerun()
    if b2.button(f"Run All Requirements ({len(requirements)})"):
        run_id = act(lambda s: dp.run_prioritization(s, profile_id, actor).run_id, "Analysis complete.")
        if run_id:
            st.session_state["workflow_run_id"] = run_id
            st.rerun()
    if b3.button("Continue to Decide", disabled="workflow_run_id" not in st.session_state):
        st.switch_page("ui/pages/decide.py")

with st.expander("Add or edit agents and weights"):
    st.caption("Advanced configuration stays here for quick edits. AI agents identify evidence only; "
               "every score is calculated in Python.")
    if st.button("Load Starter Configuration", key="prio_seed_bottom"):
        act(lambda s: (ac.seed_starter_agents(s, actor), pp.seed_illustrative_profile(s, actor)),
            "Starter configuration loaded.")
        st.rerun()
    if profiles:
        pids = [p.profile_id for p in profiles if p.status != "archived"]
        pid = st.selectbox("Edit profile weights", pids, format_func=lambda x: next(
            f"{p.profile_name} · {p.total_percent}%" for p in profiles if p.profile_id == x), key="prio_weight_profile")
        prof = next(p for p in profiles if p.profile_id == pid)
        new_weights = {}
        for agent in prof.agents:
            new_weights[agent["agent_id"]] = st.number_input(agent["name"], 0.0, 100.0,
                                                             float(agent["weight_percent"]), 0.5,
                                                             key=f"prio_w_{pid}_{agent['agent_id']}")
        total_bp = sum(percent_to_bp(v) for v in new_weights.values()) if new_weights else 0
        (st.success if total_bp == BP_TOTAL else st.warning)(f"Weight total: {bp_to_percent(total_bp)}%")
        if st.button("Save weights", disabled=not new_weights):
            act(lambda s: pp.edit_profile(s, pid, actor, weights=new_weights), "Weights saved as a new profile version.")
            st.rerun()


# ---------------------------------------------------------------------------
# 2.0 custom agent builder (seven fields)
# ---------------------------------------------------------------------------
with st.expander("Build a Custom Agent"):
    with data.write_session() as s:
        metrics = cmx.list_custom_metrics(s)
    name = st.text_input("Agent Name", key="cab_name")
    objective = st.text_area("Objective", height=70, key="cab_obj",
                             placeholder="e.g. Evaluate how often customers complain about or request this capability.")
    instructions = st.text_area(
        "Custom Instructions", height=90, key="cab_instr", max_chars=ac.MAX_INSTRUCTIONS,
        help="Scope notes for AI evidence selection (for example 'focus on touchscreen usability'). They are plain text: "
             "never executed as code or SQL, and they never change a formula.")
    res_ids = [r for r in RESOURCES]
    resources_sel = st.multiselect("Add Resources", res_ids, format_func=lambda r: RESOURCES[r].name, key="cab_res")
    uploads = st.file_uploader("Upload documents (PDF, XLSX, CSV, TXT, MD)", type=list(adocs.ALLOWED_TYPES),
                               accept_multiple_files=True, key="cab_upl",
                               help="Visible to this agent only. Treated as untrusted text; never executed.")
    context = st.text_input("Context", key="cab_ctx", max_chars=ac.MAX_CONTEXT,
                            placeholder="e.g. Next-generation BMW 5 Series")
    weight = st.number_input("Weight (%)", 0.0, 100.0, 0.0, 0.5, key="cab_w")
    execution = st.radio("Execution Method", list(EXECUTION_METHODS), horizontal=True, key="cab_exe",
                         format_func=lambda m: m.replace("_", " ").title())
    rule_ids = methods_for(execution)
    rule = st.selectbox("Approved scoring rule", rule_ids, format_func=lambda m: METHODS[m].label, key="cab_rule")
    st.caption(f"{METHODS[rule].description} Formula: {METHODS[rule].formula}")
    metric_id = None
    if rule == "DOCUMENTED_CUSTOM_METRIC":
        metric_id = st.selectbox("Documented metric", [m["metric_id"] for m in metrics], key="cab_metric",
                                 format_func=lambda m: next(f"{x['label']} ({'approved' if x['approved'] else 'NOT approved - UNSCORED'})"
                                                            for x in metrics if x["metric_id"] == m))
    final_resources = list(dict.fromkeys(resources_sel + (["RES_UPLOADED_DOCUMENTS"] if uploads else [])))
    errors, warnings = check_compatibility(execution, rule, final_resources)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.caption(w)
    add_to_profile = st.checkbox("Add to the active profile (rebalance weights to 100% afterwards)", value=True,
                                 key="cab_add", disabled=active_profile is None)
    if st.button("Create Agent", type="primary", disabled=bool(errors) or not name.strip() or not objective.strip()):
        def _create(s):
            agent, _ = ac.create_agent(s, ac.AgentSpec(
                name=name, objective=objective, instructions=instructions.strip() or None, resources=final_resources,
                context=context, weight_percent=weight, execution_method=execution, scoring_method=rule,
                custom_metric_id=metric_id), actor)
            for f in uploads or []:
                adocs.upload_document(s, agent.agent_id, f.name, f.getvalue(), actor)
            if add_to_profile and active_profile is not None:
                current = {a["agent_id"]: a["weight_percent"] for a in active_profile.agents}
                pp.edit_profile(s, active_profile.profile_id, actor, weights={**current, agent.agent_id: weight})
            return agent.agent_id
        try:
            with data.write_session() as s:
                new_id = _create(s)
            st.success(f"Agent {new_id} created." + (" Rebalance the profile weights to 100% before running."
                                                     if add_to_profile else ""))
            st.rerun()
        except (ac.AgentConfigError, adocs.DocumentError, ValueError) as exc:
            st.error(str(exc))
