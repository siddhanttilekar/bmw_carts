"""Guided workflow screen 4: Decide with evidence-based explanations."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from services import agent_debate as deb
from services import decision_assistant as da
from services import dynamic_prioritization as dp
from services import management_reports as mr
from services import requirement_review as rr
from services import what_if_simulator as wis
from services.agent_chat import run_agent_options
from ui import data
from ui.evidence_ui import show_calculation
from ui.theme import apply_theme, badge, chip, esc, header, kpi, workflow

def METHOD_LABEL(method_id: str) -> str:  # noqa: N802
    from services.agent_scoring import METHODS

    return METHODS[method_id].label if method_id in METHODS else method_id


KIND_BADGE = {"FACT": "DIRECT", "ASSESSMENT": "SUPPORTING", "ASSUMPTION": "CONTEXTUAL",
              "LIMITATION": "PENDING", "SIMULATION": "REVIEW"}

apply_theme()
workflow("DECIDE")
header("Decide", "Understand, compare and export evidence-backed prioritization decisions.")

actor = st.text_input("Your user ID", value=st.session_state.get("workflow_actor", os.getenv("USER", "")),
                      key="workflow_actor")

with data.write_session() as s:
    requirements = rr.list_requirements(s)
    runs = dp.list_runs(s)

if not requirements:
    st.info("No requirements yet. Start by selecting a customer need.")
    if st.button("Go to Discover"):
        st.switch_page("ui/pages/discover.py")
    st.stop()
if not runs:
    st.info("Run your agents to generate an evidence-backed ranking.")
    if st.button("Go to Prioritize"):
        st.switch_page("ui/pages/prioritize.py")
    st.stop()

run_ids = [r["run_id"] for r in runs]
default_run = st.session_state.get("workflow_run_id")
run_id = st.selectbox("Prioritization run", run_ids,
                      index=run_ids.index(default_run) if default_run in run_ids else 0,
                      format_func=lambda rid: next(
                          f"{r['profile_name']} · {r['complete']} complete / {r['incomplete']} incomplete · {rid}"
                          for r in runs if r["run_id"] == rid),
                      key="decide_run")
st.session_state["workflow_run_id"] = run_id

with data.write_session() as s:
    rows = dp.ranking(s, run_id)
    agents = run_agent_options(s, run_id)
run_info = next(r for r in runs if r["run_id"] == run_id)

c1, c2, c3 = st.columns(3)
kpi(c1, run_info["complete"], "Complete assessments", "formal ranking available")
kpi(c2, run_info["incomplete"], "Incomplete assessments", "not formally ranked")
kpi(c3, len(agents), "Agents in run", "configuration is saved with the run")

complete = [r for r in rows if r.priority_status == "COMPLETE"]
incomplete = [r for r in rows if r.priority_status != "COMPLETE"]

st.markdown("### Top Prioritized Requirements")
if complete:
    for r in complete[:5]:
        with st.container(border=True):
            cols = st.columns([0.7, 3, 1.1, 1.1])
            cols[0].markdown(f"### #{r.rank}")
            cols[1].markdown(f"**{esc(r.title)}**  \n<span class='pm-src'>{esc(r.requirement_id)}</span>",
                             unsafe_allow_html=True)
            cols[2].metric("Priority score", r.final_score)
            cols[3].markdown(chip(r.review_status) + badge("SUPPORTING", "evidence linked"), unsafe_allow_html=True)
            with data.write_session() as s:
                results = dp.agent_results(s, run_id, r.requirement_id)
            with st.expander("Agent contributions and missing information"):
                st.dataframe(pd.DataFrame([{
                    "Agent": a["agent_name"],
                    "Status": a["status"],
                    "Score": a["score"],
                    "Weight %": a["weight_percent"],
                    "Contribution": a["weighted_contribution"],
                    "Evidence refs": len(a["source_refs"]),
                } for a in results]), hide_index=True, width="stretch")
                for a in results:
                    st.markdown(f"**{esc(a['agent_name'])}**", unsafe_allow_html=True)
                    show_calculation(a)
                if r.missing_agents:
                    st.warning("Missing: " + ", ".join(r.missing_agents))
else:
    st.info("No complete requirements in this run. Complete the missing assessments in Prioritize.")
if incomplete:
    with st.expander(f"Not ranked yet ({len(incomplete)})"):
        st.dataframe(pd.DataFrame([{"Requirement": r.title, "Missing information": ", ".join(r.missing_agents),
                                    "Approval status": r.review_status} for r in incomplete]),
                     hide_index=True, width="stretch")

req_ids = [r.requirement_id for r in rows]
titles = {r.requirement_id: r.title for r in rows}
selected_req = st.session_state.get("workflow_requirement_id")
rid = st.selectbox("Selected requirement", req_ids,
                   index=req_ids.index(selected_req) if selected_req in req_ids else 0,
                   format_func=lambda x: f"{titles[x]} · {x}",
                   key="decide_req")
st.session_state["workflow_requirement_id"] = rid
agent_names = {a["agent_id"]: a["name"] for a in agents}


def act(fn):
    try:
        with data.write_session() as s:
            return fn(s)
    except (ValueError, RuntimeError, PermissionError) as exc:
        st.error(str(exc))
        return None


def render_response(resp) -> None:
    st.markdown(badge("SUPPORTING", "Evidence-based Decision Assistant") + chip(f"run {resp.run_id}")
                + chip(resp.intent.replace("_", " ")), unsafe_allow_html=True)
    for note in resp.notices:
        st.info(note)
    for item in resp.items:
        st.markdown(badge(KIND_BADGE.get(item.kind, "SUPPORTING"), item.kind) + esc(item.text),
                    unsafe_allow_html=True)
    if resp.suggestions:
        st.caption("Suggested next questions: " + " · ".join(resp.suggestions))
    if resp.evidence_refs:
        with st.expander(f"Validated evidence references ({len(resp.evidence_refs)})"):
            st.dataframe(pd.DataFrame(resp.evidence_refs), hide_index=True, width="stretch")


tabs = st.tabs(["Explain This Decision", "Compare Requirements", "Agent Debate", "What-If Simulation",
                "History & Reanalysis", "Human Approval", "Export Report"])

with tabs[0]:
    st.markdown("#### Evidence-based Decision Assistant")
    st.caption("Offline, structured answers from saved scores and evidence. This is not unrestricted AI chat.")
    agent_id = st.selectbox("Agent", [None] + list(agent_names), format_func=lambda x: "All agents" if x is None else agent_names[x],
                            key="decide_agent")
    questions = [
        ("Why is this ranked first?", "explain_priority", {}),
        ("Which agent contributed most?", "explain_priority", {}),
        ("What evidence supports this?", "evidence", {}),
        ("What information is missing?", "missing", {}),
        ("How reliable is this recommendation?", "reliability", {}),
    ]
    cols = st.columns(3)
    for idx, (label, intent, kwargs) in enumerate(questions):
        if cols[idx % 3].button(label, key=f"decide_q_{intent}_{idx}"):
            out = act(lambda s, i=intent, kw=kwargs: da.ask(s, rid, run_id, actor, intent=i, **kw))
            if out:
                st.session_state[f"decide_resp_{run_id}_{rid}"] = out[1].model_dump_json()
    free = st.text_input("Ask a suggested-style question", key="decide_free")
    if st.button("Ask", disabled=not free.strip()):
        out = act(lambda s: da.ask(s, rid, run_id, actor, question=free, agent_id=agent_id))
        if out:
            st.session_state[f"decide_resp_{run_id}_{rid}"] = out[1].model_dump_json()
    if raw := st.session_state.get(f"decide_resp_{run_id}_{rid}"):
        from services.decision_grounding import AssistantResponse

        render_response(AssistantResponse.model_validate_json(raw))

with tabs[1]:
    if len(req_ids) < 2:
        st.info("Comparison needs two requirements in the selected run.")
    else:
        other = st.selectbox("Compare with", [x for x in req_ids if x != rid],
                             format_func=lambda x: f"{titles[x]} · {x}", key="decide_compare")
        if st.button("Compare Requirements"):
            out = act(lambda s: da.ask(s, rid, run_id, actor, intent="compare", other_requirement_id=other))
            if out:
                render_response(out[1])

with tabs[2]:
    if len(req_ids) < 2:
        st.info("An agent debate needs two requirements.")
    else:
        other = st.selectbox("Debate against", [x for x in req_ids if x != rid],
                             format_func=lambda x: f"{titles[x]} · {x}", key="decide_debate")
        if st.button("Run Agent Debate", type="primary"):
            debate = act(lambda s: deb.run_debate(s, run_id, rid, other))
            if debate:
                st.markdown("#### Agent positions")
                st.dataframe(pd.DataFrame([{
                    "Agent": p.agent_name,
                    "Position": p.position.replace("_", " ").title(),
                    "Contribution A": p.contribution_a,
                    "Contribution B": p.contribution_b,
                    "Basis A": p.basis_a,
                    "Basis B": p.basis_b,
                } for p in debate.round1_positions]), hide_index=True, width="stretch")
                st.markdown("#### Each agent's perspective (saved snapshot - no new scores)")
                for p in debate.round1_positions:
                    with st.container(border=True):
                        st.markdown(f"**{esc(p.agent_name)}** · {esc(METHOD_LABEL(p.scoring_method))} · weight "
                                    f"{p.weight_percent}% · resources: {esc(', '.join(p.resources))}", unsafe_allow_html=True)
                        if p.abstains:
                            st.info("Abstains - " + (p.abstain_reason or "insufficient evidence"))
                        ca, cb = st.columns(2)
                        for col, side, calc, ev, counter in ((ca, "A", p.calculation_a, p.evidence_summary_a, p.counter_evidence_a),
                                                             (cb, "B", p.calculation_b, p.evidence_summary_b, p.counter_evidence_b)):
                            col.markdown(f"*Requirement {side}*")
                            for line in calc + ev:
                                col.caption(line)
                            if counter is not None:
                                col.caption(f"Counter-evidence identified: {counter} comments")
                st.markdown("#### Synthesis")
                st.write(debate.round4_synthesis["agreement"])
                st.write(debate.round4_synthesis["ranking_under_profile"])
                for m in debate.round4_synthesis["missing_evidence"]:
                    st.caption(f"Missing information: {m}")

with tabs[3]:
    st.caption("Simulation only. Official runs, profiles and assessments are not changed.")
    with data.write_session() as s:
        base_weights = wis.current_weights(s, run_id)
    weights = {}
    for aid, weight in base_weights.items():
        weights[aid] = st.number_input(agent_names.get(aid, aid), 0.0, 100.0, float(weight), 0.5,
                                       key=f"decide_wi_{run_id}_{aid}")
    total = sum(weights.values())
    (st.success if round(total, 2) == 100 else st.warning)(f"Weight total: {total:.2f}%")
    if st.button("Run What-If", disabled=round(total, 2) != 100):
        sim = act(lambda s: wis.simulate(s, run_id, weights))
        if sim:
            st.dataframe(pd.DataFrame([{
                "Requirement": r.requirement_id,
                "Official rank": r.original_rank,
                "Simulated rank": r.simulated_rank,
                "Official score": r.original_final,
                "Simulated score": r.simulated_final,
                "Status": r.simulated_status,
            } for r in sim.rows]), hide_index=True, width="stretch")

with tabs[4]:
    st.caption("Saved runs are snapshots: reopening a run never re-runs evidence selection or an LLM.")
    with data.write_session() as s:
        snap = dp.evidence_snapshot(s, run_id)
    if snap:
        st.dataframe(pd.DataFrame([{
            "Requirement": titles.get(k, k), "Selection": v["selection_id"],
            "Observation version": (v.get("observation_version") or {}).get("version_id"),
            "Model": (v.get("observation_version") or {}).get("model"),
            "Prompt": (v.get("observation_version") or {}).get("prompt_version"),
            "Complaints": v["counts"]["complaints"], "Requests": v["counts"]["improvement_requests"],
            "Unique supporting": v["counts"]["combined_demand"], "Counter-evidence": v["counts"]["counter_evidence"],
            "Uncertain": v["counts"]["uncertain"]} for k, v in snap.items()]), hide_index=True, width="stretch")
    else:
        st.caption("This run used no AI evidence (no AI analysis agents, or no evidence was available).")
    h1, h2 = st.columns(2)
    if h1.button("Verify reproducibility"):
        with data.write_session() as s:
            rep = dp.verify_reproducibility(s, run_id)
        (st.success if rep["reproducible"] else st.error)(
            f"{'Reproducible' if rep['reproducible'] else 'Mismatch'}: {rep['agent_scores_checked']} agent scores and "
            f"{rep['requirements_checked']} final scores recomputed from stored inputs only.")
        if rep["mismatches"]:
            st.dataframe(pd.DataFrame(rep["mismatches"]), hide_index=True)
    if h2.button("Reanalyze With New Evidence", type="primary", disabled=not actor.strip()):
        out = act(lambda s: dp.reanalyze_with_new_evidence(s, run_id, actor))
        if out:
            new_run, comparison = out
            st.session_state["decide_comparison"] = comparison
            st.success(f"New run {new_run.run_id} created; run {run_id} is unchanged.")
    other_runs = [r for r in run_ids if r != run_id]
    if other_runs:
        cmp_with = st.selectbox("Compare this run with an earlier run", other_runs, key="decide_cmp_run")
        if st.button("Compare runs"):
            with data.write_session() as s:
                st.session_state["decide_comparison"] = dp.compare_runs(s, cmp_with, run_id)
    if comp := st.session_state.get("decide_comparison"):
        st.markdown(f"#### Previous {comp['old_run_id']} → updated {comp['new_run_id']}")
        st.dataframe(pd.DataFrame([{
            "Requirement": r["title"], "Old score": r["old_score"], "New score": r["new_score"],
            "Old rank": r["old_rank"], "New rank": r["new_rank"], "Old status": r["old_status"],
            "New status": r["new_status"], "Evidence changed": r["evidence_changed"],
            "Old unique supporting": (r["old_evidence_counts"] or {}).get("combined_demand"),
            "New unique supporting": (r["new_evidence_counts"] or {}).get("combined_demand"),
        } for r in comp["requirements"]]), hide_index=True, width="stretch")
        for r in comp["requirements"]:
            with st.expander(f"Agent changes · {r['title']}"):
                st.dataframe(pd.DataFrame(r["agents"]), hide_index=True, width="stretch")

with tabs[5]:
    st.caption("Approval is a human decision. Priority scores and agents never change a requirement's status.")
    with data.write_session() as s:
        req_now = rr.requirement_as_dict(s, rr.get_requirement(s, rid))
    st.markdown(chip(f"Status: {req_now['review_status']}") + chip(f"version {req_now['current_version']}"),
                unsafe_allow_html=True)
    reason = st.text_input("Decision reason (required)", key=f"decide_reason_{rid}")
    b1, b2, b3, b4 = st.columns(4)
    for col, label, fn in ((b1, "Submit for Review", rr.submit_for_review), (b2, "Approve", rr.approve),
                           (b3, "Reject", rr.reject), (b4, "Reopen", rr.reopen)):
        if col.button(label, key=f"decide_{label}_{rid}", disabled=not (reason.strip() and actor.strip())):
            try:
                with data.write_session() as s:
                    fn(s, rid, actor, reason)
                st.success(f"{label}: done.")
                st.rerun()
            except (rr.InvalidTransition, ValueError) as exc:
                st.error(str(exc))
    with data.write_session() as s:
        hist = rr.history(s, rid)
    if hist:
        st.dataframe(pd.DataFrame([h.model_dump() for h in hist]), hide_index=True, width="stretch")

with tabs[6]:
    st.caption("Reports are generated from the selected saved run. Verbatim feedback is excluded by default.")
    fmt = st.radio("Format", ["xlsx", "csv", "pdf"], horizontal=True)
    include_challenges = st.checkbox("Include challenges", value=False)
    include_debates = st.checkbox("Include saved debates", value=False)
    if st.button("Prepare Report", type="primary"):
        out = act(lambda s: mr.export_report(
            s, run_id, fmt, actor,
            mr.ReportOptions(include_challenges=include_challenges, include_debates=include_debates),
        ))
        if out:
            content, name = out
            st.download_button("Download Report", data=content, file_name=name, mime="application/octet-stream")
