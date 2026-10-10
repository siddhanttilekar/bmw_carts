"""Decision Intelligence (Part 6): overview, offline decision assistant and
agent chat, structured debate, what-if simulation + scenario comparison, and
management reports. Offline by default; nothing here changes official scores,
agent configurations, requirements or review decisions."""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services import agent_debate as deb
from services import decision_assistant as da
from services import decision_llm as dl
from services import dynamic_prioritization as dp
from services import management_reports as mr
from services import priority_challenges as pc
from services import prioritization_profiles as pp
from services import requirement_review as rr
from services import scenario_comparison as sc
from services import what_if_simulator as wis
from services.agent_chat import run_agent_options
from services.agent_configuration import list_agents
from services.agent_weights import BP_TOTAL, bp_to_percent, percent_to_bp
from services.evidence_retrieval import get_feedback_by_ids
from ui import data
from ui.theme import ACCENT, BORDER, SURFACE, TEXT, TEXT_2, apply_theme, badge, bar_chart, chip, esc, header, kpi

SECOND = "#D95926"  # categorical slot 2 (dark), paired with ACCENT for A/B comparisons
KIND_BADGE = {"FACT": "DIRECT", "ASSESSMENT": "SUPPORTING", "ASSUMPTION": "CONTEXTUAL", "LIMITATION": "PENDING",
              "SIMULATION": "REVIEW"}

apply_theme()
header("Decision Intelligence",
       "Ask why requirements rank as they do, discuss individual agents, debate two requirements, simulate weights and "
       "export reports. Offline, evidence-based explanations; nothing here changes official results.")

with st.sidebar:
    actor = st.text_input("Your user ID", value=st.session_state.get("di_actor", os.getenv("USER", "")), key="di_actor")
    st.caption("Typed IDs are not authentication - the prototype has no login.")


def act(fn, success=None):
    try:
        with data.write_session() as s:
            out = fn(s)
        if success:
            st.session_state["di_flash"] = success
        return out if out is not None else True
    except (ValueError, PermissionError, RuntimeError) as exc:
        st.error(str(exc))
        return None


if msg := st.session_state.pop("di_flash", None):
    st.success(msg)

with data.write_session() as s:
    requirements = rr.list_requirements(s)
    runs = dp.list_runs(s)
    profiles = pp.list_profiles(s)
active_profile = next((p for p in profiles if p.status == "active"), None)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, len(requirements), "Saved requirements", "from Vehicle Requirements")
kpi(k2, len(runs), "Prioritization runs", "saved, reproducible")
kpi(k3, f"v{active_profile.version}" if active_profile else "-",
    f"Active profile: {active_profile.profile_name}" if active_profile else "Active profile", "none active" if not active_profile else "")
if not requirements or not runs:
    kpi(k4, "-", "Selected run", "none")
    st.info("No saved requirements yet. Create one in Vehicle Requirements, then configure agents and run a prioritization "
            "in Agent Prioritization." if not requirements else
            "No prioritization run yet. Execute agents in Agent Prioritization → Execution, then return here.")
    st.stop()

run_id = st.selectbox("Prioritization run (all answers are bound to this run)", [r["run_id"] for r in runs], key="di_run",
                      format_func=lambda i: next(f"{r['run_id']} · {r['profile_name']} v{r['profile_version']} · "
                                                 f"{r['complete']} complete / {r['incomplete']} incomplete" for r in runs if r["run_id"] == i))
with data.write_session() as s:
    rows = dp.ranking(s, run_id)
    agents = run_agent_options(s, run_id)
run_info = next(r for r in runs if r["run_id"] == run_id)
kpi(k4, f"{run_info['complete']} / {run_info['incomplete']}", "Complete / incomplete", run_id)
req_title = {r.requirement_id: r.title for r in rows}
req_ids = [r.requirement_id for r in rows]

tabs = st.tabs(["A · Decision overview", "B · Ask the assistant", "C · Agent debate", "D · What-if & scenarios",
                "E · Management reports"])


def render_response(resp, key: str) -> None:
    st.markdown(badge("SUPPORTING", resp.label) + chip(f"run {resp.run_id}") + chip(f"basis: {resp.data_basis}")
                + (chip(f"profile v{resp.profile_version}") if resp.profile_version else ""), unsafe_allow_html=True)
    st.markdown(f"**{esc(resp.title)}**")
    for n in resp.notices:
        st.info(n)
    for item in resp.items:
        st.markdown(badge(KIND_BADGE[item.kind], item.kind) + esc(item.text)
                    + (f'<br/><span class="pm-src">refs: {esc(", ".join(item.refs[:12]))}'
                       f'{" …" if len(item.refs) > 12 else ""}</span>' if item.refs else ""), unsafe_allow_html=True)
    if resp.suggestions:
        st.caption("Suggestions: " + " · ".join(resp.suggestions))
    feedback_ids = [r["key"] for r in resp.evidence_refs if r["table"] == "feedback"]
    if resp.evidence_refs:
        with st.expander(f"Evidence references ({len(resp.evidence_refs)}, validated)"):
            st.dataframe(pd.DataFrame(resp.evidence_refs), hide_index=True, width="stretch")
            if feedback_ids and st.toggle("Show original customer comments (displayed locally, not stored)", key=f"vb_{key}"):
                with data.write_session() as s:
                    for r in get_feedback_by_ids(s, feedback_ids[:15]):
                        st.markdown(f"**{r.original_feedback_id}** · {r.country} · {r.engine_type} · "
                                    f'<span class="pm-src">{esc(r.source.source_file)} row {r.source.source_row}</span>',
                                    unsafe_allow_html=True)
                        st.caption(r.customer_feedback)


# ---------------------------------------------------------------------------
# A - overview
# ---------------------------------------------------------------------------
with tabs[0]:
    complete = [r for r in rows if r.priority_status == "COMPLETE"]
    incomplete = [r for r in rows if r.priority_status != "COMPLETE"]
    st.markdown(f"#### Top ranked requirements · run {run_id}")
    if complete:
        top = complete[:10]
        st.plotly_chart(bar_chart([f"#{r.rank} {r.title[:48]}" for r in top], [r.final_score for r in top],
                                  value_name="Final priority score", value_labels=True),
                        width="stretch", config={"displayModeBar": False})
    else:
        st.info("No complete requirements in this run - every requirement is missing at least one agent result.")
    if incomplete:
        st.markdown(f"#### Incomplete (not ranked) · {len(incomplete)}")
        st.dataframe(pd.DataFrame([{"Requirement": r.requirement_id, "Title": r.title, "Missing": ", ".join(r.missing_agents)}
                                   for r in incomplete]), hide_index=True, width="stretch")
    flagged = [(r, f) for r in rows for f in r.critical_flags]
    st.markdown("#### Critical issue indicators")
    if flagged:
        for r, f in flagged:
            st.markdown(badge("REVIEW", f"⚠ {f['category']} · {f['status']}") + f" {esc(r.requirement_id)} · {esc(r.title)}"
                        + f" · rank {r.rank or 'unranked'}", unsafe_allow_html=True)
    else:
        st.caption("No critical-issue flags were recorded for requirements in this run.")

# ---------------------------------------------------------------------------
# B - assistant
# ---------------------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns(2)
    rid = c1.selectbox("Requirement", req_ids, key="di_req", format_func=lambda i: f"{req_title[i]} · {i}")
    agent_names = {a["agent_id"]: a["name"] for a in agents}
    agent_id = c2.selectbox("Agent (optional)", [None] + list(agent_names), key="di_agent",
                            format_func=lambda i: "— none —" if i is None else f"{agent_names[i]} "
                            f"({next(a['scoring_method'] for a in agents if a['agent_id'] == i)})")
    conv_key = f"di_conv_{run_id}_{rid}"

    def ask(**kw):
        out = act(lambda s: da.ask(s, rid, run_id, actor, conversation_id=st.session_state.get(conv_key), **kw))
        if out:
            st.session_state[conv_key] = out[0]

    b = st.columns(4)
    if b[0].button("Explain Ranking", width="stretch"):
        ask(intent="explain_priority")
    if b[1].button("Chat With Agent", width="stretch", disabled=agent_id is None,
                   help="Select an agent first"):
        ask(intent="explain_agent", agent_id=agent_id, question=f"Why did {agent_names.get(agent_id)} assign this score?")
    if b[2].button("Show Evidence", width="stretch"):
        ask(intent="evidence")
    if b[3].button("Challenge Priority", width="stretch"):
        st.session_state["di_show_challenge"] = True
        ask(intent="challenge")

    with st.expander("Suggested questions"):
        q1, q2 = st.columns(2)
        for i, (intent, q) in enumerate(da.SUGGESTED_QUESTIONS.items()):
            col = q1 if i % 2 == 0 else q2
            if intent in ("explain_agent", "compare", "scenario"):
                continue
            if col.button(q, key=f"sq_{intent}", width="stretch"):
                ask(intent=intent)
        other = q1.selectbox("Compare with", [i for i in req_ids if i != rid], key="di_other",
                             format_func=lambda i: f"{req_title[i]} · {i}") if len(req_ids) > 1 else None
        if other and q1.button("Why is it ranked above/below this one?", width="stretch"):
            ask(intent="compare", other_requirement_id=other)
        pct = q2.number_input("What if the selected agent receives (%)", 0.0, 100.0, 40.0, 1.0, key="di_pct")
        if q2.button("Simulate this weight", width="stretch", disabled=agent_id is None):
            ask(intent="scenario", agent_id=agent_id, percent=pct)
    free = st.text_input("Or type a question (interpreted offline; unclear questions get suggestions)", key="di_free")
    if st.button("Ask", disabled=not free.strip()):
        ask(question=free, agent_id=agent_id)

    status = dl.llm_mode_status()
    with st.expander("LLM-assisted wording (optional)" + ("" if status.enabled else " - disabled")):
        if not status.enabled:
            st.caption("Disabled. Internal BMW feedback stays local. " + " ".join(status.reasons))
        else:
            st.caption(f"Provider: {status.provider}. The LLM may only rephrase the validated answer; new numbers or IDs "
                       "are rejected. No fallback to another model.")

    if st.session_state.get(conv_key):
        with data.write_session() as s:
            hist = da.conversation_history(s, st.session_state[conv_key])
        for n in hist["notices"]:
            st.warning(n)
        for i, m in enumerate(reversed(hist["messages"])):
            with st.container(border=True):
                st.markdown(f"**Q:** {esc(m['question'])}")
                render_response(m["response"], f"{st.session_state[conv_key]}_{i}")
                if status.enabled and i == 0:
                    est = dl.estimate_cost(m["response"])
                    ok = st.checkbox(f"Send this structured answer (no raw comments) to {status.provider}; estimated "
                                     f"~${est['estimated_cost_usd']:.4f}", key=f"llm_ok_{i}")
                    if st.button("Rephrase with LLM", disabled=not ok, key=f"llm_btn_{i}"):
                        try:
                            out = dl.rephrase(m["response"], confirmed=ok, cost_acknowledged=est)
                            st.write(out["text"]) if out["text"] else st.error("Rejected: " + "; ".join(out["errors"]))
                        except (PermissionError, RuntimeError) as exc:
                            st.error(str(exc))

    if st.session_state.get("di_show_challenge"):
        st.markdown("#### Challenge this priority")
        st.caption("Recording a challenge never changes a score. Reassessments go through Agent Prioritization.")
        ch_type = st.selectbox("Challenge type", list(pc.TYPES), key="ch_type")
        ch_agent = st.selectbox("Challenged agent (optional)", [None] + list(agent_names), key="ch_agent",
                                format_func=lambda i: "— whole priority —" if i is None else agent_names[i])
        arg = st.text_area("Your argument", key="ch_arg")
        if st.button("Record challenge", disabled=not arg.strip()):
            if act(lambda s: pc.create_challenge(s, rid, run_id, arg, actor, ch_type, ch_agent), "Challenge recorded (OPEN)."):
                st.session_state["di_show_challenge"] = False
                st.rerun()
    with data.write_session() as s:
        challenges = pc.list_challenges(s, requirement_id=rid)
    if challenges:
        st.markdown(f"#### Challenges for this requirement · {len(challenges)}")
        for c in challenges:
            with st.expander(f"{c['challenge_id']} · {c['challenge_type']} · {c['status']} · run {c['run_id']}"):
                st.markdown(f"**Argument** ({esc(c['raised_by'])}): {esc(c['argument'])}")
                for n in c["needs_review"]:
                    st.markdown("- " + esc(n))
                st.dataframe(pd.DataFrame(c["events"]), hide_index=True, width="stretch")
                note = st.text_input("Response note", key=f"chn_{c['challenge_id']}")
                link = st.text_input("Linked reassessment ID (optional, from Agent Prioritization)", key=f"chl_{c['challenge_id']}")
                cols = st.columns(5)
                for col, action in zip(cols, ("start_review", "resolve", "dismiss", "reopen", "comment")):
                    if col.button(action.replace("_", " ").title(), key=f"cha_{action}_{c['challenge_id']}", disabled=not note.strip()):
                        if act(lambda s, a=action: pc.respond(s, c["challenge_id"], a, actor, note, link or None),
                               f"Challenge {action.replace('_', ' ')} recorded."):
                            st.rerun()

# ---------------------------------------------------------------------------
# C - debate
# ---------------------------------------------------------------------------
with tabs[2]:
    if len(req_ids) < 2:
        st.info("A debate needs two requirements in the selected run.")
    else:
        a1, a2 = st.columns(2)
        ra = a1.selectbox("Requirement A", req_ids, key="db_a", format_func=lambda i: f"{req_title[i]} · {i}")
        rb = a2.selectbox("Requirement B", [i for i in req_ids if i != ra], key="db_b", format_func=lambda i: f"{req_title[i]} · {i}")
        if st.button("Start Debate", type="primary"):
            st.session_state["db_result"] = act(lambda s: deb.run_debate(s, run_id, ra, rb))
        d = st.session_state.get("db_result")
        if d and d.run_id == run_id and {d.requirement_a["requirement_id"], d.requirement_b["requirement_id"]} == {ra, rb}:
            for n in d.notices:
                st.warning(n)
            st.markdown("##### Round 1 · Agent positions (from stored scores and rationales)")
            st.dataframe(pd.DataFrame([{"Agent": p.agent_name, "Method": p.scoring_method, "Weight %": p.weight_percent,
                                        "Score A": p.score_a, "Score B": p.score_b, "Contribution A": p.contribution_a,
                                        "Contribution B": p.contribution_b,
                                        "Position": p.position.replace("_", " ").title().replace("Favours A", "Favours A")
                                        .replace("Favours B", "Favours B"),
                                        "Basis A": p.basis_a, "Basis B": p.basis_b} for p in d.round1_positions]),
                         hide_index=True, width="stretch")
            scored = [p for p in d.round1_positions if p.contribution_a is not None and p.contribution_b is not None]
            if scored:
                fig = go.Figure()
                for label, color, vals in (("A", ACCENT, [p.contribution_a for p in scored]),
                                           ("B", SECOND, [p.contribution_b for p in scored])):
                    fig.add_bar(name=f"Requirement {label}", x=vals, y=[p.agent_name for p in scored], orientation="h",
                                marker=dict(color=color, cornerradius=4),
                                hovertemplate="<b>%{y}</b><br>" + label + ": %{x} points<extra></extra>")
                fig.update_layout(barmode="group", height=60 * len(scored) + 80, margin=dict(l=8, r=16, t=8, b=8),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2),
                                  legend=dict(orientation="h", y=1.08), xaxis=dict(gridcolor=BORDER, title="Weighted contribution (points)"),
                                  yaxis=dict(tickfont=dict(color=TEXT)), hoverlabel=dict(bgcolor=SURFACE))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown("##### Round 2 · Disagreements")
            r2 = d.round2_disagreements
            for k in ("FAVOURS_A", "FAVOURS_B", "NO_DIFFERENCE", "UNABLE_TO_CONCLUDE"):
                st.markdown(chip(k.replace("_", " ").title()) + esc(", ".join(r2[k]) or "—"), unsafe_allow_html=True)
            st.markdown("##### Round 3 · What drives the difference")
            st.dataframe(pd.DataFrame(d.round3_drivers), hide_index=True, width="stretch")
            st.markdown("##### Round 4 · Synthesis")
            syn = d.round4_synthesis
            st.markdown(f"- {esc(syn['agreement'])}\n- {esc(syn['ranking_under_profile'])}\n- Sensitivity: "
                        f"{esc(syn['sensitivity']['classification'])}")
            if syn["sensitivity"]["single_agent_shifts_to_tie"]:
                st.caption("SIMULATION · weight-point shifts to a single agent (others scaled proportionally) that would tie A and B: "
                           + "; ".join(f"{x['agent']} {x['weight_change_points']:+}" for x in syn["sensitivity"]["single_agent_shifts_to_tie"]))
            if syn["missing_evidence"]:
                st.warning("Unable to conclude: " + ", ".join(syn["missing_evidence"]))
            st.caption(syn["note"])
            if st.button("Save debate record"):
                saved = act(lambda s: deb.run_debate(s, run_id, ra, rb, actor, save=True).debate_id)
                if saved:
                    st.success(f"Debate saved as {saved} (linked to run {run_id}).")

# ---------------------------------------------------------------------------
# D - what-if + scenario comparison
# ---------------------------------------------------------------------------
with tabs[3]:
    st.caption("SIMULATION ONLY - official runs, profiles and assessments are never changed. Weights must total 100%.")
    with data.write_session() as s:
        base_weights = wis.current_weights(s, run_id)
        addable = [a for a in list_agents(s) if a["status"] == "ACTIVE" and a["agent_id"] not in base_weights]
    if st.button("Reset"):
        for aid, w in base_weights.items():
            st.session_state[f"wi_{run_id}_{aid}"] = float(w)
        st.session_state.pop("wi_result", None)
        st.rerun()
    weights = {}
    for a in agents:
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{esc(a['name'])}** " + chip(a["scoring_method"]) + chip(f"run weight {a['weight_percent']}%"),
                    unsafe_allow_html=True)
        weights[a["agent_id"]] = c2.number_input("weight", 0.0, 100.0, float(a["weight_percent"]), 0.5,
                                                 key=f"wi_{run_id}_{a['agent_id']}", label_visibility="collapsed")
    add = st.multiselect("Add already-configured agents (scored now from current data)", [a["agent_id"] for a in addable],
                         format_func=lambda i: next(a["name"] for a in addable if a["agent_id"] == i), key="wi_add")
    for aid in add:
        name = next(a["name"] for a in addable if a["agent_id"] == aid)
        weights[aid] = st.number_input(f"{name} weight %", 0.0, 100.0, 0.0, 0.5, key=f"wi_add_{run_id}_{aid}")
    active = {k: v for k, v in weights.items() if v > 0}
    total = sum(percent_to_bp(round(v, 2)) for v in active.values())
    (st.success if total == BP_TOTAL else st.warning)(
        f"TOTAL WEIGHT: {bp_to_percent(total)}%" + ("" if total == BP_TOTAL else f" · REMAINING: {bp_to_percent(BP_TOTAL - total)}%")
        + " · agents at 0% are treated as deactivated")
    s1, s2 = st.columns(2)
    if s1.button("Simulate", type="primary", disabled=total != BP_TOTAL):
        st.session_state["wi_result"] = act(lambda s: wis.simulate(s, run_id, {k: round(v, 2) for k, v in active.items()}))
    other_profiles = [p for p in profiles if p.valid and p.status != "archived"]
    if other_profiles:
        prof = s2.selectbox("…or apply another profile's weights", [p.profile_id for p in other_profiles], key="wi_prof",
                            format_func=lambda i: next(p.profile_name for p in other_profiles if p.profile_id == i))
        if s2.button("Simulate with this profile"):
            st.session_state["wi_result"] = act(lambda s: wis.simulate_profile(s, run_id, prof))
    sim = st.session_state.get("wi_result")
    if sim and sim.base_run_id == run_id:
        for n in sim.notes:
            st.caption(n)
        st.dataframe(pd.DataFrame([{"Requirement": r.requirement_id, "Title": r.title, "Official score": r.original_final,
                                    "Official rank": r.original_rank, "Simulated score": r.simulated_final,
                                    "Simulated rank": r.simulated_rank, "Rank change": r.rank_change,
                                    "Simulated status": r.simulated_status, "Missing": ", ".join(r.missing_agents),
                                    "Critical flags": "; ".join(f"{f['category']} ({f['status']})" for f in r.critical_flags)}
                                   for r in sim.rows]), hide_index=True, width="stretch")
        both = [r for r in sim.rows if r.original_final is not None and r.simulated_final is not None]
        if both:
            fig = go.Figure()
            for label, color, vals in (("Official", ACCENT, [r.original_final for r in both]),
                                       ("Simulated", SECOND, [r.simulated_final for r in both])):
                fig.add_bar(name=label, x=vals, y=[r.title[:45] for r in both], orientation="h",
                            marker=dict(color=color, cornerradius=4), hovertemplate="<b>%{y}</b><br>" + label + ": %{x}<extra></extra>")
            fig.update_layout(barmode="group", height=60 * len(both) + 80, margin=dict(l=8, r=16, t=8, b=8),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2),
                              legend=dict(orientation="h", y=1.08), xaxis=dict(gridcolor=BORDER, title="Priority score"),
                              yaxis=dict(tickfont=dict(color=TEXT)), hoverlabel=dict(bgcolor=SURFACE))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        n1, n2 = st.columns([3, 1])
        name = n1.text_input("Scenario name", key="wi_name")
        if n2.button("Save as Scenario", disabled=not name.strip()):
            sid = act(lambda s: wis.save_scenario(s, sim, name, actor).scenario_id)
            if sid:
                st.success(f"Saved scenario {sid}. The official run {run_id} is unchanged.")

    st.markdown("#### Compare scenarios")
    with data.write_session() as s:
        scenarios = wis.list_scenarios(s)
    options = [("run", r["run_id"]) for r in runs] + [("scenario", x["scenario_id"]) for x in scenarios]
    if sim and sim.base_run_id == run_id:
        options.append(("simulation", "current"))
    label = lambda o: {"run": "Run ", "scenario": "Scenario ", "simulation": "Current simulation"}[o[0]] + (o[1] if o[0] != "simulation" else "")
    l1, l2 = st.columns(2)
    left = l1.selectbox("Left", options, format_func=label, key="cmp_l")
    right = l2.selectbox("Right", options, index=min(1, len(options) - 1), format_func=label, key="cmp_r")
    if st.button("Compare", disabled=left == right):
        def side(s, o):
            if o[0] == "run":
                return sc.side_from_run(s, o[1])
            if o[0] == "scenario":
                return sc.side_from_scenario(s, o[1])
            return sc.side_from_simulation(s, sim)
        st.session_state["cmp_result"] = act(lambda s: sc.compare_sides(s, side(s, left), side(s, right)))
    cmp = st.session_state.get("cmp_result")
    if cmp:
        st.markdown(f"**{esc(cmp['left']['label'])}** vs **{esc(cmp['right']['label'])}**")
        (st.info if "attributable to weight" in cmp["attribution"] else st.warning)(cmp["attribution"])
        st.dataframe(pd.DataFrame(cmp["rows"]).drop(columns=["critical_flags"], errors="ignore"), hide_index=True, width="stretch")
        st.json(cmp["agent_configuration_differences"], expanded=False)
        if cmp["input_differences"]:
            with st.expander(f"Input differences disclosed ({len(cmp['input_differences'])})"):
                for x in cmp["input_differences"]:
                    st.markdown("- " + esc(x))
        for l in cmp["limitations"]:
            st.caption(l)

# ---------------------------------------------------------------------------
# E - reports
# ---------------------------------------------------------------------------
with tabs[4]:
    st.caption(f"Report scope: {mr.SCOPE}. Built from run {run_id} and saved records only. {mr.DISCLAIMER}")
    r1, r2 = st.columns(2)
    pstat = r1.multiselect("Priority status", ["COMPLETE", "INCOMPLETE"], default=["COMPLETE", "INCOMPLETE"], key="rp_p")
    rstat = r2.multiselect("Review status (at run time)", sorted({r.review_status for r in rows}), key="rp_r")
    o1, o2, o3 = st.columns(3)
    inc_refs = o1.checkbox("Include evidence references", True, key="rp_refs")
    inc_verb = o1.checkbox("Include verbatim customer feedback (confidential)", False, key="rp_verb")
    inc_rat = o2.checkbox("Include assessment rationales", True, key="rp_rat")
    inc_hist = o2.checkbox("Include decision / approval history", True, key="rp_hist")
    inc_ch = o3.checkbox("Include challenges", False, key="rp_ch")
    inc_db = o3.checkbox("Include saved debates", False, key="rp_db")
    with data.write_session() as s:
        run_scenarios = wis.list_scenarios(s, base_run_id=run_id)
    scn = st.multiselect("Include saved scenarios", [x["scenario_id"] for x in run_scenarios], key="rp_scn",
                         format_func=lambda i: next(f"{x['name']} · {i}" for x in run_scenarios if x["scenario_id"] == i))
    if inc_verb:
        st.warning("Verbatim feedback will be included (up to 5 comments per requirement). Handle the file as confidential.")
    opts = mr.ReportOptions(priority_statuses=pstat or ["COMPLETE", "INCOMPLETE"], review_statuses=rstat or None,
                            include_evidence_refs=inc_refs, include_verbatim_feedback=inc_verb,
                            include_assessment_rationales=inc_rat, include_challenges=inc_ch, include_debates=inc_db,
                            scenario_ids=scn, include_review_history=inc_hist)
    e1, e2, e3 = st.columns(3)
    for col, fmt, label, mime in ((e1, "xlsx", "Export Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                                  (e2, "csv", "Export CSV", "text/csv"), (e3, "pdf", "Export PDF", "application/pdf")):
        if col.button(label, width="stretch", key=f"rp_btn_{fmt}"):
            out = act(lambda s, f=fmt: mr.export_report(s, run_id, f, actor, opts))
            if out:
                st.session_state[f"rp_file_{fmt}"] = out
        if st.session_state.get(f"rp_file_{fmt}"):
            content, name = st.session_state[f"rp_file_{fmt}"]
            col.download_button(f"Download {fmt.upper()}", content, file_name=name, mime=mime, key=f"rp_dl_{fmt}",
                                width="stretch")
