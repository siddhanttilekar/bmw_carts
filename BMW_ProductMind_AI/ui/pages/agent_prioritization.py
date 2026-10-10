"""Page 6 - Agent Prioritization: configurable agents, weights, profiles, runs,
ranking and explanations. Agents are configurable evaluation components, not
autonomous LLM agents. No external LLM calls."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from core.prioritization_models import EXECUTION_METHODS
from services import agent_configuration as ac
from services import critical_issues as ci
from services import decision_context as dc
from services import dynamic_prioritization as dp
from services import prioritization_profiles as pp
from services import requirement_review as rr
from services.agent_resources import RESOURCES, list_resources
from services.agent_scoring import AI_UNAVAILABLE, METHODS, check_compatibility
from services.agent_weights import BP_TOTAL, bp_to_percent, distribute_remaining, percent_to_bp
from ui import data
from ui.theme import apply_theme, badge, bar_chart, chip, esc, header, kpi

apply_theme()
header("Agent Prioritization",
       "Configurable evaluation agents score saved requirements; weights are set per profile and must total 100%. "
       "Agents are configurable evaluation components - not autonomous LLM agents. A priority score never approves a requirement.")

with st.sidebar:
    actor = st.text_input("Your user ID", value=st.session_state.get("prio_actor", os.getenv("USER", "")), key="prio_actor")


def act(fn, success: str | None = None):
    """Run a write action in its own session; show errors instead of crashing."""
    try:
        with data.write_session() as s:
            out = fn(s)
        st.cache_data.clear()
        if success:
            st.session_state["prio_flash"] = success
        return out if out is not None else True
    except (ValueError, PermissionError) as exc:
        st.error(str(exc))
        return None


if msg := st.session_state.pop("prio_flash", None):
    st.success(msg)

with data.write_session() as s:
    agents = ac.list_agents(s)
    profiles = pp.list_profiles(s)
    requirements = rr.list_requirements(s)
    runs = dp.list_runs(s)
    resources = list_resources(s)
res_names = {r.resource_id: r.name for r in resources}
agent_by_id = {a["agent_id"]: a for a in agents}

k1, k2, k3, k4 = st.columns(4)
kpi(k1, sum(a["status"] == "ACTIVE" for a in agents), "Active agents", f"{len(agents)} configured")
kpi(k2, sum(p.status == "active" for p in profiles), "Active profile", f"{len(profiles)} profiles")
kpi(k3, len(requirements), "Saved requirements", "from Vehicle Requirements")
kpi(k4, len(runs), "Prioritization runs", "stored with full snapshots")

tabs = st.tabs(["A · Agent Builder", "B · Weights", "C · Execution", "D · Ranking", "E · Profiles", "F · Explain my score"])

# ---------------------------------------------------------------------------
# A - Agent builder
# ---------------------------------------------------------------------------
with tabs[0]:
    if agents:
        st.dataframe(pd.DataFrame([{
            "Agent Name": a["name"], "Objective": a["objective"], "Execution Method": a["execution_method"],
            "Scoring Method": a["scoring_method"], "Resources": ", ".join(res_names[r] for r in a["resources"]),
            "Default Weight %": str(a["default_weight_percent"]), "Status": a["status"],
            "Version": a["configuration_version"], "Agent ID": a["agent_id"]} for a in agents]),
            hide_index=True, width="stretch")
    else:
        st.info("No agents yet. Create one below, or load the illustrative starter set (configuration only - no scores).")
        if st.button("Load illustrative starter agents and profile"):
            act(lambda s: (ac.seed_starter_agents(s, actor), pp.seed_illustrative_profile(s, actor)),
                "Starter agents and an illustrative profile were created (not official BMW configurations).")
            st.rerun()

    st.markdown("#### Actions")
    if agents:
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
        sel = c1.selectbox("Agent", [a["agent_id"] for a in agents], format_func=lambda i: agent_by_id[i]["name"], key="ab_sel")
        if c2.button("Edit", width="stretch", key="ab_btn_edit"):
            st.session_state["ab_edit"] = sel
        if c3.button("Duplicate", width="stretch", key="ab_btn_dup"):
            act(lambda s: ac.duplicate_agent(s, sel, actor), "Agent duplicated.")
            st.rerun()
        is_active = agent_by_id[sel]["status"] == "ACTIVE"
        if c4.button("Deactivate" if is_active else "Activate", width="stretch", key="ab_btn_toggle"):
            act(lambda s: ac.set_enabled(s, sel, not is_active, actor), "Agent status changed (new version).")
            st.rerun()
        confirm_remove = st.checkbox("Confirm removal (soft delete - history is kept)", key="ab_rm_ok")
        if c5.button("Remove", width="stretch", disabled=not confirm_remove, key="ab_btn_rm"):
            act(lambda s: ac.remove_agent(s, sel, actor), "Agent removed (soft delete; historical results kept).")
            st.rerun()

    editing = st.session_state.get("ab_edit")
    st.markdown(f"#### {'Edit agent' if editing else 'Create agent'}")
    if editing:
        with data.write_session() as s:
            spec = ac.current_spec(s, editing)
        st.caption(f"Editing {agent_by_id.get(editing, {}).get('name', editing)} - saving creates a new configuration "
                   "version; historical results are not rewritten.")
    else:
        spec = ac.AgentSpec(name="", objective="", execution_method="DATA_ANALYSIS",
                            scoring_method="LOG_NORMALIZED_FEEDBACK_COUNT", resources=["RES_G60_FEEDBACK", "RES_REQUIREMENTS"])
    fk = editing or "new"
    name = st.text_input("Agent Name", spec.name, key=f"ab_name_{fk}")
    desc = st.text_input("Description", spec.description, key=f"ab_desc_{fk}")
    objective = st.text_area("Objective", spec.objective, height=70, key=f"ab_obj_{fk}")
    e1, e2 = st.columns(2)
    exe = e1.selectbox("Execution Method", list(EXECUTION_METHODS), index=list(EXECUTION_METHODS).index(spec.execution_method),
                       key=f"ab_exe_{fk}", format_func=lambda m: m.replace("_", " ").title())
    method_ids = [m for m, d in METHODS.items() if d.execution_method == exe]
    method = e2.selectbox("Scoring Method", method_ids, key=f"ab_m_{fk}", format_func=lambda m: METHODS[m].label,
                          index=method_ids.index(spec.scoring_method) if spec.scoring_method in method_ids else 0)
    st.caption(METHODS[method].description)
    res = st.multiselect("Resources", list(RESOURCES), default=[r for r in spec.resources if r in RESOURCES],
                         format_func=res_names.get, key=f"ab_res_{fk}")
    w1, w2 = st.columns(2)
    weight = w1.number_input("Default weight (%)", 0.0, 100.0, float(spec.weight_percent), 0.5, key=f"ab_w_{fk}")
    questions = w2.text_area("Evaluation questions (optional, one per line)", "\n".join(spec.evaluation_questions),
                             key=f"ab_q_{fk}", height=70)
    errors, warnings = check_compatibility(exe, method, res)
    for e in errors:
        st.error(f"Incompatible: {e}")
    for w in warnings:
        st.warning(w)
    if exe == "AI_ANALYSIS":
        st.info(AI_UNAVAILABLE)
    b1, b2 = st.columns([1, 4])
    if b1.button("Save Agent", type="primary", disabled=bool(errors)):
        new = ac.AgentSpec(name=name, description=desc, objective=objective, execution_method=exe, scoring_method=method,
                           resources=res, weight_percent=str(weight), enabled=spec.enabled,
                           evaluation_questions=[q for q in questions.splitlines() if q.strip()])
        if editing:
            ok = act(lambda s: ac.edit_agent(s, editing, new, actor), "Agent saved as a new configuration version.")
        else:
            ok = act(lambda s: ac.create_agent(s, new, actor), "Agent created.")
        if ok:
            st.session_state.pop("ab_edit", None)
            st.rerun()
    if editing and b2.button("Cancel edit"):
        st.session_state.pop("ab_edit", None)
        st.rerun()
    with st.expander("Resource registry"):
        st.dataframe(pd.DataFrame([{"Resource": r.name, "Records": r.available_records, "Source": r.data_source,
                                    "Supported methods": ", ".join(r.supported_analysis_methods),
                                    "Limitations": " ".join(r.limitations)} for r in resources]),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# B - Weights
# ---------------------------------------------------------------------------
editable_profiles = [p for p in profiles if p.status != "archived"]
with tabs[1]:
    if not editable_profiles:
        st.info("Create a profile in section E first.")
    else:
        pid = st.selectbox("Profile", [p.profile_id for p in editable_profiles], key="w_profile",
                           format_func=lambda i: next(f"{p.profile_name} · {p.status} · v{p.version}" for p in profiles if p.profile_id == i))
        prof = next(p for p in profiles if p.profile_id == pid)
        if prof.illustrative:
            st.caption(pp.ILLUSTRATIVE_NOTE)
        current = {a["agent_id"]: a for a in prof.agents}
        addable = [a["agent_id"] for a in agents if a["status"] == "ACTIVE" and a["agent_id"] not in current]
        members = list(current) + st.multiselect("Add active agents", addable, format_func=lambda i: agent_by_id[i]["name"],
                                                 key=f"w_add_{pid}")
        weights = {}
        for aid in members:
            a = agent_by_id.get(aid) or {"name": current[aid]["name"], "status": "REMOVED", "default_weight_percent": 0}
            default = float(current[aid]["weight_percent"]) if aid in current else float(a["default_weight_percent"])
            c1, c2, c3 = st.columns([3, 1.2, 1])
            c1.markdown(f"**{esc(a['name'])}** " + chip(a["status"]) + (chip(current[aid]["scoring_method"]) if aid in current else ""),
                        unsafe_allow_html=True)
            key = f"w_{pid}_{aid}"
            if f"{key}_proposal" in st.session_state:
                st.session_state[key] = st.session_state.pop(f"{key}_proposal")
            weights[aid] = c2.number_input("Weight %", 0.0, 100.0, default, 0.5, key=key, label_visibility="collapsed")
            if c3.checkbox("remove", key=f"w_rm_{pid}_{aid}"):
                weights.pop(aid)
        try:
            bp = {aid: percent_to_bp(round(v, 2)) for aid, v in weights.items()}
        except ValueError as exc:
            st.error(str(exc))
            bp = {}
        total = sum(bp.values())
        status_text = f"TOTAL WEIGHT: {bp_to_percent(total)}%" + ("" if total == BP_TOTAL else f"  ·  REMAINING: {bp_to_percent(BP_TOTAL - total)}%")
        (st.success if total == BP_TOTAL else st.warning)(status_text)
        inactive = [agent_by_id.get(a, {}).get("name", a) for a in bp if agent_by_id.get(a, {}).get("status") != "ACTIVE"]
        if inactive:
            st.warning("Inactive or removed agents cannot contribute: " + ", ".join(inactive) + ". Remove them from the profile.")
        d1, d2 = st.columns(2)
        if total < BP_TOTAL and bp and d1.button("Distribute remaining weight (proposal)"):
            proposal = distribute_remaining(bp, [a for a in bp if agent_by_id.get(a, {}).get("status") == "ACTIVE"])
            for aid, v in proposal.items():
                st.session_state[f"w_{pid}_{aid}_proposal"] = float(bp_to_percent(v))
            st.session_state["prio_flash"] = ("Remaining weight was distributed evenly across active agents in the form. "
                                              "Review the values, then save.")
            st.rerun()
        if d2.button("Save weights (new profile version)", disabled=not bp):
            act(lambda s: pp.edit_profile(s, pid, actor, weights={a: bp_to_percent(v) for a, v in bp.items()}),
                "Saved as a new profile version." + ("" if total == BP_TOTAL else " The profile is a draft until weights total 100%."))
            st.rerun()
        st.caption("Execution and activation stay disabled until active weights total exactly 100%.")

# ---------------------------------------------------------------------------
# C - Execution
# ---------------------------------------------------------------------------
valid_profiles = [p for p in profiles if p.valid and p.status != "archived"]
with tabs[2]:
    if not requirements:
        st.info("No saved requirements yet. Create a requirement in the Vehicle Requirements page.")
    elif not valid_profiles:
        st.info("No executable profile. A profile needs active agents whose weights total 100% (sections B and E).")
    else:
        req_ids = [r["requirement_id"] for r in requirements]
        pre = st.session_state.pop("prio_requirement", None)
        if pre in req_ids:
            st.session_state["x_req"] = pre
        rid = st.selectbox("Saved requirement", req_ids, key="x_req",
                           format_func=lambda i: next(f"{r['title']} · {r['status']} · {i}" for r in requirements if r["requirement_id"] == i))
        active_pid = next((p.profile_id for p in valid_profiles if p.status == "active"), valid_profiles[0].profile_id)
        xpid = st.selectbox("Prioritization profile", [p.profile_id for p in valid_profiles], key="x_profile",
                            index=[p.profile_id for p in valid_profiles].index(active_pid),
                            format_func=lambda i: next(f"{p.profile_name} · v{p.version} · {p.status}" for p in valid_profiles if p.profile_id == i))
        xprof = next(p for p in valid_profiles if p.profile_id == xpid)
        if xprof.newer_agent_versions:
            st.info("These agents changed since this profile version (re-save the profile to adopt): "
                    + ", ".join(xprof.newer_agent_versions))

        human_agents = [a for a in xprof.agents if a["scoring_method"] == "HUMAN_SCORE"]
        if human_agents:
            st.markdown("#### Human assessments")
            st.caption("Scores are never filled in automatically. Each needs a score, a rationale and the assessor ID.")
            with data.write_session() as s:
                ev_keys = [e["source_key"] for e in rr.requirement_as_dict(s, rr.get_requirement(s, rid))["evidence"]]
            for a in human_agents:
                with st.expander(f"{a['name']} - assessment"):
                    sc = st.number_input("Score (0-100)", 0, 100, None, 1, key=f"hs_{rid}_{a['agent_id']}")
                    why = st.text_area("Rationale", key=f"hr_{rid}_{a['agent_id']}")
                    refs = st.multiselect("Evidence references (optional)", ev_keys, key=f"he_{rid}_{a['agent_id']}")
                    if st.button("Save assessment", key=f"hb_{rid}_{a['agent_id']}", disabled=sc is None):
                        act(lambda s: dp.record_human_assessment(s, rid, a["agent_id"], sc, why, actor, refs),
                            f"Assessment saved for {a['name']}.")
                        st.rerun()

        c1, c2 = st.columns(2)
        if c1.button("Execute agents for this requirement", type="primary"):
            run = act(lambda s: dp.run_prioritization(s, xpid, actor, [rid]).run_id, "Run stored.")
            if run:
                st.session_state["x_last_run"] = run
                st.rerun()
        if c2.button(f"Execute agents for all {len(requirements)} saved requirements"):
            run = act(lambda s: dp.run_prioritization(s, xpid, actor).run_id, "Run stored.")
            if run:
                st.session_state["x_last_run"] = run
                st.rerun()
        last = st.session_state.get("x_last_run")
        if last:
            with data.write_session() as s:
                results = dp.agent_results(s, last, rid)
                row = next((r for r in dp.ranking(s, last) if r.requirement_id == rid), None)
            if row:
                st.markdown(f"#### Result · run {last}")
                st.markdown(badge("SUPPORTING" if row.priority_status == "COMPLETE" else "PENDING", row.priority_status)
                            + chip(f"Final score: {row.final_score if row.final_score is not None else 'NULL (incomplete)'}")
                            + chip(f"Rank: {row.rank or '-'}"), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"Agent": r["agent_name"], "Status": r["status"], "Score": r["score"],
                                            "Weight %": r["weight_percent"], "Weighted contribution": r["weighted_contribution"],
                                            "Evidence refs": len(r["source_refs"]), "Explanation": r["rationale"]}
                                           for r in results]), hide_index=True, width="stretch")
                if row.missing_agents:
                    st.warning("INCOMPLETE - missing: " + ", ".join(row.missing_agents) + ". Missing weight is not redistributed.")

# ---------------------------------------------------------------------------
# D - Ranking
# ---------------------------------------------------------------------------
with tabs[3]:
    if not requirements:
        st.info("No saved requirements yet. Create a requirement in the Vehicle Requirements page.")
    elif not runs:
        st.info("No prioritization runs yet. Execute agents in section C.")
    else:
        run_id = st.selectbox("Run", [r["run_id"] for r in runs], key="rk_run",
                              format_func=lambda i: next(f"{r['run_id']} · {r['profile_name']} v{r['profile_version']} · "
                                                         f"{r['complete']} complete / {r['incomplete']} incomplete" for r in runs if r["run_id"] == i))
        with data.write_session() as s:
            rows = dp.ranking(s, run_id)
        complete = [r for r in rows if r.priority_status == "COMPLETE"]
        incomplete = [r for r in rows if r.priority_status != "COMPLETE"]
        f1, f2 = st.columns(2)
        statuses = f1.multiselect("Review status", sorted({r.review_status for r in rows}), key="rk_status")
        flagged_only = f2.toggle("Only requirements with critical flags", key="rk_flag")

        def keep(r):
            return (not statuses or r.review_status in statuses) and (not flagged_only or r.critical_flags)

        def flag_text(r):
            return "; ".join(f"⚠ {f['category']} ({f['status']})" for f in r.critical_flags) or "-"

        st.markdown(f"#### Ranked (complete) · {len(complete)}")
        shown = [r for r in complete if keep(r)]
        if shown:
            st.dataframe(pd.DataFrame([{"Rank": r.rank, "Requirement ID": r.requirement_id, "Requirement Title": r.title,
                                        "Review Status": r.review_status, "Priority Status": r.priority_status,
                                        "Final Score": r.final_score, "Critical Issue Flag": flag_text(r)} for r in shown]),
                         hide_index=True, width="stretch")
            st.plotly_chart(bar_chart([f"#{r.rank} {r.title[:50]}" for r in shown], [r.final_score for r in shown],
                                      value_name="Final priority score", value_labels=True),
                            width="stretch", config={"displayModeBar": False})
        else:
            st.caption("No complete requirements match.")
        st.markdown(f"#### Incomplete (not ranked) · {len(incomplete)}")
        if [r for r in incomplete if keep(r)]:
            st.dataframe(pd.DataFrame([{"Requirement ID": r.requirement_id, "Requirement Title": r.title,
                                        "Review Status": r.review_status, "Missing agents": ", ".join(r.missing_agents),
                                        "Critical Issue Flag": flag_text(r)} for r in incomplete if keep(r)]),
                         hide_index=True, width="stretch")
        flagged = [r for r in rows if any(f["status"] == "CONFIRMED" for f in r.critical_flags)]
        if flagged:
            st.error("Confirmed critical issues (shown regardless of rank): "
                     + "; ".join(f"{r.requirement_id} {r.title}" for r in flagged))
        st.caption("A priority score does not mean a requirement is approved; approval stays in the review workflow.")

# ---------------------------------------------------------------------------
# E - Profiles
# ---------------------------------------------------------------------------
with tabs[4]:
    if profiles:
        st.dataframe(pd.DataFrame([{"Profile": p.profile_name, "Status": p.status, "Version": p.version,
                                    "Agents": ", ".join(f"{a['name']} {a['weight_percent']}%" for a in p.agents),
                                    "Total %": p.total_percent, "Valid": p.valid, "Illustrative": p.illustrative,
                                    "Profile ID": p.profile_id} for p in profiles]), hide_index=True, width="stretch")
        sel = st.selectbox("Profile", [p.profile_id for p in profiles], key="pf_sel",
                           format_func=lambda i: next(p.profile_name for p in profiles if p.profile_id == i))
        prof = next(p for p in profiles if p.profile_id == sel)
        if prof.errors:
            st.warning("Not executable: " + " ".join(prof.errors))
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Activate", width="stretch", disabled=not prof.valid or prof.status in ("active", "archived"), key="pf_btn_act"):
            act(lambda s: pp.activate_profile(s, sel, actor), "Profile activated.")
            st.rerun()
        if c2.button("Duplicate", width="stretch", key="pf_btn_dup"):
            act(lambda s: pp.duplicate_profile(s, sel, actor), "Profile duplicated.")
            st.rerun()
        if c3.button("Archive", width="stretch", disabled=prof.status == "archived", key="pf_btn_arch"):
            act(lambda s: pp.archive_profile(s, sel, actor), "Profile archived.")
            st.rerun()
        with c4.popover("Edit name / description", width="stretch"):
            nn = st.text_input("Name", prof.profile_name, key=f"pf_n_{sel}")
            nd = st.text_area("Description", prof.description, key=f"pf_d_{sel}")
            if st.button("Save", key=f"pf_s_{sel}"):
                act(lambda s: pp.edit_profile(s, sel, actor, name=nn, description=nd), "Profile saved (new version).")
                st.rerun()
        st.dataframe(pd.DataFrame(prof.agents), hide_index=True, width="stretch")
    st.markdown("#### Create profile")
    active_agents = [a for a in agents if a["status"] == "ACTIVE"]
    if not active_agents:
        st.info("Create or activate agents first.")
    else:
        pn = st.text_input("Profile name", key="pf_new_name")
        pdsc = st.text_input("Description", key="pf_new_desc")
        chosen = st.multiselect("Agents", [a["agent_id"] for a in active_agents], format_func=lambda i: agent_by_id[i]["name"],
                                key="pf_new_agents")
        new_w = {aid: st.number_input(f"{agent_by_id[aid]['name']} weight %", 0.0, 100.0,
                                      float(agent_by_id[aid]["default_weight_percent"]), 0.5, key=f"pf_new_w_{aid}")
                 for aid in chosen}
        tot = sum(percent_to_bp(round(v, 2)) for v in new_w.values())
        (st.success if tot == BP_TOTAL else st.warning)(
            f"TOTAL WEIGHT: {bp_to_percent(tot)}%" + ("" if tot == BP_TOTAL else f" · REMAINING: {bp_to_percent(BP_TOTAL - tot)}%"))
        if st.button("Create profile", disabled=not (pn.strip() and chosen)):
            act(lambda s: pp.create_profile(s, pn, {a: round(v, 2) for a, v in new_w.items()}, actor, pdsc),
                "Profile created as draft." + (" Activate it in the list above." if tot == BP_TOTAL else " Weights must total 100% before activation."))
            st.rerun()

# ---------------------------------------------------------------------------
# F - Explain my score
# ---------------------------------------------------------------------------
with tabs[5]:
    if not requirements:
        st.info("No saved requirements yet. Create a requirement in the Vehicle Requirements page.")
    else:
        ids = [r["requirement_id"] for r in requirements]
        if st.session_state.get("x_req") in ids and "ex_req_f" not in st.session_state:
            st.session_state["ex_req_f"] = st.session_state["x_req"]
        rid = st.selectbox("Requirement", ids, key="ex_req_f",
                           format_func=lambda i: next(f"{r['title']} · {i}" for r in requirements if r["requirement_id"] == i))
        with data.write_session() as s:
            ctx = dc.get_decision_context(s, rid)
        if ctx["run_id"] is None:
            st.info("This requirement has not been prioritized yet (section C).")
        else:
            c1, c2, c3 = st.columns(3)
            kpi(c1, ctx["final_score"] if ctx["final_score"] is not None else "NULL", "Final score",
                ctx["priority_status"] + " · " + ctx["run_id"])
            kpi(c2, ctx["rank"] or "-", "Rank", "only complete requirements are ranked")
            kpi(c3, ctx["review_status"], "Review status", "priority does not imply approval")
            scored = [c for c in ctx["agent_contributions"] if c["weighted_contribution"] is not None]
            unscored = [c for c in ctx["agent_contributions"] if c["weighted_contribution"] is None]
            if scored:
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("##### Weighted contribution")
                    st.plotly_chart(bar_chart([c["agent_name"] for c in scored], [c["weighted_contribution"] for c in scored],
                                              value_name="Points", value_labels=True,
                                              extra=[f"<br>Score {c['score']} × weight {c['weight_percent']}%" for c in scored]),
                                    width="stretch", config={"displayModeBar": False})
                with g2:
                    st.markdown("##### Agent scores (0-100)")
                    st.plotly_chart(bar_chart([c["agent_name"] for c in scored], [c["score"] for c in scored],
                                              value_name="Score", value_labels=True),
                                    width="stretch", config={"displayModeBar": False})
            if unscored:
                st.warning("UNSCORED (no score shown, weight not redistributed): "
                           + "; ".join(f"{c['agent_name']}: {c['rationale']}" for c in unscored))
            st.markdown("##### Agent explanations")
            for c in ctx["agent_contributions"]:
                ex = c["explanation"]
                with st.expander(f"{c['agent_name']} · {c['status']} · score {c['score']} · weight {c['weight_percent']}%"):
                    st.caption(f"{ex['methodology']['label']} v{ex['methodology']['version']} · "
                               f"{ex['methodology']['description']}")
                    for item in ex["items"]:
                        st.markdown(chip(item["kind"]) + esc(item["text"]), unsafe_allow_html=True)
                    st.json(ex["methodology"]["inputs"], expanded=False)
            if ctx["human_assessments"]:
                st.markdown("##### Human assessments")
                st.dataframe(pd.DataFrame(ctx["human_assessments"]), hide_index=True, width="stretch")
            with st.expander(f"Source evidence ({len(ctx['evidence_references'])} references)"):
                st.dataframe(pd.DataFrame(ctx["evidence_references"]), hide_index=True, width="stretch")
            if ctx["missing_assessments"]:
                st.warning("Missing information: " + ", ".join(ctx["missing_assessments"]))
            with st.expander("Assumptions and limitations"):
                for a in ctx["assumptions"]:
                    st.markdown(chip("ASSUMPTION") + esc(a), unsafe_allow_html=True)
                for l in ctx["limitations"]:
                    st.markdown(chip("LIMITATION") + esc(l), unsafe_allow_html=True)

        st.markdown("##### Critical issue flags")
        st.caption("Raised and decided by people only; never inferred from broad VFC labels; never changes the score.")
        for f in ctx["critical_flags"]:
            st.markdown(badge("REVIEW", f"⚠ {f['category']} · {f['status']}") + esc(f["rationale"]), unsafe_allow_html=True)
            if f["status"] == "PROPOSED":
                note = st.text_input("Decision note", key=f"fl_n_{f['flag_id']}")
                d1, d2 = st.columns(2)
                if d1.button("Confirm", key=f"fl_c_{f['flag_id']}", disabled=not note):
                    act(lambda s: ci.decide_flag(s, f["flag_id"], "CONFIRMED", actor, note), "Flag confirmed.")
                    st.rerun()
                if d2.button("Dismiss", key=f"fl_d_{f['flag_id']}", disabled=not note):
                    act(lambda s: ci.decide_flag(s, f["flag_id"], "DISMISSED", actor, note), "Flag dismissed.")
                    st.rerun()
        with st.expander("Raise a critical issue"):
            cat = st.selectbox("Category", list(ci.CATEGORIES), key="fl_cat")
            why = st.text_area("Rationale (required)", key="fl_why")
            with data.write_session() as s:
                keys = [e["source_key"] for e in rr.requirement_as_dict(s, rr.get_requirement(s, rid))["evidence"]]
            refs = st.multiselect("Evidence references", keys, key="fl_refs")
            if st.button("Raise flag", disabled=not why.strip()):
                act(lambda s: ci.raise_flag(s, rid, cat, why, actor, refs), "Flag raised (PROPOSED).")
                st.rerun()
