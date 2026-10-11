"""Product Prioritization - one continuous four-step workflow.

STEP 1 Vehicle Area -> STEP 2 Requirements -> STEP 3 AI Agents -> STEP 4 Decision Results

Selections live in plain session-state keys (wiz_*) so they survive BACK /
NEXT; widgets use separate keys. All logic is in services.product_workflow,
services.agent_analysis and services.argue_chat; scores come from the
existing Python engine.
"""
from __future__ import annotations

import os

import streamlit as st

from services import argue_chat
from services import dynamic_prioritization as dp
from services import llm_budget as budget
from services import product_workflow as pw
from services.agent_weights import BP_TOTAL, bp_to_percent
from services.requirement_generator import NotGeneratable, RequirementDraft
from ui import data
from ui.evidence_ui import llm_settings, paid_call_blocker, paid_client
from ui.theme import apply_theme, esc, progress

STEPS = ["Vehicle Area", "Requirements", "AI Agents", "Decision Results"]
AI_UNAVAILABLE = "AI unavailable — configure OpenAI API in application settings."
EXEC_LABEL = {"AI_ANALYSIS": "AI Analysis", "DATA_ANALYSIS": "Data Analysis", "HUMAN_ASSESSMENT": "Human Assessment"}

apply_theme()
ss = st.session_state
for key, default in (("wiz_step", 1), ("wiz_domains", []), ("wiz_features", []), ("wiz_reqs", []),
                     ("wiz_drafts", {}), ("wiz_agents", []), ("wiz_weights", {}), ("wiz_run_id", None),
                     ("wiz_chats", {}), ("wiz_conv", {})):
    ss.setdefault(key, default)

head, who = st.columns([3, 1])
head.markdown("## Product Prioritization")
head.caption("BMW 5 Series G60 · evidence-first prioritization with configurable agents")
actor = who.text_input("Your user ID", value=ss.get("workflow_actor", os.getenv("USER", "")), key="wiz_actor")
ss["workflow_actor"] = actor
progress(STEPS, ss.wiz_step)
settings = llm_settings()


def go(step: int) -> None:
    ss.wiz_step = step
    st.rerun()


def nav(back: int | None, next_: int | None, next_ok: bool = True, next_help: str | None = None) -> None:
    st.write("")
    left, _, right = st.columns([1, 4, 1])
    if back is not None and left.button("BACK", key=f"back_{ss.wiz_step}", use_container_width=True):
        go(back)
    if next_ is not None and right.button("NEXT", type="primary", key=f"next_{ss.wiz_step}", disabled=not next_ok,
                                          help=next_help, use_container_width=True):
        go(next_)


# ---------------------------------------------------------------------------
# STEP 1 - vehicle area
# ---------------------------------------------------------------------------
def step_area() -> None:
    st.markdown("### Which part of the vehicle do you want to prioritize?")
    st.caption("Vehicle: BMW 5 Series G60. Areas come from the normalised customer-need taxonomy; features are the "
               "original BMW feedback topics mapped to each area.")
    with data.write_session() as s:
        doms = [d for d in pw.domains(s, settings) if d["code"] != "OTHER"]
    names = {d["code"]: d["name"] for d in doms}
    all_on = st.checkbox("All vehicle areas", value=len(ss.wiz_domains) == len(doms), key="w_all")
    chosen = [d["code"] for d in doms] if all_on else st.multiselect(
        "Vehicle areas", list(names), default=[d for d in ss.wiz_domains if d in names], format_func=names.get,
        key="w_domains", placeholder="Choose one or more areas")
    if chosen != ss.wiz_domains:
        ss.wiz_features = []
    ss.wiz_domains = chosen
    if not chosen:
        st.info("Select at least one vehicle area.")
        nav(None, 2, False)
        return
    with data.write_session() as s:
        feats = pw.features(s, chosen)
    fnames = {f["feature_id"]: f"{f['label']} · {f['unique_comments']} comments" + (f"  ({f['domain']})"
                                                                                       if len(chosen) > 1 else "")
              for f in feats}
    picked = st.multiselect("Vehicle features (optional - leave empty to include the whole area)", list(fnames),
                            default=[f for f in ss.wiz_features if f in fnames], format_func=fnames.get,
                            key="w_features")
    ss.wiz_features = picked
    with data.write_session() as s:
        prev = pw.evidence_preview(s, settings, chosen, picked)
    c = st.columns(3)
    c[0].metric("Relevant customer comments", f"{prev['relevant_comments']:,}")
    c[1].metric("Comments with a complaint", f"{prev['complaint_comments']:,}")
    c[2].metric("Comments with a request", f"{prev['request_comments']:,}")
    st.caption("Unique comments (deduplicated by original feedback ID).")
    nav(None, 2, True)


# ---------------------------------------------------------------------------
# STEP 2 - requirements
# ---------------------------------------------------------------------------
def step_requirements() -> None:
    st.markdown("### Which requirements should be prioritized?")
    with data.write_session() as s:
        saved = pw.requirements_in_scope(s, ss.wiz_domains, ss.wiz_features)
        candidates = pw.candidate_needs(s, settings, ss.wiz_domains, ss.wiz_features)
    selected = [r for r in ss.wiz_reqs]
    if saved:
        st.markdown("#### Saved requirements")
    for r in saved:
        with st.container(border=True):
            top, pick = st.columns([6, 1])
            top.markdown(f"**{esc(r['title'])}**  \n<span class='pm-note'>{esc(r['feature'])} · status "
                         f"{esc(r['status'])}</span>", unsafe_allow_html=True)
            on = pick.checkbox("Select", value=r["requirement_id"] in selected, key=f"w_req_{r['requirement_id']}")
            if on and r["requirement_id"] not in selected:
                selected.append(r["requirement_id"])
            if not on and r["requirement_id"] in selected:
                selected.remove(r["requirement_id"])
            st.caption("Customer problem: " + (r["customer_problem"] or "")[:300])
            with st.expander("Evidence summary"):
                st.write(r["evidence_summary"])
    ss.wiz_reqs = selected
    if candidates:
        st.markdown("#### Customer needs without a requirement yet")
        st.caption("Taken from the existing feedback topics and validated AI observations. Generate a draft, review it, "
                   "and save it as a DRAFT requirement to include it. Drafts are never approved automatically.")
    for c in candidates:
        cid = c["cluster_id"]
        with st.container(border=True):
            st.markdown(f"**{esc(c['label'])}**  \n<span class='pm-note'>{esc(c['domain'])} · {c['unique_comments']} "
                        f"comments · {c['complaint_comments']} complaints · {c['request_comments']} requests</span>",
                        unsafe_allow_html=True)
            if c["top_needs"]:
                st.caption("Most common needs: " + "; ".join(c["top_needs"]))
            draft_json = ss.wiz_drafts.get(cid)
            if draft_json is None:
                if st.button("GENERATE REQUIREMENT DRAFT", key=f"w_gen_{cid}"):
                    try:
                        with data.write_session() as s:
                            ss.wiz_drafts[cid] = pw.generate_draft(s, cid).model_dump_json()
                        st.rerun()
                    except NotGeneratable as exc:
                        st.warning(str(exc))
            else:
                draft = RequirementDraft.model_validate_json(draft_json)
                st.markdown(f"*Draft:* **{esc(draft.title)}**")
                st.caption(draft.proposed_requirement[:400])
                b1, b2 = st.columns([1, 1])
                if b1.button("Save draft and select", key=f"w_save_{cid}", disabled=not actor.strip(), type="primary"):
                    with data.write_session() as s:
                        rid = pw.save_draft(s, draft, actor)
                    ss.wiz_drafts.pop(cid, None)
                    if rid not in ss.wiz_reqs:
                        ss.wiz_reqs.append(rid)
                    st.rerun()
                if b2.button("Discard draft", key=f"w_drop_{cid}"):
                    ss.wiz_drafts.pop(cid, None)
                    st.rerun()
    if not saved and not candidates:
        st.info("No requirements or customer needs were found for this selection. Go back and widen it.")
    st.caption(f"{len(ss.wiz_reqs)} requirement(s) selected.")
    nav(1, 3, bool(ss.wiz_reqs), None if ss.wiz_reqs else "Select at least one requirement.")


# ---------------------------------------------------------------------------
# STEP 3 - agents, weights, run
# ---------------------------------------------------------------------------
def step_agents() -> None:
    st.markdown("### Which agents should evaluate the requirements?")
    with data.write_session() as s:
        agents = pw.available_agents(s)
    selected = [a for a in ss.wiz_agents if any(x["agent_id"] == a for x in agents)]
    for row in range(0, len(agents), 3):
        cols = st.columns(3)
        for col, a in zip(cols, agents[row:row + 3]):
            with col.container(border=True):
                on = st.checkbox(f"**{a['name']}**", value=a["agent_id"] in selected, key=f"w_ag_{a['agent_id']}")
                st.caption(a["objective"][:180])
                st.caption(f"{EXEC_LABEL.get(a['execution_method'], a['execution_method'])} · "
                           + (", ".join(a["resource_names"]) or "no resources"))
                if on and a["agent_id"] not in selected:
                    selected.append(a["agent_id"])
                if not on and a["agent_id"] in selected:
                    selected.remove(a["agent_id"])
    ss.wiz_agents = selected
    with st.expander("+ CREATE AGENT"):
        from ui import agent_builder

        new_id = agent_builder.render("wab", actor)
        if new_id:
            ss.wiz_agents.append(new_id)
            st.rerun()
    if not selected:
        st.info("Select at least one agent.")
        nav(2, None)
        return

    st.markdown("#### Agent weights")
    by_id = {a["agent_id"]: a for a in agents}
    weights = {}
    for aid in selected:
        a = by_id[aid]
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"{esc(a['name'])}  \n<span class='pm-note'>{EXEC_LABEL.get(a['execution_method'])}</span>",
                    unsafe_allow_html=True)
        default = float(ss.wiz_weights.get(aid, a["default_weight_percent"]))
        weights[aid] = c2.number_input("Weight %", 0.0, 100.0, default, 0.5, key=f"w_wt_{aid}",
                                       label_visibility="collapsed")
    ss.wiz_weights.update(weights)
    try:
        total_bp = pw.weight_total_bp(weights)
    except ValueError:
        total_bp = -1
    total_ok = total_bp == BP_TOTAL
    (st.success if total_ok else st.warning)(
        f"TOTAL: {bp_to_percent(max(total_bp, 0))}%" + ("" if total_ok else " - Please adjust the selected agent "
                                                                            "weights to 100%."))

    with data.write_session() as s:
        ready = pw.check_readiness(s, settings, ss.wiz_reqs, weights, actor or "system")
    for e in ready.errors:
        if "100%" not in e:
            st.error(e)
    if ready.missing_human:
        st.markdown("#### Human assessments needed")
        for m in ready.missing_human:
            key = f"{m['agent_id']}_{m['requirement_id']}"
            with st.container(border=True):
                st.markdown(f"**{esc(m['agent_name'])}** on *{esc(m['title'])}*")
                sc, rt = st.columns([1, 4])
                score = sc.number_input("Score (0-100)", 0, 100, None, key=f"w_hs_{key}")
                why_ = rt.text_input("Rationale (required)", key=f"w_hr_{key}")
                if st.button("Save assessment", key=f"w_hb_{key}", disabled=score is None or not why_.strip()
                             or not actor.strip()):
                    try:
                        with data.write_session() as s:
                            dp.record_human_assessment(s, m["requirement_id"], m["agent_id"], score, why_, actor)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    approve = None
    blocker = paid_call_blocker()
    if ready.ai_plans:
        st.markdown("#### AI analysis")
        if ready.needs_paid_ai:
            if blocker:
                st.warning(AI_UNAVAILABLE)
            st.write(f"{ready.ai_cached} saved analyses will be reused at no cost. New analyses: estimated "
                     f"${ready.estimated_usd:.4f} (up to ${ready.upper_usd:.4f}).")
            with data.write_session() as s:
                remaining = budget.budget_status(s, settings).remaining_usd
            st.caption(f"Application budget remaining: ${remaining:.2f} (estimate from logged token usage).")
            approve = st.number_input("I approve spending up to (USD)", 0.0, 5.0, 0.0, 0.01, key="w_approve")
        else:
            st.caption(f"All {ready.ai_cached} AI analyses are saved and will be reused at no cost.")
    paid_ok = not ready.needs_paid_ai or (not blocker and approve and approve >= ready.estimated_usd)
    can_run = ready.ok and paid_ok and bool(actor.strip())
    st.write("")
    left, _, right = st.columns([1, 3, 2])
    if left.button("BACK", key="back_3", use_container_width=True):
        go(2)
    if right.button("RUN PRIORITIZATION", type="primary", disabled=not can_run, use_container_width=True):
        try:
            llm = paid_client() if ready.needs_paid_ai else None
            with st.spinner("Running agents and calculating scores"), data.write_session() as s:
                run_id = pw.execute(s, settings, ss.wiz_reqs, weights, actor, llm=llm, approved_max_usd=approve)
            ss.wiz_run_id = run_id
            go(4)
        except (ValueError, RuntimeError, budget.ApprovalRequired, budget.BudgetExceeded) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - never show request content or keys
            st.error(f"The run stopped ({type(exc).__name__}). Saved analyses are kept; nothing was ranked.")
    if not can_run:
        reasons = [e for e in ready.errors] + (["Save the human assessments above."] if ready.missing_human else [])
        if ready.needs_paid_ai and not paid_ok:
            reasons.append(AI_UNAVAILABLE if blocker else "Approve the estimated AI cost above.")
        if not actor.strip():
            reasons.append("Enter your user ID at the top.")
        st.caption("To run: " + " ".join(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# STEP 4 - results, WHY, ARGUE
# ---------------------------------------------------------------------------
def why_panel(run_id: str, rid: str) -> None:
    with data.write_session() as s:
        w = pw.why(s, run_id, rid)
    st.markdown("##### Why this matters")
    for line in w["why_it_matters"]:
        st.write(line)
    if w["top_needs"]:
        st.write("Customers mostly ask for: " + "; ".join(w["top_needs"]) + ".")
    st.markdown("##### Customer evidence")
    if w.get("evidence_basis"):
        st.caption(f"From {w['evidence_basis']}.")
    for q in w["quotes"][:3]:
        st.markdown(f"<div class='pm-quote'>“{esc(q['excerpt'])}”<small>{esc(q['type'].replace('_', ' ').title())} · "
                    f"comment {esc(q['feedback_id'])}</small></div>", unsafe_allow_html=True)
    if not w["quotes"]:
        st.caption("No validated customer excerpts are attached to this result.")
    with st.expander("VIEW SUPPORTING EVIDENCE"):
        for q in w["quotes"]:
            st.markdown(f"<div class='pm-quote'>“{esc(q['excerpt'])}”<small>{esc(q['customer_need'])} · comment "
                        f"{esc(q['feedback_id'])}</small></div>", unsafe_allow_html=True)
        if w["studies"]:
            st.caption("Related customer studies: " + ", ".join(w["studies"]))
        if w["competitor_refs"]:
            st.caption(f"{len(w['competitor_refs'])} competitor observations are linked as context.")
    st.markdown("##### Agent perspectives")
    for p in w["perspectives"]:
        st.markdown(f"- **{esc(p['agent'])}** ({p['weight_percent']:g}%): {esc(p['view'])} - {esc(p['basis'])}")
    st.markdown("##### What is still uncertain")
    for u in w["uncertain"][:6]:
        st.markdown(f"- {esc(u)}")
    with st.expander("ADVANCED SCORE BREAKDOWN"):
        for b in w["breakdown"]:
            st.markdown(f"**{esc(b['agent'])}**")
            for line in b["lines"]:
                st.caption(line)


def argue_panel(run_id: str, rid: str, allow_paid: bool, cap: float, blocker: str | None) -> None:
    key = f"{run_id}|{rid}"
    history = ss.wiz_chats.setdefault(key, [])
    for msg in history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])
            for n in msg.get("notes", []):
                st.caption(n)
            if msg.get("evidence_ids"):
                st.caption("Evidence cited: comments " + ", ".join(msg["evidence_ids"]))
    if blocker:
        st.warning(AI_UNAVAILABLE)
        return
    with st.form(key=f"argue_form_{key}", clear_on_submit=True):
        q = st.text_area("Challenge or question", key=f"argue_q_{key}", height=80,
                         placeholder="e.g. I disagree with this priority - what evidence contradicts it?")
        sent = st.form_submit_button("Send", disabled=not allow_paid)
    if not allow_paid:
        st.caption("Tick 'Allow paid AI replies' above to send questions.")
    if sent and q.strip():
        try:
            llm = paid_client()
            with st.spinner("Thinking"), data.write_session() as s:
                out = argue_chat.ask(s, llm, settings, run_id, rid, history, q, actor, approved_max_usd=cap,
                                     conversation_id=ss.wiz_conv.get(key))
            ss.wiz_conv[key] = out["conversation_id"]
            history += [{"role": "user", "content": q.strip()},
                        {"role": "assistant", "content": out["answer"], "notes": out["notes"],
                         "evidence_ids": out["evidence_ids"]}]
            st.rerun()
        except (argue_chat.ChatError, budget.ApprovalRequired, budget.BudgetExceeded) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"The AI could not answer ({type(exc).__name__}). Nothing was changed.")


def step_results() -> None:
    run_id = ss.wiz_run_id
    if not run_id:
        st.info("No prioritization has been run in this session yet.")
        nav(3, None)
        return
    with data.write_session() as s:
        rows = dp.ranking(s, run_id)
    complete = [r for r in rows if r.priority_status == "COMPLETE"]
    incomplete = [r for r in rows if r.priority_status != "COMPLETE"]
    st.markdown("### Decision results")
    st.caption("Scores are calculated in Python from validated evidence and your agent weights; each run is saved and "
               "never overwritten.")
    blocker = paid_call_blocker()
    est = 0.0
    with st.expander("ARGUE settings (OpenAI)"):
        allow_paid = st.checkbox("Allow paid AI replies", key="w_allow_chat", disabled=bool(blocker))
        if rows:
            with data.write_session() as s:
                est = argue_chat.estimate(settings, argue_chat.build_context(s, run_id, rows[0].requirement_id), [], "")
        cap = st.number_input("Maximum cost per reply (USD)", 0.0, 0.5, round(max(est * 3, 0.01), 3), 0.005,
                              key="w_chat_cap")
        st.caption(f"Each reply is estimated at about ${est:.4f}. Opening a panel never calls the API.")
    for r in complete:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([5, 2, 1, 1])
            c1.markdown(f"<div class='pm-rank'><span class='r'>Rank {r.rank}</span><span class='t'>{esc(r.title)}</span>"
                        f"</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='pm-score'>{r.final_score:g} <span>/ 100</span></div>", unsafe_allow_html=True)
            wk, ak = f"why_{run_id}_{r.requirement_id}", f"argue_{run_id}_{r.requirement_id}"
            if c3.button("WHY?", key=f"b_{wk}", use_container_width=True):
                ss[wk] = not ss.get(wk, False)
            if c4.button("ARGUE", key=f"b_{ak}", use_container_width=True):
                ss[ak] = not ss.get(ak, False)
            if ss.get(wk):
                why_panel(run_id, r.requirement_id)
            if ss.get(ak):
                st.markdown("##### Discuss this priority")
                argue_panel(run_id, r.requirement_id, allow_paid, cap, blocker)
    if not complete:
        st.info("No requirement could be fully assessed, so there is no formal ranking.")
    if incomplete:
        st.markdown("#### Not ranked (incomplete)")
        for r in incomplete:
            with st.container(border=True):
                st.markdown(f"**{esc(r.title)}**")
                st.caption("Missing assessments: " + ", ".join(r.missing_agents) + ". Missing weight is not "
                           "redistributed.")
                wk = f"why_{run_id}_{r.requirement_id}"
                if st.button("WHY?", key=f"b_{wk}"):
                    ss[wk] = not ss.get(wk, False)
                if ss.get(wk):
                    why_panel(run_id, r.requirement_id)
    with st.expander("What-if: different weights (not saved)"):
        from services import what_if_simulator as wis

        with data.write_session() as s:
            base = wis.current_weights(s, run_id)
            names = {a["agent_id"]: a["name"] for a in pw.available_agents(s)}
        trial = {aid: st.number_input(names.get(aid, aid), 0.0, 100.0, float(w), 0.5, key=f"wi_{run_id}_{aid}")
                 for aid, w in base.items()}
        if st.button("Simulate", disabled=round(sum(trial.values()), 2) != 100):
            with data.write_session() as s:
                sim = wis.simulate(s, run_id, trial)
            st.dataframe([{"Requirement": next((x.title for x in rows if x.requirement_id == x2.requirement_id), ""),
                           "Saved rank": x2.original_rank, "What-if rank": x2.simulated_rank,
                           "Saved score": x2.original_final, "What-if score": x2.simulated_final} for x2 in sim.rows],
                         hide_index=True, width="stretch")
    b1, b2, b3 = st.columns(3)
    if b1.button("Adjust agents or weights (new run)", use_container_width=True):
        go(3)
    with b2:
        if st.button("Prepare report (Excel)", use_container_width=True, disabled=not actor.strip()):
            from services import management_reports as mr

            with data.write_session() as s:
                content, name = mr.export_report(s, run_id, "xlsx", actor)
            st.download_button("Download report", content, file_name=name, use_container_width=True)
    if b3.button("Start a new prioritization", use_container_width=True):
        for k in ("wiz_domains", "wiz_features", "wiz_reqs", "wiz_agents"):
            ss[k] = []
        ss.wiz_run_id = None
        go(1)


{1: step_area, 2: step_requirements, 3: step_agents, 4: step_results}[ss.wiz_step]()
