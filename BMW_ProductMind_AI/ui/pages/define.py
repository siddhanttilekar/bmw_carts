"""Guided workflow screen 2: Define product requirements."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from services import ai_evidence_selector as sel
from services import ai_feedback_analysis as ai
from services import llm_budget as budget
from services import requirement_generator as gen
from services import requirement_review as rr
from ui import data
from ui.evidence_ui import paid_client, llm_settings, paid_call_blocker
from ui.theme import apply_theme, badge, chip, esc, header, kpi, workflow

apply_theme()
workflow("DEFINE")
header("Define", "Turn a selected customer need into an evidence-based vehicle requirement.")

key = data.db_key()
need_names = {n["code"]: n["name"] for n in data.filter_options(key)["needs"]}
clusters = {c.cluster_id: c for c in data.cluster_options(key)}

target = st.session_state.pop("req_target", None)
if target and "cluster_id" in target:
    st.session_state["workflow_cluster_id"] = target["cluster_id"]
    if target["cluster_id"] in clusters:
        st.session_state["workflow_need_code"] = clusters[target["cluster_id"]].need_code

if "workflow_cluster_id" not in st.session_state:
    st.info("No customer need selected yet. Start by choosing a need in Discover.")
    if st.button("Go to Discover"):
        st.switch_page("ui/pages/discover.py")
    st.stop()

cluster_id = st.session_state["workflow_cluster_id"]
if cluster_id not in clusters:
    st.error("The selected need is no longer available. Choose another need in Discover.")
    st.stop()

pkg = data.evidence_package(key, cluster_id)
left, right = st.columns([0.92, 1.28])

with left:
    st.markdown("### Selected Need")
    st.markdown(chip(pkg.need_name) + chip(f"{pkg.unique_feedback_count:,} direct comments")
                + (badge("REVIEW", "review mappings") if pkg.mapping_requires_review else badge("SUPPORTING", "evidence linked")),
                unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    kpi(k1, f"{pkg.unique_feedback_count:,}", "Customer comments", "direct G60 evidence")
    kpi(k2, f"{len(pkg.supporting_evidence)}", "Supporting items", "public feedback and studies")
    st.markdown("#### Evidence")
    for item in pkg.direct_evidence[:4]:
        with st.expander(f"{item.feedback_id} · {item.country or 'market n/a'}"):
            st.write(item.text)
            st.caption(f"{item.source_file} · row {item.source_row}")
    with st.expander("Supporting evidence and limitations"):
        for item in pkg.supporting_evidence[:8]:
            st.markdown(f"- **{esc(item.title)}**")
        for msg in pkg.missing_evidence + pkg.limitations:
            st.caption(msg)

with right:
    st.markdown("### Requirement Workspace")
    with data.write_session() as s:
        existing = rr.list_requirements(s, cluster_id=cluster_id)
    existing_ids = [r["requirement_id"] for r in existing]
    chosen_existing = None
    if existing:
        selected_existing = st.selectbox(
            "Saved requirements for this need",
            ["new"] + existing_ids,
            format_func=lambda rid: "Create a new requirement" if rid == "new" else next(
                f"{r['title']} · {r['status']} · {rid}" for r in existing if r["requirement_id"] == rid),
            key="define_existing",
        )
        chosen_existing = None if selected_existing == "new" else selected_existing
        if chosen_existing:
            st.session_state["workflow_requirement_id"] = chosen_existing

    if chosen_existing:
        with data.write_session() as s:
            cur = rr.requirement_as_dict(s, rr.get_requirement(s, chosen_existing))
        draft = None
    else:
        if st.button("Generate Evidence-Based Draft", type="primary"):
            try:
                st.session_state["define_draft"] = gen.generate_rule_based(pkg).model_dump_json()
            except gen.NotGeneratable as exc:
                st.error(str(exc))
        st.button("Generate with AI", disabled=True, help="AI generation is reserved for a later upgrade and is not configured.")
        if "define_draft" not in st.session_state:
            st.info("Generate an evidence-based draft to begin. No external AI call is made.")
            st.stop()
        draft = gen.RequirementDraft.model_validate_json(st.session_state["define_draft"])
        cur = {**draft.model_dump(), "acceptance_criteria": [c.model_dump() for c in draft.acceptance_criteria],
               "assumptions": draft.assumptions, "review_status": "UNSAVED", "requirement_id": None,
               "current_version": 0, "validation": []}

    st.markdown(badge("SUPPORTING", "Evidence-based draft") + chip(f"Status: {cur['review_status']}")
                + (chip(cur["requirement_id"]) if cur["requirement_id"] else chip("not saved yet")),
                unsafe_allow_html=True)
    if cur.get("requires_refinement"):
        st.warning("This draft needs refinement before approval because the selected need is broad or evidence is thin.")

    form_key = cur["requirement_id"] or "new"
    title = st.text_input("Requirement name", cur["title"], key=f"def_title_{form_key}")
    problem = st.text_area("Customer problem", cur["customer_problem"], height=95, key=f"def_problem_{form_key}")
    need_text = st.text_area("Customer need", cur["customer_need"], height=80, key=f"def_need_{form_key}")
    statement = st.text_area("Proposed requirement", cur["proposed_requirement"], height=120, key=f"def_statement_{form_key}")
    st.markdown("#### Acceptance criteria")
    criteria_df = st.data_editor(pd.DataFrame(cur["acceptance_criteria"]), num_rows="dynamic", width="stretch",
                                 key=f"def_criteria_{form_key}")
    assumptions = st.text_area("Assumptions", "\n".join(cur["assumptions"]), height=120, key=f"def_ass_{form_key}")
    with st.expander("Supporting evidence references"):
        st.write(cur["evidence_summary"])
        if cur.get("supporting_evidence_ids"):
            st.json(cur["supporting_evidence_ids"], expanded=False)

    actor = st.text_input("Your reviewer / editor ID", value=st.session_state.get("workflow_actor", os.getenv("USER", "")),
                          key="workflow_actor")
    note = st.text_input("Reason or review note", key=f"def_note_{form_key}")
    edits = {
        "title": title,
        "customer_problem": problem,
        "customer_need": need_text,
        "proposed_requirement": statement,
        "acceptance_criteria": [{k: ("" if pd.isna(v) else v) for k, v in row.items()}
                                for row in criteria_df.to_dict("records") if str(row.get("criterion") or "").strip()],
        "assumptions": [a.strip() for a in assumptions.splitlines() if a.strip()],
    }

    def run_action(fn):
        try:
            with data.write_session() as s:
                out = fn(s)
            st.cache_data.clear()
            return out
        except (rr.InvalidTransition, ValueError) as exc:
            st.error(str(exc))
            return None

    b1, b2, b3 = st.columns(3)
    if b1.button("Save Requirement", type="primary"):
        if chosen_existing:
            changes = {k: v for k, v in edits.items() if v != cur[k]}
            out = run_action(lambda s: rr.edit_requirement(s, chosen_existing, changes, actor, note or None).requirement_id)
        else:
            changes = {k: v for k, v in edits.items() if v != cur[k]}
            out = run_action(lambda s: rr.create_requirement(s, draft, actor, edited=changes or None,
                                                              edit_note=note or None).requirement_id)
        if out:
            st.session_state["workflow_requirement_id"] = out
            st.session_state["prio_requirement"] = out
            st.success(f"Requirement saved: {out}")
    saved_id = st.session_state.get("workflow_requirement_id")
    if b2.button("Submit for Review", disabled=not saved_id):
        if run_action(lambda s: rr.submit_for_review(s, saved_id, actor, note or None).requirement_id):
            st.success("Requirement submitted for review.")
            st.rerun()
    if b3.button("Continue to Prioritization", disabled=not saved_id):
        st.switch_page("ui/pages/prioritize.py")


# ---------------------------------------------------------------------------
# 2.0 requirement-specific evidence and human review
# ---------------------------------------------------------------------------
if chosen_existing:
    st.markdown("### Evidence for This Requirement")
    settings = llm_settings()
    version_id = ai.current_observation_version_id(settings)
    reviewer = st.session_state.get("workflow_actor", "")
    with data.write_session() as s:
        current = sel.latest_selection(s, chosen_existing, version_id, requirement_version=cur["current_version"])
    st.caption("Taxonomy matching links validated AI observations of this requirement's direct comments. It costs nothing "
               "and creates an immutable selection run; human reviews always take precedence over AI matches.")
    if st.button("Select Evidence", type="primary" if current is None else "secondary", disabled=not reviewer.strip()):
        try:
            with data.write_session() as s:
                current = sel.run_selection(s, chosen_existing, version_id, reviewer)
            st.success(f"Evidence selection {current.selection_id} ready.")
        except sel.SelectionError as exc:
            st.error(str(exc))
    if current is None:
        st.info("No evidence selection for this requirement version yet.")
    else:
        c = current.counts()
        cov = current.coverage
        k = st.columns(6)
        kpi(k[0], c["complaints"], "Complaint comments", "validated, distinct")
        kpi(k[1], c["improvement_requests"], "Request comments", "validated, distinct")
        kpi(k[2], c["combined_demand"], "Unique supporting", "union - no double counting")
        kpi(k[3], c["positive"], "Positive comments", "")
        kpi(k[4], c["counter_evidence"], "Counter-evidence", "points the other way")
        kpi(k[5], c["uncertain"], "Uncertain", "not counted until reviewed")
        (st.success if cov["complete"] else st.warning)(
            f"Extraction coverage: {cov['extracted_comments']} of {cov['direct_comments']} direct comments "
            f"({'complete' if cov['complete'] else 'incomplete - AI demand scores stay UNSCORED'}). "
            f"Selection {current.selection_id} · {current.method} · {current.human_reviewed} human-reviewed, "
            f"{current.machine_validated} machine-validated.")
        with data.write_session() as s:
            rows = sel.match_rows(s, current.selection_id)
        t_neg, t_pos, t_review, t_hist = st.tabs(["Negative observations", "Positive observations",
                                                  "Evidence match review", "Review history"])
        cols = ["feedback_id", "observation_type", "customer_need", "evidence_excerpt", "ai_status", "human_decision"]
        with t_neg:
            neg = [r for r in rows if r["observation_type"] in ("COMPLAINT", "IMPROVEMENT_REQUEST") and r["ai_status"] != "NOT_RELEVANT"]
            st.dataframe(pd.DataFrame(neg, columns=cols), hide_index=True, width="stretch")
        with t_pos:
            pos = [r for r in rows if r["observation_type"] == "POSITIVE" and r["ai_status"] != "NOT_RELEVANT"]
            st.dataframe(pd.DataFrame(pos, columns=cols), hide_index=True, width="stretch")
        with t_review:
            status_filter = st.radio("Show", ["UNCERTAIN", "PROPOSED", "NOT_RELEVANT", "All"], horizontal=True,
                                     key=f"def_rev_filter_{chosen_existing}")
            shown = [r for r in rows if status_filter == "All" or r["ai_status"] == status_filter]
            page = st.number_input("Page", 1, max(1, -(-len(shown) // 10)), 1, key=f"def_rev_page_{chosen_existing}")
            for r in shown[(page - 1) * 10: page * 10]:
                label = (f"{r['feedback_id']} · {r['observation_type'].replace('_', ' ').title()} · AI {r['ai_status']}"
                         + (f" · reviewed: {r['human_decision']}" if r["human_decision"] else ""))
                with st.expander(label):
                    st.markdown(f"**Customer need:** {esc(r['customer_need'] or '')} · **System:** "
                                f"{esc(r['vehicle_system'] or '')} · **Category:** {esc(r['taxonomy_category'] or '')}",
                                unsafe_allow_html=True)
                    st.markdown(f"**Evidence excerpt:** {esc(r['evidence_excerpt'] or '')}", unsafe_allow_html=True)
                    st.caption("Original comment (untrusted customer text)")
                    st.write(r["original_comment"])
                    st.caption(f"AI match reason: {r['match_reason']}")
                    key = f"{chosen_existing}_{r['observation_id']}"
                    decision = st.radio("Decision", ["ACCEPTED", "REJECTED", "UNCERTAIN"], horizontal=True, key=f"dec_{key}")
                    why = st.text_input("Justification (required)", key=f"why_{key}")
                    if st.button("Save review", key=f"save_{key}", disabled=not (why.strip() and reviewer.strip())):
                        try:
                            with data.write_session() as s:
                                sel.review_match(s, chosen_existing, r["observation_id"], decision, why, reviewer,
                                                 current.selection_id)
                            st.success("Review saved. It applies to this and every future evidence selection.")
                            st.rerun()
                        except sel.SelectionError as exc:
                            st.error(str(exc))
            if not reviewer.strip():
                st.caption("Enter your reviewer ID above to record reviews.")
        with t_hist:
            with data.write_session() as s:
                hist = sel.review_history(s, chosen_existing)
            if hist:
                st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch")
            else:
                st.caption("No reviews yet.")

        with st.expander("Optional: LLM semantic matching for uncertain matches (paid)"):
            with data.write_session() as s:
                candidates = sel.semantic_candidates(s, current.selection_id)
            blocker = paid_call_blocker()
            st.write(f"{len(candidates)} unreviewed uncertain matches (up to 40 per request). The model only judges "
                     "relevance; it never scores. The result is stored as a new selection run.")
            if blocker:
                st.info(blocker)
            approve = st.number_input("Approve up to (USD)", 0.0, 5.0, 0.5, 0.05, key=f"sem_approve_{chosen_existing}")
            if st.button("Run semantic matching", disabled=bool(blocker) or not candidates or not reviewer.strip()):
                try:
                    with data.write_session() as s:
                        new = sel.run_semantic_matching(s, paid_client(), settings, current.selection_id, reviewer,
                                                        approved_max_usd=approve)
                    st.success(f"New selection {new.selection_id} created.")
                except (sel.SelectionError, budget.ApprovalRequired, budget.BudgetExceeded) as exc:
                    st.error(str(exc))
