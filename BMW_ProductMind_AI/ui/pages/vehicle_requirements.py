"""Page 5 - Vehicle Requirements: evidence -> rule-based draft -> edit -> human review."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from services import requirement_generator as gen
from services import requirement_review as rr
from services.need_taxonomy import MISSING_VFC
from ui import data
from ui.theme import apply_theme, badge, chip, esc, header, kpi

STATUS_COLORS = {"DRAFT": "#7F8BA6", "PENDING_REVIEW": "#C98500", "APPROVED": "#199E70", "REJECTED": "#D95926"}

apply_theme()
header("Vehicle Requirements",
       "Turn a G60 customer need into a draft requirement for the next-generation BMW 5 Series. "
       "Drafts are evidence-based and rule-based; product managers edit, review and decide.")

key = data.db_key()
need_names = {n["code"]: n["name"] for n in data.filter_options(key)["needs"]}
clusters = {c.cluster_id: c for c in data.cluster_options(key)}
insights = data.insight_options(key)

with data.write_session() as s:
    counts = rr.status_counts(s)
c1, c2, c3, c4 = st.columns(4)
for col, status in zip((c1, c2, c3, c4), ("DRAFT", "PENDING_REVIEW", "APPROVED", "REJECTED")):
    kpi(col, counts.get(status, 0), status.replace("_", " ").title(), "requirements")

# ---------------------------------------------------------------------------
# A - select customer need
# ---------------------------------------------------------------------------
st.markdown("### A · Select customer need")
target = st.session_state.pop("req_target", None)  # set by other pages
if target:
    if "insight_id" in target:
        st.session_state["req_mode"] = "Customer insight"
        st.session_state["req_insight"] = target["insight_id"]
    elif target.get("cluster_id", "").startswith("NEED-"):
        st.session_state["req_mode"] = "Customer-need category"
        st.session_state["req_need"] = target["cluster_id"].removeprefix("NEED-")
    elif "cluster_id" in target:
        cid = target["cluster_id"]
        st.session_state["req_mode"] = "VFC cluster"
        st.session_state["req_vfc_need"] = clusters[cid].need_code if cid in clusters else None
        st.session_state["req_cluster"] = cid
    st.session_state.pop("req_draft", None)

mode = st.radio("Select by", ["Customer insight", "VFC cluster", "Customer-need category"], horizontal=True, key="req_mode")
if mode == "Customer insight":
    ids = [i["insight_id"] for i in insights]
    labels = {i["insight_id"]: f"{i['title']} · {i['count']} comments · {i['status']}" for i in insights}
    if st.session_state.get("req_insight") not in ids:
        st.session_state["req_insight"] = ids[0] if ids else None
    insight_id = st.selectbox("Existing customer insight", ids, format_func=labels.get, key="req_insight")
    cluster_id = next((i["cluster_id"] for i in insights if i["insight_id"] == insight_id), None)
elif mode == "VFC cluster":
    a, b = st.columns([1, 1.5])
    need = a.selectbox("Customer-need category", list(need_names), format_func=need_names.get, key="req_vfc_need")
    options = [c.cluster_id for c in clusters.values() if c.need_code == need and c.cluster_level != "need_category"]
    if st.session_state.get("req_cluster") not in options:
        st.session_state["req_cluster"] = options[0] if options else None
    cluster_id = b.selectbox(
        "VFC cluster", options, key="req_cluster",
        format_func=lambda c: f"{'(empty VFC cell)' if clusters[c].label == MISSING_VFC else clusters[c].label} · "
                              f"{clusters[c].unique_feedback_count} comments" + (" · ⚠ review" if clusters[c].requires_review else ""))
else:
    need = st.selectbox("Customer-need category", list(need_names), format_func=need_names.get, key="req_need")
    cluster_id = f"NEED-{need}"
    st.caption("A whole category spans several VFC classes - drafts at this level are problem-oriented and require refinement.")

if not cluster_id or cluster_id not in clusters:
    st.info("Select a customer need to continue.")
    st.stop()
if clusters[cluster_id].need_code == "OTHER":
    st.warning("'Other / Unclassified' comments have no usable classification; no requirement can be derived from them.")
    st.stop()

pkg = data.evidence_package(key, cluster_id)

# ---------------------------------------------------------------------------
# B - review evidence
# ---------------------------------------------------------------------------
st.markdown("### B · Review evidence")
st.markdown(chip(f"Need: {pkg.need_name}") + chip(f"Cluster: {pkg.cluster_label} ({pkg.cluster_id})")
            + (chip(f"Insight: {pkg.insight_id}") if pkg.insight_id else "")
            + (badge("REVIEW", "⚠ mapping requires review") if pkg.mapping_requires_review else ""),
            unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
kpi(k1, f"{pkg.unique_feedback_count:,}", "Unique G60 comments", "direct evidence")
kpi(k2, f"{pkg.classification_count:,}", "Classification rows", "not customers")
kpi(k3, f"{pkg.label_groups.get('issue', 0)} / {pkg.label_groups.get('request', 0)} / {pkg.label_groups.get('positive', 0)}",
    "Issue / request / positive", "distinct comments by VFC level-3 label group")
kpi(k4, f"{len(pkg.supporting_evidence)} / {len(pkg.contextual_evidence) + len(pkg.equipment_context)}",
    "Supporting / contextual", "linked at need-category level")

t1, t2, t3, t4 = st.tabs([f"Direct · {pkg.unique_feedback_count}", f"Supporting · {len(pkg.supporting_evidence)}",
                          f"Contextual · {len(pkg.contextual_evidence) + len(pkg.equipment_context)}",
                          f"Limitations · {len(pkg.limitations) + len(pkg.missing_evidence)}"])
with t1:
    st.caption(f"First {len(pkg.direct_evidence)} of {pkg.unique_feedback_count} comments (all IDs are attached to the draft).")
    for d in pkg.direct_evidence:
        with st.expander(f"{d.feedback_id} · {d.country} · {d.engine_type} — {' '.join(d.text.split())[:100]}"):
            st.markdown(badge("DIRECT") + "".join(chip(l) for l in d.vfc_labels), unsafe_allow_html=True)
            st.markdown(f'<div class="pm-text">{esc(d.text)}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="pm-src">{esc(d.source_file)} · {esc(d.source_sheet)} · row {d.source_row}</span>',
                        unsafe_allow_html=True)
with t2:
    st.caption("Public BMW observations and BMW studies linked to the need category - not proof of individual complaints.")
    for i in pkg.supporting_evidence:
        st.markdown(badge("SUPPORTING") + chip(i.kind.replace("_", " ")) + (badge("REVIEW") if i.requires_review else "")
                    + f" **{esc(i.title)}**", unsafe_allow_html=True)
        if i.source_url:
            st.caption(f"Source URL: {i.source_url}")
with t3:
    st.caption("Competitor observations and current G60 option-list equipment: context only.")
    for i in pkg.contextual_evidence:
        st.markdown(badge("CONTEXTUAL") + (badge("REVIEW") if i.requires_review else "") + f" **{esc(i.title)}**",
                    unsafe_allow_html=True)
    if pkg.equipment_context:
        st.dataframe(pd.DataFrame([e.model_dump() for e in pkg.equipment_context]), hide_index=True, width="stretch")
with t4:
    for m in pkg.missing_evidence:
        st.warning(m)
    for lim in pkg.limitations:
        st.markdown(f"- {esc(lim)}")

# ---------------------------------------------------------------------------
# C - generate
# ---------------------------------------------------------------------------
st.markdown("### C · Generate requirement draft")
with data.write_session() as s:
    existing = rr.list_requirements(s, cluster_id=cluster_id)
if existing:
    with st.expander(f"Existing requirements for this cluster ({len(existing)})", expanded=True):
        for r in existing:
            a, b = st.columns([4, 1])
            a.markdown(badge("TEMPLATE", r["status"]).replace("#7F8BA6", STATUS_COLORS[r["status"]])
                       + f" **{esc(r['title'])}** · {esc(r['requirement_id'])} · v{r['version']} · {esc(r['method'])}",
                       unsafe_allow_html=True)
            if b.button("Open", key=f"open_{r['requirement_id']}"):
                st.session_state["req_open_id"] = r["requirement_id"]
                st.session_state.pop("req_draft", None)
                st.rerun()

if st.button("Generate Requirement Draft", type="primary"):
    try:
        st.session_state["req_draft"] = gen.generate_rule_based(pkg).model_dump_json()
        st.session_state.pop("req_open_id", None)
    except gen.NotGeneratable as exc:
        st.error(str(exc))

gate = gen.llm_generation_status()
with st.expander("LLM-assisted drafting (optional)" + ("" if gate.enabled else " - disabled")):
    if not gate.enabled:
        st.info("Disabled. External API calls are not authorised by default, and internal BMW feedback is not sent "
                "to external providers.")
        for r in gate.reasons:
            st.markdown(f"- {esc(r)}")
    else:
        est = gen.estimate_llm_cost(pkg)
        st.warning(f"This sends up to 10 internal G60 comments and the draft to {est.model}. Estimated cost "
                   f"~${est.estimated_cost_usd:.3f} (~{est.input_tokens:,} input / {est.output_tokens:,} output tokens). "
                   "No fallback to another model.")
        ok = st.checkbox("I confirm sending this evidence to the external LLM provider", key="req_llm_confirm")
        if st.button("Generate LLM-assisted draft", disabled=not ok):
            try:
                st.session_state["req_draft"] = gen.generate_llm_assisted(pkg, confirmed=ok, cost_acknowledged=est).model_dump_json()
                st.session_state.pop("req_open_id", None)
            except Exception as exc:  # noqa: BLE001 - shown to the user; rule-based draft stays available
                st.error(f"LLM-assisted generation failed: {exc}")

# ---------------------------------------------------------------------------
# D/E - edit, save, review
# ---------------------------------------------------------------------------
open_id = st.session_state.get("req_open_id")
draft_json = st.session_state.get("req_draft")
if not open_id and not draft_json:
    st.stop()

with data.write_session() as s:
    saved = rr.requirement_as_dict(s, rr.get_requirement(s, open_id)) if open_id else None
if saved and saved["source_cluster_id"] != cluster_id:
    st.session_state.pop("req_open_id", None)
    st.stop()
draft = gen.RequirementDraft.model_validate_json(draft_json) if not saved else None
cur = saved or {**draft.model_dump(), "acceptance_criteria": [c.model_dump() for c in draft.acceptance_criteria],
                "review_status": "UNSAVED", "validation": [], "requirement_id": None}

st.markdown("### D · Edit requirement")
method = cur["generation_method"]
label = {"RULE_BASED": "Evidence-based draft — not AI-generated",
         "LLM_ASSISTED": "LLM-assisted draft — wording drafted by an external model from the evidence",
         "HUMAN_AUTHORED": "Human-authored"}[method]
st.markdown(badge("SUPPORTING", label) + chip(f"Status: {cur['review_status']}")
            + (chip(cur["requirement_id"]) if cur["requirement_id"] else chip("not saved yet"))
            + chip(f"Type: {cur['requirement_type']}"), unsafe_allow_html=True)
if cur["requires_refinement"]:
    st.warning("Problem-oriented draft — requires refinement. Missing information: "
               + " ".join(f"• {n}" for n in cur["refinement_notes"]))
st.caption(f"Vehicle: {cur['vehicle_model']} → {cur['target_generation']}")

form_key = cur["requirement_id"] or "new"
title = st.text_input("Title", cur["title"], key=f"f_title_{form_key}")
statement = st.text_area("Requirement statement", cur["proposed_requirement"], height=110, key=f"f_stmt_{form_key}")
need_text = st.text_area("Customer need", cur["customer_need"], height=70, key=f"f_need_{form_key}")
problem = st.text_area("Customer problem (from evidence)", cur["customer_problem"], height=90, key=f"f_prob_{form_key}")
criteria_df = st.data_editor(pd.DataFrame(cur["acceptance_criteria"]), num_rows="dynamic", width="stretch",
                             key=f"f_crit_{form_key}",
                             column_config={"target_status": st.column_config.SelectboxColumn(
                                 options=["proposed_requires_validation", "evidence_baseline"])})
assumptions = st.text_area("Assumptions (one per line)", "\n".join(cur["assumptions"]), height=160, key=f"f_ass_{form_key}")
st.caption("Evidence summary: " + cur["evidence_summary"])

edits = {
    "title": title, "proposed_requirement": statement, "customer_need": need_text, "customer_problem": problem,
    "acceptance_criteria": [{k: ("" if pd.isna(v) else v) for k, v in row.items()}
                            for row in criteria_df.to_dict("records") if str(row.get("criterion") or "").strip()],
    "assumptions": [a for a in assumptions.splitlines() if a.strip()],
}
for c in edits["acceptance_criteria"]:
    c.setdefault("target", gen.PROPOSED_TARGET)
    c["target_status"] = c.get("target_status") or "proposed_requires_validation"

st.markdown("### E · Save and review")
actor = st.text_input("Your reviewer / editor ID", value=st.session_state.get("req_actor", os.getenv("USER", "")),
                      key="req_actor")
reason = st.text_input("Reason / comment (required for approve, reject and reopen)", key=f"f_reason_{form_key}")
b1, b2, b3, b4, b5 = st.columns(5)


def _run(fn):
    """Run a workflow action in its own session; actions return plain IDs (no ORM objects leave the session)."""
    try:
        with data.write_session() as s:
            result = fn(s)
        st.cache_data.clear()
        return result
    except (rr.InvalidTransition, ValueError) as exc:
        st.error(str(exc))
        return None


if b1.button("Save Draft", width="stretch"):
    if saved:
        changes = {k: v for k, v in edits.items() if v != saved[k]}
        if _run(lambda s: rr.edit_requirement(s, saved["requirement_id"], changes, actor, reason or None).requirement_id):
            st.success("Saved as a new version." if changes else "No changes to save.")
    else:
        changes = {k: v for k, v in edits.items() if v != cur[k]}
        new_id = _run(lambda s: rr.create_requirement(s, draft, actor, edited=changes or None,
                                                       edit_note=reason or None).requirement_id)
        if new_id:
            st.session_state["req_open_id"] = new_id
            st.session_state.pop("req_draft", None)
            st.rerun()
if b2.button("Submit for Review", width="stretch", disabled=not saved):
    if _run(lambda s: rr.submit_for_review(s, saved["requirement_id"], actor, reason or None).requirement_id):
        st.rerun()
if b3.button("Approve", width="stretch", disabled=not saved):
    if _run(lambda s: rr.approve(s, saved["requirement_id"], actor, reason).requirement_id):
        st.rerun()
if b4.button("Reject", width="stretch", disabled=not saved):
    if _run(lambda s: rr.reject(s, saved["requirement_id"], actor, reason).requirement_id):
        st.rerun()
if b5.button("Reopen", width="stretch", disabled=not saved):
    if _run(lambda s: rr.reopen(s, saved["requirement_id"], actor, reason).requirement_id):
        st.rerun()
if saved and st.button("Add review comment", disabled=not reason):
    if _run(lambda s: rr.add_comment(s, saved["requirement_id"], actor, reason).requirement_id):
        st.rerun()
if saved and saved["requires_refinement"] and st.button("Mark as refined (after editing the statement)"):
    if _run(lambda s: rr.mark_refined(s, saved["requirement_id"], actor, reason).requirement_id):
        st.rerun()

if not saved:
    st.info("Unsaved draft. Save it to create a persistent requirement ID; the generated text is kept as version 1.")
else:
    st.markdown(badge("TEMPLATE", f"Status: {saved['review_status']}").replace("#7F8BA6", STATUS_COLORS[saved["review_status"]]),
                unsafe_allow_html=True)
    issues = saved["validation"]
    errors = [i for i in issues if i["severity"] == "error"]
    if errors:
        st.error("Validation errors (block submission/approval):\n" + "\n".join(f"- {i['field']}: {i['message']}" for i in errors))
    elif issues:
        st.info("Validation passed with notes:\n" + "\n".join(f"- {i['field']}: {i['message']}" for i in issues))
    else:
        st.success("Validation passed.")
    with data.write_session() as s:
        hist = rr.history(s, saved["requirement_id"])
        vers = rr.versions(s, saved["requirement_id"])
    with st.expander(f"Review history ({len(hist)})", expanded=True):
        st.dataframe(pd.DataFrame([h.model_dump() for h in hist]), hide_index=True, width="stretch")
    with st.expander(f"Versions ({len(vers)}) - version 1 is the original generated draft"):
        for v in vers:
            st.markdown(f"**v{v['version']}** · {v['change_type']} · {v['generation_method']} · by {esc(v['changed_by'])} "
                        f"· {v['created_at']:%Y-%m-%d %H:%M} · changed: {', '.join(v['changed_fields']) or '-'}")
            st.caption(v["snapshot"]["proposed_requirement"])
    with st.expander("Evidence attached to this requirement"):
        st.json(saved["supporting_evidence_ids"], expanded=False)
