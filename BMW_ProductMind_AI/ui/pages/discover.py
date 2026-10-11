"""Guided workflow screen 1: Discover customer needs."""
from __future__ import annotations

from sqlalchemy import func, select

import pandas as pd
import streamlit as st

from core.ai_models import AIFeedbackExtraction
from core.models import SourceDocument
from services import ai_feedback_analysis as ai
from services import evidence_extraction as ee
from services import llm_budget as budget
from services import requirement_review as rr
from services.dashboard_queries import DashboardFilter
from ui import data
from ui.evidence_ui import budget_kpis, paid_client, llm_settings, paid_call_blocker
from ui.theme import apply_theme, badge, card, chip, esc, header, kpi, workflow

apply_theme()
workflow("DISCOVER")
header("BMW ProductMind AI", "From Customer Voices to Future Vehicle Decisions")
st.caption("Vehicle: BMW 5 Series G60")

key = data.db_key()
options = data.filter_options(key)

with data.write_session() as s:
    requirements = rr.list_requirements(s)
    source_count = s.scalar(select(func.count()).select_from(SourceDocument)) or 0
    ai_processed = s.scalar(
        select(func.count(func.distinct(AIFeedbackExtraction.original_feedback_id))).where(
            AIFeedbackExtraction.vehicle_code == "G60",
            AIFeedbackExtraction.status == "succeeded",
        )
    ) or 0

base = data.overview(key, DashboardFilter().model_dump_json())
c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{base.unique_feedback:,}", "Customer comments analyzed", f"AI-processed comments: {ai_processed:,}")
kpi(c2, f"{base.need_categories_with_feedback}", "Customer needs identified", "derived from the database taxonomy")
kpi(c3, f"{source_count}", "Available evidence sources", "internal, public and briefing material")
kpi(c4, f"{len(requirements)}", "Requirements created", "saved product requirement drafts")

# ---------------------------------------------------------------------------
# 2.0 AI evidence extraction (extract once, reuse everywhere)
# ---------------------------------------------------------------------------
settings = llm_settings()
with data.write_session() as s:
    ext = ee.extraction_status(s, settings)
    ai_counts = ee.counts_by_category(s, settings)
st.markdown("### AI Evidence Extraction")
e1, e2, e3, e4, e5 = st.columns(5)
kpi(e1, f"{ext['processed_comments']:,} / {ext['g60_comments']:,}", "Comments with validated observations",
    f"{ext['failed_comments']} failed · version {ext['observation_version_id']}")
kpi(e2, f"{ext['validated_observations']:,}", "Validated observations", f"{ext['rejected_observations']} rejected by validation")
kpi(e3, f"{ext['comments_by_type'].get('COMPLAINT', 0):,} / {ext['comments_by_type'].get('IMPROVEMENT_REQUEST', 0):,}",
    "Complaint / request comments", "distinct comments, all categories")
budget_kpis([e4, e5])
st.caption(f"Model {ext['model']} · prompt {ext['prompt_version']} · taxonomy {ext['taxonomy_version']}. "
           "The AI identifies evidence only; every excerpt is checked verbatim against the original comment.")

with st.expander("Run AI extraction (paid - explicit approval required)"):
    blocker = paid_call_blocker()
    if blocker:
        st.info(blocker + " Estimates are still shown; no request is sent.")
    with data.write_session() as s:
        sample = ee.plan_sample(s, settings)
        full = ee.plan_full(s, settings)
    est = sample.estimate
    st.markdown("**Step 1 - representative 20-comment sample** (spread across need categories)")
    st.write(f"{est.to_send} of {est.comments} comments still to send ({est.cached} cached) · estimate "
             f"${est.estimated_usd:.2f}, up to ${est.upper_bound_usd:.2f} with retries · {est.basis}.")
    approve = st.number_input("Approve up to (USD) for the sample", 0.0, 26.0,
                              float(round(est.upper_bound_usd + 0.005, 2)), 0.05, key="disc_sample_approve")
    if st.button("Run 20-comment sample", type="primary", disabled=bool(blocker) or est.to_send == 0):
        try:
            with st.spinner("Extracting sample - each batch is checkpointed"), data.write_session() as s:
                job = ee.run_observation_extraction(s, paid_client(), settings, sample, approved_max_usd=approve)
            st.success(f"Sample job {job.job_id}: {job.status}, {job.processed_items} processed, "
                       f"{job.failed_items} failed, ${job.estimated_cost_usd:.4f}.")
            st.cache_data.clear()
        except (budget.ApprovalRequired, budget.BudgetExceeded, ai.FullRunNotConfirmed, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface API errors without leaking credentials
            st.error(f"Extraction stopped: {type(exc).__name__}. Completed batches are kept; resume via the CLI.")
    sample_jobs = [j for j in ext["jobs"] if j["total"] <= ai.FULL_RUN_THRESHOLD]
    if sample_jobs:
        st.markdown("**Review extraction quality** (latest sample)")
        with data.write_session() as s:
            rows = ee.sample_review_rows(s, sample_jobs[0]["job_id"])
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fe = full.estimate
    st.markdown("**Step 2 - full extraction of all G60 comments**")
    st.write(f"{fe.to_send:,} comments still to send · estimate ${fe.estimated_usd:.2f}, up to "
             f"${fe.upper_bound_usd:.2f} with retries · {fe.basis}.")
    if not ext["sample_completed"]:
        st.warning("Run and review the 20-comment sample first. Full extraction is blocked until then.")
    else:
        st.caption("Full extraction runs for a long time, so it is started from the terminal with explicit "
                   "confirmation and an approved amount; it stops before exceeding either limit and can be resumed:")
        st.code(f"python -m scripts.customer_intelligence observe --all --confirm-full-run "
                f"--approve-usd {min(fe.upper_bound_usd, 26):.2f}", language="bash")
    if ext["jobs"]:
        st.dataframe(pd.DataFrame(ext["jobs"]), hide_index=True, width="stretch")

st.markdown("### Customer Needs")
f1, f2, f3 = st.columns([1, 1, 1.4])
markets = f1.multiselect("Market", options["countries"], key="wf_market")
engines = f2.multiselect("Engine type", options["engine_types"], key="wf_engine")
need_options = {n["code"]: n["name"] for n in options["needs"] if n["code"] != "OTHER"}
need_pick = f3.multiselect("Customer need", list(need_options), format_func=need_options.get, key="wf_need_filter")
with st.expander("Advanced filters"):
    vfc = st.multiselect("Original technical topic", options["vfc_names"], key="wf_vfc")
    ftypes = st.multiselect("Original feedback label", options["feedback_types"], key="wf_type",
                            format_func=lambda x: "(not provided)" if x is None else x)

flt = DashboardFilter(
    countries=markets or None,
    engine_types=engines or None,
    need_codes=need_pick or None,
    vfc_names=vfc or None,
    feedback_types=ftypes or None,
)
overview = data.overview(key, flt.model_dump_json())
needs = [n for n in overview.by_need if n.need_code != "OTHER"]

if not needs:
    st.info("No customer needs match these filters. Remove a filter to widen the selection.")
    st.stop()

selected = st.session_state.get("workflow_need_code")
if selected not in {n.need_code for n in needs}:
    selected = needs[0].need_code
    st.session_state["workflow_need_code"] = selected
    st.session_state["workflow_cluster_id"] = f"NEED-{selected}"

for row in range(0, min(len(needs), 12), 3):
    cols = st.columns(3)
    for col, need in zip(cols, needs[row:row + 3]):
        body = (f"{need.unique_feedback_count:,} unique G60 comments are mapped to this need. "
                "Open it to inspect original comments, supporting evidence and limitations.")
        ac = ai_counts.get(need.need_code, {})
        ai_chip = (f"AI: {ac.get('COMPLAINT', 0)} complaints · {ac.get('IMPROVEMENT_REQUEST', 0)} requests"
                   if ext["processed_comments"] else "AI extraction pending")
        col.markdown(card(need.need_name, body, f"{need.unique_feedback_count:,} comments", ai_chip,
                          "evidence available" if need.unique_feedback_count else "no matching comments"),
                     unsafe_allow_html=True)
        if col.button("Explore Need", key=f"discover_{need.need_code}", type="primary" if need.need_code == selected else "secondary"):
            st.session_state["workflow_need_code"] = need.need_code
            st.session_state["workflow_cluster_id"] = f"NEED-{need.need_code}"
            st.rerun()

selected = st.session_state["workflow_need_code"]
cluster_id = st.session_state.get("workflow_cluster_id", f"NEED-{selected}")
st.markdown("### Need Overview")
pkg = data.evidence_package(key, cluster_id)
st.markdown(chip(pkg.need_name) + chip(f"{pkg.unique_feedback_count:,} direct comments")
            + (badge("REVIEW", "mapping requires review") if pkg.mapping_requires_review else badge("SUPPORTING", "evidence linked")),
            unsafe_allow_html=True)

left, right = st.columns([1.2, 1])
with left:
    st.markdown("#### Customer problem overview")
    st.write(
        f"{pkg.need_name} is represented by {pkg.unique_feedback_count:,} distinct G60 customer comments. "
        f"The current selection is broad, so the next screen will create an evidence-based draft that can be refined."
    )
    st.markdown("#### Representative original comments")
    for item in pkg.direct_evidence[:5]:
        with st.expander(f"{item.feedback_id} · {item.country or 'market n/a'} · {item.engine_type or 'engine n/a'}"):
            st.write(item.text)
            st.caption(f"{item.source_file} · {item.source_sheet} · row {item.source_row}")
    st.markdown("#### AI-identified customer needs (validated observations)")
    with data.write_session() as s:
        summary = ee.need_summary(s, settings, selected)
    if not summary["top_needs"]:
        st.caption("No validated AI observations in this category yet. Run the extraction sample above.")
    else:
        ct = summary["comments_by_type"]
        st.markdown(chip(f"{ct.get('COMPLAINT', 0)} complaint comments") + chip(f"{ct.get('IMPROVEMENT_REQUEST', 0)} requests")
                    + chip(f"{ct.get('POSITIVE', 0)} positive"), unsafe_allow_html=True)
        st.caption(f"Counts cover the {ext['processed_comments']:,} comments processed so far, not all comments.")
        st.dataframe(pd.DataFrame(summary["top_needs"]), hide_index=True, width="stretch")
        for ex in summary["examples"][:4]:
            with st.expander(f"{ex['feedback_id']} · {ex['observation_type'].replace('_', ' ').title()}: {ex['customer_need']}"):
                st.markdown(f"**Evidence excerpt:** {esc(ex['evidence_excerpt'])}", unsafe_allow_html=True)
                st.caption("Original comment")
                st.write(ex["original_comment"])
with right:
    st.markdown("#### Related public feedback")
    public_items = [i for i in pkg.supporting_evidence if i.kind in ("public_owner_feedback", "public_bmw_observation")]
    if public_items:
        for item in public_items[:4]:
            st.markdown(f"**{esc(item.title)}**")
            st.caption(item.detail or item.mapping_note or "")
    else:
        st.caption("No public BMW owner observation is mapped to this need.")
    st.markdown("#### Competitor observations")
    competitors = [i for i in pkg.contextual_evidence if i.kind == "competitor_observation"]
    if competitors:
        for item in competitors[:4]:
            st.markdown(f"**{esc(item.title)}**")
            st.caption(item.detail or item.mapping_note or "")
    else:
        st.caption("No competitor observation is mapped to this need.")

with st.expander("Data limitations"):
    for msg in pkg.missing_evidence + pkg.limitations:
        st.markdown(f"- {esc(msg)}")

if st.button("Create Product Requirement", type="primary"):
    st.session_state["req_target"] = {"cluster_id": cluster_id}
    st.switch_page("ui/pages/define.py")
