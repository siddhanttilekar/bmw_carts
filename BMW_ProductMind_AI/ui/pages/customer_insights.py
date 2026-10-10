"""Page 3 - Customer Insights: stored insights with their derived generation status."""
from __future__ import annotations

import streamlit as st

from services.insight_generator import STATUS_AI_ENRICHED, STATUS_PENDING, STATUS_TEMPLATE
from services.need_taxonomy import MISSING_VFC
from ui import data
from ui.theme import apply_theme, badge, chip, esc, header, kpi

PAGE_SIZE = 12
STATUSES = [STATUS_TEMPLATE, STATUS_AI_ENRICHED, STATUS_PENDING]

apply_theme()
header("Customer Insights",
       "One insight per cluster. Counts and evidence IDs are computed from SQLite and validated against the database.")

key = data.db_key()
cards = data.insight_cards(key)
need_names = {n["code"]: n["name"] for n in data.filter_options(key)["needs"]}

counts = {s: sum(c.status == s for c in cards) for s in STATUSES}
req_by_cluster = data.requirements_by_cluster(key)
c1, c2, c3, c4 = st.columns(4)
kpi(c1, len(cards), "Insights stored")
kpi(c2, counts[STATUS_TEMPLATE], "TEMPLATE", "from VFC classifications - not AI-generated")
kpi(c3, counts[STATUS_AI_ENRICHED], "AI_ENRICHED", "use actual LLM extraction")
kpi(c4, counts[STATUS_PENDING], "PENDING", "need further analysis")

if counts[STATUS_AI_ENRICHED] == 0:
    st.info("No real LLM extraction has run yet. Every insight below is a TEMPLATE: its text is assembled from the "
            "original BMW VFC classifications and database counts. Nothing here is an AI-generated finding.")
with st.expander("What the statuses mean"):
    for s in STATUSES:
        st.markdown(badge(s) + f'<span class="pm-note">{esc(data.STATUS_HELP[s])}</span>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Filters")
    sel_needs = st.multiselect("Customer-need category", sorted({c.need_code for c in cards}),
                               format_func=lambda c: need_names.get(c, c), key="ins_need")
    sel_status = st.multiselect("Generation status", STATUSES, key="ins_status")
    text_q = st.text_input("Search title / VFC cluster", key="ins_q")
    review_only = st.toggle("Only clusters requiring review", key="ins_review")

shown = [c for c in cards
         if (not sel_needs or c.need_code in sel_needs) and (not sel_status or c.status in sel_status)
         and (not review_only or c.cluster_requires_review)
         and (not text_q or text_q.lower() in (c.title + " " + c.cluster_label).lower())]
if not shown:
    st.info("No insights match these filters.")
    st.stop()

pages = max(1, -(-len(shown) // PAGE_SIZE))
st.session_state["ins_page"] = min(st.session_state.get("ins_page", 1), pages)
pc, pi = st.columns([1, 3])
page = pc.number_input("Page", 1, pages, key="ins_page")
pi.caption(f"{len(shown)} insights · sorted by unique comments · page {page} of {pages}")

for card in shown[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
    with st.container(border=True):
        cluster_name = "(empty VFC cell)" if card.cluster_label == MISSING_VFC else card.cluster_label
        st.markdown(f"#### {esc(card.title)}")
        st.markdown(
            badge(card.status)
            + chip(f"Need: {need_names.get(card.need_code, card.need_code)}")
            + chip(f"{'VFC cluster' if card.cluster_level == 'vfc_category' else card.cluster_level}: {cluster_name}")
            + chip(f"{card.unique_feedback_count:,} unique comments")
            + chip(f"{card.classification_count:,} classification rows")
            + (badge("REVIEW", "⚠ cluster mapping requires review") if card.cluster_requires_review else ""),
            unsafe_allow_html=True)
        source_note = ("Text assembled by template from VFC classifications." if card.status == STATUS_TEMPLATE
                       else "Text uses validated LLM extraction results." if card.status == STATUS_AI_ENRICHED
                       else "Insight needs further analysis before use.")
        st.caption(f"{source_note} Insight {card.insight_id} · cluster {card.cluster_id}")
        if card.status == STATUS_TEMPLATE and card.customer_problem.startswith("Original BMW VFC"):
            # Template text lists original labels (often praise) - it is not a problem statement.
            st.markdown(f"**Original VFC level-3 labels** — {esc(card.customer_problem.split(': ', 1)[-1])}")
        else:
            st.markdown(f"**Customer problem** — {esc(card.customer_problem)}")
        st.markdown(f"**Customer need** — {esc(card.customer_need)}")
        st.markdown(f"**Evidence summary** — {esc(card.evidence_summary)}")
        if card.validation_errors:
            st.warning("Validation notes: " + "; ".join(card.validation_errors[:5]))

        e1, e2 = st.columns(2)
        with e1.expander(f"Supporting evidence IDs ({len(card.internal_evidence_ids) + len(card.public_evidence_ids) + len(card.study_evidence_ids) + len(card.contextual_evidence_ids)})"):
            ids = card.internal_evidence_ids
            st.markdown(badge("DIRECT") + f" {len(ids)} feedback IDs", unsafe_allow_html=True)
            st.code(", ".join(ids[:40]) + (f", … (+{len(ids) - 40} more)" if len(ids) > 40 else ""), language=None)
            st.markdown(badge("SUPPORTING") + " public: " + esc(", ".join(card.public_evidence_ids) or "none"),
                        unsafe_allow_html=True)
            st.markdown(badge("SUPPORTING") + f" studies ({len(card.study_evidence_ids)}): "
                        + esc("; ".join(card.study_evidence_ids) or "none"), unsafe_allow_html=True)
            st.markdown(badge("CONTEXTUAL") + " competitor: " + esc(", ".join(card.contextual_evidence_ids) or "none"),
                        unsafe_allow_html=True)
        with e2.expander(f"Evidence limitations ({len(card.limitations)})"):
            for lim in card.limitations:
                st.markdown(f"- {esc(lim)}")
        v1, v2, v3 = st.columns([1, 1, 2])
        if v1.button("View Supporting Evidence", key=f"view_{card.insight_id}"):
            st.session_state["explorer_target_cluster"] = card.cluster_id
            st.switch_page("ui/pages/evidence_explorer.py")
        if v2.button("Draft Requirement", key=f"req_{card.insight_id}"):
            st.session_state["req_target"] = {"insight_id": card.insight_id}
            st.switch_page("ui/pages/vehicle_requirements.py")
        saved_reqs = req_by_cluster.get(card.cluster_id, [])
        if saved_reqs and v3.button(f"Prioritize saved requirement ({len(saved_reqs)})", key=f"prio_{card.insight_id}"):
            st.session_state["prio_requirement"] = saved_reqs[0]
            st.switch_page("ui/pages/agent_prioritization.py")
