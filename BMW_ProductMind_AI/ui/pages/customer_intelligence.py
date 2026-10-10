"""Page 1 - Customer Intelligence overview (all numbers computed from SQLite)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.dashboard_queries import MISSING_TYPE_LABEL
from ui import data
from ui.filters import sidebar_filters
from ui.theme import apply_theme, bar_chart, header, kpi

apply_theme()
header("Customer Intelligence",
       "BMW 5 Series (G60) customer feedback. Comment counts use COUNT(DISTINCT feedback ID); "
       "a comment with several VFC classifications is counted once.")

f = sidebar_filters("ci")
o = data.overview(data.db_key(), f.model_dump_json())

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{o.unique_feedback:,}", "Unique G60 feedback comments", "distinct feedback IDs matching the filters")
kpi(c2, f"{o.classification_rows:,}", "Classification records", "original VFC rows; not customers")
kpi(c3, f"{o.need_categories_with_feedback} / {o.need_categories_total}", "Customer-need categories",
    "with matching feedback / curated total")
kpi(c4, f"{o.vfc_clusters_with_feedback} / {o.vfc_clusters_total}", "VFC clusters",
    "with matching feedback / total")

if o.unique_feedback == 0:
    st.info("No feedback matches this filter combination. Remove a filter to widen the selection.")
    st.stop()

st.write("")
left, right = st.columns([1.15, 1])

with left:
    st.markdown("#### Feedback by customer-need category")
    include_other = st.toggle("Include 'Other / Unclassified'", value=False, key="ci_other",
                              help="Comments whose only labels are BMW 'no_class_found' or an empty VFC cell.")
    needs = [n for n in o.by_need if include_other or n.need_code != "OTHER"]
    if needs:
        st.plotly_chart(bar_chart([n.need_name for n in needs], [n.unique_feedback_count for n in needs],
                                  value_name="Unique comments",
                                  extra=[f"<br>Classification rows: {n.classification_count:,}" for n in needs]),
                        width="stretch", config={"displayModeBar": False})
    st.caption("A comment counts in every category its VFC labels map to, so category totals can exceed the comment total.")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame([n.model_dump() for n in o.by_need]), hide_index=True, width="stretch")

with right:
    st.markdown("#### Most frequently discussed topics")
    st.caption("Original BMW VFC level-2 categories, ranked by unique comments. Excludes unclassified rows.")
    if o.top_vfc:
        st.plotly_chart(bar_chart([c.label for c in o.top_vfc], [c.feedback_ids for c in o.top_vfc],
                                  value_name="Unique comments",
                                  extra=[f"<br>Classification rows: {c.classification_rows:,}" for c in o.top_vfc]),
                        width="stretch", config={"displayModeBar": False})
    else:
        st.info("No classified topics for this selection.")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame([{"vfc_category": c.label, "unique_comments": c.feedback_ids,
                                    "classification_rows": c.classification_rows} for c in o.top_vfc]),
                     hide_index=True, width="stretch")

m, e = st.columns([1.15, 1])
with m:
    st.markdown("#### Feedback by market")
    st.plotly_chart(bar_chart([d.value or "(missing)" for d in o.by_market], [d.feedback_ids for d in o.by_market],
                              value_name="Unique comments", value_labels=True), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Market = country code supplied with each comment. Small markets have few comments.")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame([{"market": d.value, "unique_comments": d.feedback_ids,
                                    "classification_rows": d.classification_rows} for d in o.by_market]),
                     hide_index=True, width="stretch")
with e:
    st.markdown("#### Feedback by engine type")
    st.plotly_chart(bar_chart([d.value or "(missing)" for d in o.by_engine], [d.feedback_ids for d in o.by_engine],
                              value_name="Unique comments", height=200, value_labels=True), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Engine type as supplied: ICE, BEVE (battery-electric), PHEV.")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame([{"engine_type": d.value, "unique_comments": d.feedback_ids,
                                    "classification_rows": d.classification_rows} for d in o.by_engine]),
                     hide_index=True, width="stretch")

with st.expander("Original Feedback Type labels (BMW classification-row labels, not sentiment)"):
    st.caption("Shown as counts only. Feedback Type is attached to each classification row, a comment can carry "
               "conflicting labels, and channels use the labels differently - so no positive/negative shares are derived.")
    st.dataframe(pd.DataFrame([{"feedback_type": t.feedback_type or MISSING_TYPE_LABEL,
                                "classification_rows": t.classification_rows, "unique_comments": t.feedback_ids}
                               for t in o.feedback_types]), hide_index=True, width="stretch")

st.markdown("#### From insight to requirement")
needs_with_data = [n for n in o.by_need if n.need_code != "OTHER"]
if needs_with_data:
    a, b = st.columns([3, 1])
    pick = a.selectbox("Customer-need category", [n.need_code for n in needs_with_data], key="ci_to_req",
                       format_func=lambda c: next(f"{n.need_name} · {n.unique_feedback_count} comments" for n in needs_with_data
                                                  if n.need_code == c))
    b.write("")
    if b.button("Open in Vehicle Requirements", width="stretch"):
        st.session_state["req_target"] = {"cluster_id": f"NEED-{pick}"}
        st.switch_page("ui/pages/vehicle_requirements.py")
