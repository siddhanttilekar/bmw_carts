"""Main dashboard: what G60 customers say, with traceable evidence."""
from __future__ import annotations

import streamlit as st

from services import product_workflow as pw
from ui import data
from ui.evidence_ui import llm_settings
from ui.theme import apply_theme, esc, grouped_bar, kpi

apply_theme()
st.markdown('<div class="pm-hero"><div class="pm-tag">From Customer Feedback to Vehicle Decisions</div>'
            '<div class="pm-veh">Vehicle · BMW 5 Series G60</div></div>', unsafe_allow_html=True)
st.write("")

settings = llm_settings()
with data.write_session() as s:
    m = pw.dashboard_metrics(s, settings)
    problems = pw.top_topics(s, settings, "COMPLAINT", 5)
    requests = pw.top_topics(s, settings, "IMPROVEMENT_REQUEST", 5)
    praised = pw.top_topics(s, settings, "POSITIVE", 5)
    areas = pw.needs_by_domain(s, settings)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{m['unique_comments']:,}", "Unique G60 customer comments", "distinct original feedback IDs")
kpi(c2, f"{m['ai_processed_comments']:,}", "AI-processed comments",
    f"{m['observations']:,} validated observations (one comment can contain several)")
kpi(c3, f"{m['complaint_comments']:,}", "Comments with a complaint", "unique comments, not observations")
kpi(c4, f"{m['request_comments']:,}", "Comments with an improvement request", "unique comments, not observations")

st.write("")
if st.button("START PRODUCT PRIORITIZATION", type="primary", use_container_width=False):
    st.session_state["wiz_step"] = 1
    st.switch_page("ui/pages/product_prioritization.py")


def topic_column(col, title: str, subtitle: str, rows: list[dict], key: str) -> None:
    with col:
        st.markdown(f"#### {title}")
        st.caption(subtitle)
        if not rows:
            st.caption("No validated observations yet.")
        for i, r in enumerate(rows, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {esc(r['topic'])}**  \n<span class='pm-note'>{r['comments']:,} customers · "
                            f"{esc(r['domain'])}</span>", unsafe_allow_html=True)
                with st.expander("View evidence"):
                    for e in r["examples"]:
                        st.markdown(f"<div class='pm-quote'>“{esc(e['excerpt'])}”<small>Comment {esc(e['feedback_id'])} · "
                                    f"{esc(e['customer_need'])}</small></div>", unsafe_allow_html=True)


st.write("")
a, b, c = st.columns(3)
topic_column(a, "Top customer problems", "Vehicle systems most often named in complaints", problems, "p")
topic_column(b, "Top improvement requests", "What customers explicitly ask to change or add", requests, "r")
topic_column(c, "Most appreciated features", "What customers praise - worth protecting", praised, "a")

st.markdown("#### Customer needs by vehicle area")
st.caption("Distinct customers per area with at least one complaint or improvement request (AI-validated observations).")
areas = sorted(areas, key=lambda r: -(r["COMPLAINT"] + r["IMPROVEMENT_REQUEST"]))
st.plotly_chart(grouped_bar([r["domain"] for r in areas], [r["COMPLAINT"] for r in areas],
                            [r["IMPROVEMENT_REQUEST"] for r in areas], a_name="Complaints",
                            b_name="Improvement requests"), width="stretch", config={"displayModeBar": False})
with st.expander("Table view"):
    st.dataframe([{"Vehicle area": r["domain"], "Complaint comments": r["COMPLAINT"],
                   "Request comments": r["IMPROVEMENT_REQUEST"], "Positive comments": r["POSITIVE"]} for r in areas],
                 hide_index=True, width="stretch")
st.caption("Counts are distinct comments from the supplied G60 export (mostly US customers). Frequency is not severity. "
           "Technical details are under Advanced.")
