"""Page 4 - Data Quality & transparency."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import data
from ui.theme import apply_theme, esc, header, kpi

apply_theme()
header("Data Quality", "What was imported, how it was checked, and what the data cannot tell you.")

q = data.data_quality(data.db_key())

passed = sum(c["passed"] for c in q["integrity"])
audits_ok = sum(a["status"] == "ok" for a in q["audits"])
gaps = sum(q["provenance_gaps"].values())
c1, c2, c3, c4 = st.columns(4)
kpi(c1, len(q["documents"]), "Source documents", "registered with SHA-256")
kpi(c2, f"{passed}/{len(q['integrity'])}", "Integrity checks passing", "recomputed now")
kpi(c3, f"{audits_ok}/{len(q['audits'])}", "Raw-file count checks", "recorded at last ingestion")
kpi(c4, gaps, "Records missing provenance", "should be 0")

st.markdown("### Known limitations")
for i, lim in enumerate(q["limitations"], start=1):
    st.markdown(f"**{i}. {esc(lim.title)}** — {esc(lim.evidence)}")

st.markdown("### Imported datasets")
st.dataframe(pd.DataFrame(q["documents"]), hide_index=True, width="stretch")

st.markdown("### Record counts (live)")
st.dataframe(pd.DataFrame([{"table": k, "records": v} for k, v in q["table_counts"].items()]),
             hide_index=True, width="stretch")

st.markdown("### Data-quality checks")
st.dataframe(pd.DataFrame([{"check": c["check"], "result": "PASS" if c["passed"] else "FAIL", "detail": c["detail"]}
                           for c in q["integrity"]]), hide_index=True, width="stretch")
with st.expander("Raw-file recount vs. database (ingestion audit)"):
    st.dataframe(pd.DataFrame(q["audits"]), hide_index=True, width="stretch")

st.markdown("### Missing data (internal feedback)")
st.caption("Missing values are stored as NULL and never imputed. 'no_class_found' is BMW's own label.")
st.dataframe(pd.DataFrame(q["feedback_quality"]), hide_index=True, width="stretch")

r1, r2 = st.columns(2)
with r1:
    st.markdown(f"### Unreviewed category mappings ({len(q['review_mappings'])})")
    st.caption("Original VFC level-2 classes whose mapping to a customer-need category is marked review_required.")
    st.dataframe(pd.DataFrame(q["review_mappings"]), hide_index=True, width="stretch", height=360)
with r2:
    st.markdown(f"### Unreviewed cross-source links ({len(q['review_links'])})")
    st.caption("Supporting/contextual evidence links marked requires_review.")
    st.dataframe(pd.DataFrame(q["review_links"]), hide_index=True, width="stretch", height=360)
