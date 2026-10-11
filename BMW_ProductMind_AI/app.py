"""BMW ProductMind AI - guided product-management workflow.

    streamlit run app.py

Reads the existing SQLite database through the existing services. Makes no
LLM calls. For the command-line status view use `python -m scripts.status`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.theme import apply_theme  # noqa: E402

st.set_page_config(page_title="ProductMind AI · G60 Workflow", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()

PAGES = {
    "discover": st.Page("ui/pages/discover.py", title="DISCOVER", default=True),
    "define": st.Page("ui/pages/define.py", title="DEFINE"),
    "prioritize": st.Page("ui/pages/prioritize.py", title="PRIORITIZE"),
    "decide": st.Page("ui/pages/decide.py", title="DECIDE"),
    "intelligence": st.Page("ui/pages/customer_intelligence.py", title="Customer Intelligence"),
    "explorer": st.Page("ui/pages/evidence_explorer.py", title="Evidence Explorer"),
    "insights": st.Page("ui/pages/customer_insights.py", title="Customer Insights"),
    "requirements": st.Page("ui/pages/vehicle_requirements.py", title="Vehicle Requirements"),
    "prioritization": st.Page("ui/pages/agent_prioritization.py", title="Agent Prioritization"),
    "decision": st.Page("ui/pages/decision_intelligence.py", title="Decision Intelligence"),
    "quality": st.Page("ui/pages/data_quality.py", title="Data Quality"),
    "sources": st.Page("ui/pages/source_management.py", title="Source Management"),
    "settings": st.Page("ui/pages/application_settings.py", title="Application Settings"),
}
with st.sidebar:
    st.markdown("**ProductMind AI**  \n<span style='color:#B6C2D9;font-size:0.85rem'>DISCOVER → DEFINE → PRIORITIZE → DECIDE</span>",
                unsafe_allow_html=True)
st.navigation({
    "Workflow": [PAGES["discover"], PAGES["define"], PAGES["prioritize"], PAGES["decide"]],
    "Advanced": [PAGES["quality"], PAGES["sources"], PAGES["explorer"], PAGES["intelligence"], PAGES["insights"],
                 PAGES["requirements"], PAGES["prioritization"], PAGES["decision"], PAGES["settings"]],
}).run()
