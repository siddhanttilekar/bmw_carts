"""BMW ProductMind AI - Dashboard and continuous Product Prioritization.

    streamlit run app.py

Reads the existing SQLite database through the existing services. Paid LLM
calls happen only after explicit approval (AI analysis, ARGUE). For the
command-line status view use `python -m scripts.status`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.theme import apply_theme  # noqa: E402

st.set_page_config(page_title="BMW ProductMind AI", layout="wide", initial_sidebar_state="collapsed")
apply_theme()

PAGES = {
    "dashboard": st.Page("ui/pages/dashboard.py", title="Dashboard", default=True),
    "wizard": st.Page("ui/pages/product_prioritization.py", title="Product Prioritization"),
    # Advanced (all earlier screens stay available)
    "discover": st.Page("ui/pages/discover.py", title="Discover (step view)"),
    "define": st.Page("ui/pages/define.py", title="Define (step view)"),
    "prioritize": st.Page("ui/pages/prioritize.py", title="Prioritize (step view)"),
    "decide": st.Page("ui/pages/decide.py", title="Decide (step view)"),
    "intelligence": st.Page("ui/pages/customer_intelligence.py", title="Customer Intelligence"),
    "explorer": st.Page("ui/pages/evidence_explorer.py", title="Evidence Explorer"),
    "insights": st.Page("ui/pages/customer_insights.py", title="Customer Insights"),
    "requirements": st.Page("ui/pages/vehicle_requirements.py", title="Vehicle Requirements"),
    "prioritization": st.Page("ui/pages/agent_prioritization.py", title="Agent Configuration & History"),
    "decision": st.Page("ui/pages/decision_intelligence.py", title="Decision Intelligence"),
    "quality": st.Page("ui/pages/data_quality.py", title="Data Quality"),
    "sources": st.Page("ui/pages/source_management.py", title="Source Management"),
    "settings": st.Page("ui/pages/application_settings.py", title="Application Settings"),
}
# Top navigation: two primary destinations; everything else in the collapsed "Advanced" menu.
st.navigation({
    "": [PAGES["dashboard"], PAGES["wizard"]],
    "Advanced": [PAGES["intelligence"], PAGES["explorer"], PAGES["insights"], PAGES["requirements"],
                 PAGES["prioritization"], PAGES["decision"], PAGES["discover"], PAGES["define"],
                 PAGES["prioritize"], PAGES["decide"], PAGES["quality"], PAGES["sources"], PAGES["settings"]],
}, position="top").run()
