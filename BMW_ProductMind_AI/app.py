"""BMW ProductMind AI - Customer Intelligence Dashboard (Part 3).

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

st.set_page_config(page_title="ProductMind AI · G60 Customer Intelligence", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()

PAGES = {
    "intelligence": st.Page("ui/pages/customer_intelligence.py", title="Customer Intelligence", default=True),
    "explorer": st.Page("ui/pages/evidence_explorer.py", title="Evidence Explorer"),
    "insights": st.Page("ui/pages/customer_insights.py", title="Customer Insights"),
    "requirements": st.Page("ui/pages/vehicle_requirements.py", title="Vehicle Requirements"),
    "prioritization": st.Page("ui/pages/agent_prioritization.py", title="Agent Prioritization"),
    "quality": st.Page("ui/pages/data_quality.py", title="Data Quality"),
}
with st.sidebar:
    st.markdown("**ProductMind AI**  \n<span style='color:#B6C2D9;font-size:0.85rem'>5 Series (G60) customer intelligence</span>",
                unsafe_allow_html=True)
PAGES["decision"] = st.Page("ui/pages/decision_intelligence.py", title="Decision Intelligence")
st.navigation({
    "Customer Intelligence": [PAGES["intelligence"], PAGES["explorer"], PAGES["insights"]],
    "Requirements & Prioritization": [PAGES["requirements"], PAGES["prioritization"]],
    "Decision Intelligence": [PAGES["decision"]],
    "Transparency": [PAGES["quality"]],
}).run()
