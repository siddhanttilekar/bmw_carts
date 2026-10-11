"""Advanced application settings view."""
from __future__ import annotations

import os

import streamlit as st

import pandas as pd

from core.config import get_llm_settings, get_settings
from services import ai_feedback_analysis as ai
from services import custom_metrics as cmx
from services import llm_budget as budget
from ui import data
from ui.theme import apply_theme, chip, header

apply_theme()
header("Application Settings", "Runtime paths and feature gates for the local prototype.")

settings = get_settings()
llm = get_llm_settings()

st.markdown(chip("local only") + chip("no authentication") + chip("external LLM disabled by default"),
            unsafe_allow_html=True)
st.write("Project root:", str(settings.project_root))
st.write("Data directory:", str(settings.data_dir))
st.write("Database:", str(settings.database_path))
st.write("Reports directory:", str(settings.reports_dir))

st.markdown("### AI Feature Gates")
st.write("External LLM calls authorised:", llm.external_calls_authorized)
st.write("Model fallback enabled:", llm.use_fallbacks)
st.write("Configured model:", llm.model, "· model version label:", llm.model_version or llm.model)
st.write("Extraction prompt:", ai.OBSERVATION_PROMPT_VERSION, "· taxonomy:", ai.TAXONOMY_VERSION,
         "· observation version:", ai.current_observation_version_id(llm))
st.caption("Set environment variables such as PRODUCTMIND_ALLOW_EXTERNAL_LLM only outside this UI. "
           "API keys are never displayed or stored by the application.")

st.markdown("### LLM Budget")
from services import argue_chat  # noqa: E402

with data.write_session() as s:
    b = budget.budget_status(s, llm)
    metrics = cmx.list_custom_metrics(s)
    allowance = argue_chat.allowance_status(s, llm)
st.write(f"API provider: {llm.provider} · configured model: {llm.model}")
st.write(f"Application budget ${b.budget_usd:,.2f} · estimated spend ${b.spent_usd:,.4f} · remaining "
         f"${b.remaining_usd:,.4f} · {b.api_calls} API calls ({b.retry_or_failed_calls} retries/failed, "
         f"${b.retry_or_failed_cost_usd:,.4f}).")
st.caption("This is an application budget computed from logged token usage at list prices, not a provider billing cap.")
st.markdown("#### BMW_CAST Bot debate allowance")
if allowance["authorized"]:
    st.write(f"Authorised by the deployment owner: ${allowance['allowance_usd']:,.2f} · used ${allowance['spent_usd']:,.4f}"
             f" · available ${allowance['remaining_usd']:,.4f} (never more than the remaining application budget).")
else:
    st.write("Not authorised. Users must approve paid replies once per session.")
st.caption("Only the deployment owner can set or change the allowance, through the PRODUCTMIND_DEBATE_ALLOWANCE_USD "
           "environment variable before starting the app. It is not editable here.")

st.markdown("### Documented Custom Metrics")
st.dataframe(pd.DataFrame(metrics), hide_index=True, width="stretch")
st.caption("Only code-implemented metrics exist. An administrator approves one from the terminal: "
           "`python -m scripts.customer_intelligence approve-metric METRIC_ID --admin ID --note TEXT`.")

with st.expander("Relevant environment variables"):
    names = [
        "PRODUCTMIND_DATA_DIR",
        "PRODUCTMIND_DB_PATH",
        "PRODUCTMIND_REPORTS_DIR",
        "PRODUCTMIND_ALLOW_EXTERNAL_LLM",
        "PRODUCTMIND_LLM_FALLBACKS",
        "PRODUCTMIND_LLM_MODEL",
        "PRODUCTMIND_LLM_MODEL_VERSION",
        "PRODUCTMIND_LLM_BUDGET_USD",
        "PRODUCTMIND_MAX_API_SPEND_USD",
        "PRODUCTMIND_LLM_PROVIDER",
        "PRODUCTMIND_DEBATE_ALLOWANCE_USD",
    ]
    for name in names:
        st.write(name, "=", os.getenv(name, ""))
