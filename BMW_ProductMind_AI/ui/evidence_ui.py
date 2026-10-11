"""Shared Streamlit fragments for the 2.0 evidence-first workflow."""
from __future__ import annotations

import streamlit as st

from core.config import get_llm_settings
from services import llm_budget as budget
from services.agent_explanations import calculation_lines, evidence_lines
from services.evidence_extraction import has_credentials
from ui import data
from ui.theme import kpi


def llm_settings():
    return get_llm_settings()


def budget_kpis(cols) -> budget.BudgetStatus:
    with data.write_session() as s:
        b = budget.budget_status(s, llm_settings())
    kpi(cols[0], f"${b.spent_usd:,.2f}", "LLM spend (estimated)", f"{b.api_calls} API calls logged")
    kpi(cols[1], f"${b.remaining_usd:,.2f}", "Remaining budget", f"of ${b.budget_usd:,.2f} application budget")
    return b


def paid_call_blocker() -> str | None:
    """Reason why a paid call cannot run now (None if it can)."""
    st_ = llm_settings()
    if not st_.external_calls_authorized:
        return "External LLM processing is not enabled (PRODUCTMIND_ALLOW_EXTERNAL_LLM=1, set outside the UI)."
    if not has_credentials(st_):
        key = "OPENAI_API_KEY" if st_.provider == "openai" else "ANTHROPIC_API_KEY"
        return f"No API key for provider '{st_.provider}' is configured ({key}, set outside the UI)."
    return None


def paid_client():
    from services.evidence_extraction import default_client

    return default_client(llm_settings())


def show_calculation(result: dict) -> None:
    """The calculation behind one stored agent score (real stored values)."""
    for line in calculation_lines(result):
        st.markdown(f"- {line}")
    for line in evidence_lines(result):
        st.caption(line)
