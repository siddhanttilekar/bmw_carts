"""Sidebar filters shared by the Customer Intelligence and Evidence Explorer pages."""
from __future__ import annotations

import streamlit as st

from services.dashboard_queries import MISSING_TYPE_LABEL, DashboardFilter
from services.need_taxonomy import MISSING_VFC
from ui import data


def sidebar_filters(prefix: str, *, include_need: bool = True, include_vfc: bool = True) -> DashboardFilter:
    """All filters combine (AND). Empty selection = no restriction."""
    opts = data.filter_options(data.db_key())
    need_names = {n["code"]: n["name"] for n in opts["needs"]}
    with st.sidebar:
        st.markdown("### Filters")
        st.caption("All filters combine. Vehicle: BMW 5 Series G60 only.")
        countries = st.multiselect("Market", opts["countries"], key=f"{prefix}_market")
        engines = st.multiselect("Engine type", opts["engine_types"], key=f"{prefix}_engine",
                                 help="As supplied by BMW: ICE, BEVE (battery-electric), PHEV.")
        needs = (st.multiselect("Customer-need category", list(need_names), format_func=need_names.get,
                                key=f"{prefix}_need") if include_need else [])
        vfc = (st.multiselect("Original VFC category", opts["vfc_names"], key=f"{prefix}_vfc",
                              format_func=lambda v: "(empty VFC cell)" if v == MISSING_VFC else v) if include_vfc else [])
        type_labels = [t if t is not None else MISSING_TYPE_LABEL for t in opts["feedback_types"]]
        types = st.multiselect("Feedback Type (original BMW label)", type_labels, key=f"{prefix}_type",
                               help="BMW's label per classification row. Not a sentiment measure.")
        if st.button("Reset filters", key=f"{prefix}_reset", width="stretch"):
            for k in ("market", "engine", "need", "vfc", "type"):
                st.session_state.pop(f"{prefix}_{k}", None)
            st.rerun()
    return DashboardFilter(
        countries=countries or None, engine_types=engines or None, need_codes=needs or None, vfc_names=vfc or None,
        feedback_types=[None if t == MISSING_TYPE_LABEL else t for t in types] or None,
    )
