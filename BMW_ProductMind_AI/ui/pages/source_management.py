"""Advanced source-management view."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import data
from ui.theme import apply_theme, header

apply_theme()
header("Source Management", "Canonical source files, import status and source provenance.")

quality = data.data_quality(data.db_key())
st.dataframe(pd.DataFrame(quality["documents"]), hide_index=True, width="stretch")

with st.expander("Import count checks"):
    st.dataframe(pd.DataFrame(quality["audits"]), hide_index=True, width="stretch")
