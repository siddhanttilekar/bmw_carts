"""Visual system for the dashboard: dark navy surfaces, white type, one
restrained blue for data marks (validated: #3987E5 on #111C33 passes the
lightness, chroma and >=3:1 contrast checks). Identity badges always carry a
text label, never colour alone."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

BG = "#0B1426"
SURFACE = "#111C33"
BORDER = "#1E2C4A"
TEXT = "#FFFFFF"
TEXT_2 = "#B6C2D9"
MUTED = "#7F8BA6"
ACCENT = "#3987E5"
GRID = "#1E2C4A"

# Evidence tiers / statuses: hue + text label (never colour alone).
BADGES = {
    "DIRECT": ("#3987E5", "Direct · internal BMW feedback"),
    "SUPPORTING": ("#199E70", "Supporting · public BMW feedback / BMW study"),
    "CONTEXTUAL": ("#9085E9", "Contextual · competitor observation"),
    "TEMPLATE": ("#7F8BA6", "TEMPLATE · from VFC classifications"),
    "AI_ENRICHED": ("#3987E5", "AI_ENRICHED · uses LLM extraction"),
    "PENDING": ("#C98500", "PENDING · needs further analysis"),
    "REVIEW": ("#C98500", "⚠ Requires review"),
}

CSS = f"""
<style>
  .stApp {{ background: {BG}; }}
  section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .pm-sub {{ color: {TEXT_2}; margin-top: -0.6rem; margin-bottom: 1.2rem; }}
  .pm-kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; height: 100%; }}
  .pm-kpi .v {{ font-size: 2rem; font-weight: 650; color: {TEXT}; line-height: 1.1; }}
  .pm-kpi .l {{ color: {TEXT_2}; font-size: 0.86rem; margin-top: 4px; }}
  .pm-kpi .n {{ color: {MUTED}; font-size: 0.75rem; margin-top: 6px; }}
  .pm-badge {{ display: inline-block; border: 1px solid; border-radius: 999px; padding: 1px 10px; font-size: 0.74rem;
               margin: 0 6px 4px 0; white-space: nowrap; }}
  .pm-chip {{ display: inline-block; background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 1px 8px;
              font-size: 0.78rem; color: {TEXT_2}; margin: 0 6px 4px 0; }}
  .pm-src {{ color: {MUTED}; font-size: 0.76rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .pm-text {{ color: {TEXT}; font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; }}
  .pm-note {{ color: {TEXT_2}; font-size: 0.82rem; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def badge(kind: str, text: str | None = None) -> str:
    color, label = BADGES[kind]
    return f'<span class="pm-badge" style="color:{color};border-color:{color}">{esc(text or label)}</span>'


def chip(text: str) -> str:
    return f'<span class="pm-chip">{esc(text)}</span>'


def header(title: str, subtitle: str) -> None:
    st.markdown(f"## {esc(title)}")
    st.markdown(f'<div class="pm-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)


def kpi(col, value, label: str, note: str = "") -> None:
    col.markdown(f'<div class="pm-kpi"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div>'
                 f'<div class="n">{esc(note)}</div></div>', unsafe_allow_html=True)


def bar_chart(labels: list[str], values: list[int], *, value_name: str, extra: list[str] | None = None,
              height: int | None = None, value_labels: bool = False) -> go.Figure:
    """Horizontal single-series bar chart (largest at top), one hue, thin bars,
    recessive grid, hover tooltip with the exact value."""
    labels, values = list(reversed(labels)), list(reversed(values))
    custom = list(reversed(extra)) if extra else [""] * len(labels)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=ACCENT, cornerradius=4), width=0.62,
        customdata=custom,
        text=[f"{v:,}" for v in values] if value_labels else None, textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT_2, size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{value_name}: %{{x:,}}%{{customdata}}<extra></extra>",
    ))
    fig.update_layout(
        height=height or max(220, 34 * len(labels) + 60), margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2, size=12),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickformat=",d", title=None),
        yaxis=dict(showgrid=False, title=None, automargin=True, tickfont=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT)), bargap=0.3,
    )
    return fig
