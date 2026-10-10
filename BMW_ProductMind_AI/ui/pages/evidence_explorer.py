"""Page 2 - Evidence Explorer: original comments plus linked supporting and
contextual evidence, each with its source reference."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.need_taxonomy import MISSING_VFC
from ui import data
from ui.filters import sidebar_filters
from ui.theme import apply_theme, badge, chip, esc, header

PAGE_SIZE = 20
ALL = "__all__"

apply_theme()
header("Evidence Explorer",
       "Every item shows its evidence tier and original source reference. Direct = internal BMW feedback; "
       "Supporting = public BMW feedback and BMW studies; Contextual = competitor observations.")

key = data.db_key()
opts = data.filter_options(key)
need_names = {n["code"]: n["name"] for n in opts["needs"]}
clusters = data.cluster_options(key)
by_id = {c.cluster_id: c for c in clusters}

# Deep link from the Customer Insights page: select the cluster's need + cluster.
target = st.session_state.pop("explorer_target_cluster", None)
if target and target in by_id:
    st.session_state["ex_need"] = by_id[target].need_code
    st.session_state["ex_cluster"] = target
    st.session_state["ex_page"] = 1

sel1, sel2 = st.columns([1, 1.4])
need_code = sel1.selectbox("Customer-need category", list(need_names), format_func=need_names.get, key="ex_need")
need_clusters = [c for c in clusters if c.need_code == need_code and c.cluster_level != "need_category"]
cluster_ids = [ALL] + [c.cluster_id for c in need_clusters]
if st.session_state.get("ex_cluster") not in cluster_ids:
    st.session_state["ex_cluster"] = ALL


def _cluster_label(cid: str) -> str:
    if cid == ALL:
        return "All VFC categories in this need"
    c = by_id[cid]
    kind = "VFC" if c.cluster_level == "vfc_category" else "AI need"
    label = "(empty VFC cell)" if c.label == MISSING_VFC else c.label
    return f"{kind}: {label} · {c.unique_feedback_count} comments" + (" · ⚠ review" if c.requires_review else "")


cluster_id = sel2.selectbox("Cluster", cluster_ids, format_func=_cluster_label, key="ex_cluster")
cluster = by_id.get(cluster_id)

f = sidebar_filters("ex", include_need=False, include_vfc=False)
f.need_codes = [need_code]
if cluster and cluster.cluster_level == "vfc_category":
    f.vfc_names = [cluster.vfc_level2_name]

if need_code != "OTHER" and st.button("Draft requirement from this selection", key="ex_to_req"):
    st.session_state["req_target"] = {"cluster_id": cluster_id if cluster else f"NEED-{need_code}"}
    st.switch_page("ui/pages/vehicle_requirements.py")

mapping = data.mapping_status(key)
if cluster and cluster.requires_review:
    note = mapping.get(cluster.vfc_level2_name or "", (None, None, None))[2]
    st.warning(f"⚠ This cluster's VFC → need mapping requires review. {note or ''}")

items = data.related_evidence(key, need_code)
supporting = [i for i in items if i.tier == "SUPPORTING"]
contextual = [i for i in items if i.tier == "CONTEXTUAL"]

search = st.text_input("Search original comment text", key="ex_search",
                       placeholder="Literal, case-insensitive match, e.g. touchscreen")
extracted_ids: list[str] | None = None
if cluster and cluster.cluster_level == "extracted_need":
    extracted_ids = data.cluster_member_ids(key, need_code, cluster_id)
    st.caption("AI-need cluster: showing its linked comments; sidebar filters and search do not apply.")

if extracted_ids is None:
    total, _ = data.feedback_page(key, f.model_dump_json(), search, 1, 0)
else:
    total = len(extracted_ids)

tab_d, tab_s, tab_c = st.tabs([f"Direct · {total:,} comments", f"Supporting · {len(supporting)}",
                               f"Contextual · {len(contextual)}"])


def render_comment(r) -> None:
    preview = " ".join(r.customer_feedback.split())[:110]
    with st.expander(f"{r.original_feedback_id} · {r.country or '?'} · {r.engine_type or '?'} — {preview}"):
        st.markdown(badge("DIRECT") + chip(f"Source channel: {r.feedback_source or 'n/a'}") + chip(f"Market: {r.country}")
                    + chip(f"Engine: {r.engine_type}"), unsafe_allow_html=True)
        st.markdown(f'<div class="pm-text">{esc(r.customer_feedback)}</div>', unsafe_allow_html=True)
        st.markdown("**Original VFC classifications**")
        for c in r.classifications:
            name = c.vfc_level2_name if c.vfc_level2_name is not None else MISSING_VFC
            need, status, reason = mapping.get(name, (None, None, None))
            html = chip(c.vfc_level2_3_name or "(empty VFC cell)") + chip(f"Feedback Type: {c.feedback_type or 'not provided'}")
            html += chip(f"→ {need_names.get(need, need)}") if need else ""
            html += badge("REVIEW", f"⚠ mapping requires review: {reason}") if status == "review_required" else ""
            html += f'<span class="pm-src">{esc(c.source.source_file)} · {esc(c.source.source_sheet)} · row {c.source.source_row}</span>'
            st.markdown(html, unsafe_allow_html=True)
        st.markdown(f'<span class="pm-src">Feedback record {esc(r.source.record_id)} · first row {r.source.source_row} · '
                    f'{esc(r.source.data_origin)}</span>', unsafe_allow_html=True)


with tab_d:
    if total == 0:
        st.info("No comments match this selection. Clear the search or widen the sidebar filters.")
    else:
        pages = max(1, -(-total // PAGE_SIZE))
        st.session_state["ex_page"] = min(st.session_state.get("ex_page", 1), pages)
        cpage, cinfo = st.columns([1, 3])
        page = cpage.number_input("Page", 1, pages, key="ex_page")
        cinfo.caption(f"{total:,} distinct comments · page {page} of {pages} · ordered by feedback ID")
        if extracted_ids is None:
            _, records = data.feedback_page(key, f.model_dump_json(), search, PAGE_SIZE, (page - 1) * PAGE_SIZE)
        else:
            records = list(data.resolve_feedback(key, tuple(extracted_ids[(page - 1) * PAGE_SIZE: page * PAGE_SIZE])))
        for r in records:
            render_comment(r)


def render_item(i) -> None:
    with st.container(border=True):
        st.markdown(badge(i.tier) + chip(i.kind.replace("_", " ")) + (badge("REVIEW") if i.requires_review else ""),
                    unsafe_allow_html=True)
        st.markdown(f"**{esc(i.title)}**")
        if i.requires_review:
            st.warning(f"⚠ Mapping to this need requires review. {i.mapping_note or ''}")
        elif i.mapping_note:
            st.caption(i.mapping_note)
        if i.detail:
            st.markdown(f'<div class="pm-text">{esc(i.detail)}</div>', unsafe_allow_html=True)
        if i.study_values:
            st.dataframe(pd.DataFrame([{"market": v.survey_market, "vehicle": v.vehicle_label, "measure": v.measure,
                                        "value": round(v.value, 3), "value_type": v.value_type,
                                        "source": f"{v.source.source_sheet} row {v.source.source_row}"}
                                       for v in i.study_values]), hide_index=True, width="stretch")
        if i.source_url:
            st.markdown(f"Source URL: [{esc(i.source_url)}]({i.source_url})")
        st.markdown(f'<span class="pm-src">{esc(i.source.source_file)} · {esc(i.source.source_sheet)} · '
                    f'row {i.source.source_row} · {esc(i.source.data_origin)}</span>', unsafe_allow_html=True)


with tab_s:
    st.caption("Linked to the customer-need category (not to individual comments). Public observations are curated "
               "and not representative; survey values have no sample sizes.")
    if not supporting:
        st.info("No supporting evidence is mapped to this need.")
    for i in supporting:
        render_item(i)

with tab_c:
    st.caption("Competitor observations are market context only. They are not evidence of BMW customer dissatisfaction.")
    if not contextual:
        st.info("No competitor observations are mapped to this need.")
    for i in contextual:
        render_item(i)
