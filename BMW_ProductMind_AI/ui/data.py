"""Cached data access for the dashboard.

Every loader takes ``db_key`` = "<database path>|<mtime>", so cached results are
invalidated as soon as the database file changes, and also expire after
``TTL_SECONDS``. All data comes from the existing services (no SQL here).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.database import ensure_ai_schema, get_engine, session_scope
from services import dashboard_queries as dq
from services.cross_source_mapping import comments_for_need, evidence_for_need
from services.data_validation import feedback_quality, integrity_checks, provenance_gaps, table_counts
from services.evidence_retrieval import get_feedback_by_ids, search_feedback_text
from services.feedback_analysis import FeedbackFilter, filter_feedback
from services.insight_generator import STATUS_DESCRIPTIONS, all_insight_cards

TTL_SECONDS = 300


def db_path() -> Path:
    return get_settings().database_path.resolve()


def db_key() -> str:
    path = db_path()
    if not path.exists():
        st.error(f"Database not found at {path}. Run `python -m scripts.initialize_database` "
                 "and `python -m scripts.customer_intelligence setup` first.")
        st.stop()
    return f"{path}|{path.stat().st_mtime_ns}"


@st.cache_resource(show_spinner=False)
def _engine(path: str):
    engine = get_engine(Path(path))
    ensure_ai_schema(engine)  # additive only; no-op when up to date
    return engine


def _session(key: str):
    return session_scope(_engine(key.split("|")[0]))


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def filter_options(key: str) -> dict:
    with _session(key) as s:
        return dq.filter_options(s)


@st.cache_data(ttl=TTL_SECONDS, show_spinner="Computing from SQLite…")
def overview(key: str, f_json: str) -> dq.Overview:
    with _session(key) as s:
        return dq.overview(s, dq.DashboardFilter.model_validate_json(f_json))


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def cluster_options(key: str) -> list[dq.ClusterOption]:
    with _session(key) as s:
        return dq.cluster_options(s)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def mapping_status(key: str) -> dict:
    with _session(key) as s:
        return dq.vfc_mapping_status(s)


@st.cache_data(ttl=TTL_SECONDS, show_spinner="Loading feedback…")
def feedback_page(key: str, f_json: str, search: str, limit: int, offset: int):
    """(total matching comments, records) for the explorer."""
    with _session(key) as s:
        flt: FeedbackFilter = dq.to_feedback_filter(s, dq.DashboardFilter.model_validate_json(f_json))
        if search.strip():
            total, records = search_feedback_text(s, search, flt, limit=offset + limit)
            return total, records[offset:offset + limit]
        page = filter_feedback(s, flt, limit=limit, offset=offset)
        return page.total_feedback_ids, page.items


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def related_evidence(key: str, need_code: str) -> list[dq.EvidenceItem]:
    with _session(key) as s:
        return dq.related_evidence(s, need_code)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def direct_evidence(key: str, need_code: str):
    with _session(key) as s:
        return evidence_for_need(s, need_code, dq.VEHICLE)


@st.cache_data(ttl=TTL_SECONDS, show_spinner="Loading insights…")
def insight_cards(key: str):
    with _session(key) as s:
        return all_insight_cards(s, dq.VEHICLE)


STATUS_HELP = STATUS_DESCRIPTIONS


@st.cache_data(ttl=TTL_SECONDS, show_spinner="Running data-quality checks…")
def data_quality(key: str) -> dict:
    with _session(key) as s:
        audits = dq.audit_rows(s)
        return {
            "documents": dq.source_documents(s),
            "table_counts": table_counts(s),
            "audits": audits,
            "provenance_gaps": provenance_gaps(s),
            "integrity": integrity_checks(s, get_settings().data_dir, audits),
            "feedback_quality": feedback_quality(s),
            "review_mappings": dq.review_mappings(s),
            "review_links": dq.review_links(s),
            "limitations": dq.known_limitations(s),
        }


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def resolve_feedback(key: str, ids: tuple[str, ...]):
    with _session(key) as s:
        return get_feedback_by_ids(s, list(ids), dq.VEHICLE)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def need_comment_ids(key: str, need_code: str) -> list[str]:
    with _session(key) as s:
        return comments_for_need(s, need_code, dq.VEHICLE)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def cluster_member_ids(key: str, need_code: str, cluster_id: str) -> list[str]:
    """Original feedback IDs linked to a cluster (used for AI-need clusters)."""
    from services.need_clustering import clusters_for_need

    with _session(key) as s:
        return next((v.feedback_ids for v in clusters_for_need(s, need_code, dq.VEHICLE) if v.cluster_id == cluster_id), [])


@st.cache_data(ttl=TTL_SECONDS, show_spinner="Building evidence package from SQLite…")
def evidence_package(key: str, cluster_id: str):
    from services.requirement_evidence import build_evidence_package

    with _session(key) as s:
        return build_evidence_package(s, cluster_id=cluster_id)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def insight_options(key: str) -> list[dict]:
    with _session(key) as s:
        return [{"insight_id": c.insight_id, "cluster_id": c.cluster_id, "title": c.title, "need_code": c.need_code,
                 "count": c.unique_feedback_count, "status": c.status} for c in all_insight_cards(s, dq.VEHICLE)]


def write_session():
    """Uncached session for requirement reads/writes (workflow state must be fresh)."""
    return _session(db_key())
