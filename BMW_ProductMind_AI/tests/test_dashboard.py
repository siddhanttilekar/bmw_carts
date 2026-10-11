"""Part 3 dashboard tests: data layer against raw files, stable IDs, and each
Streamlit page loading on a real-data database copy. No LLM calls."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from core import config
from core.ai_models import CustomerNeedCluster, GeneratedCustomerInsight
from core.database import ensure_ai_schema, get_engine, session_scope
from services import dashboard_queries as dq
from services.cross_source_mapping import build_cross_source_links
from services.insight_generator import STATUS_TEMPLATE, all_insight_cards, generate_insights
from services.need_clustering import build_clusters, vfc_cluster_id
from services.need_taxonomy import MISSING_VFC, sync_taxonomy, vfc_mapping

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dash_db(built, tmp_path_factory):
    """Real-data DB with the Part 2 deterministic layer built (no LLM)."""
    path = tmp_path_factory.mktemp("dash") / "dash.db"
    shutil.copy2(built["db_path"], path)
    engine = get_engine(path)
    ensure_ai_schema(engine)
    with session_scope(engine) as s:
        sync_taxonomy(s)
        build_cross_source_links(s)
        build_clusters(s)
        generate_insights(s, min_unique_feedback=10)
    engine.dispose()
    return path


@pytest.fixture()
def s(dash_db):
    engine = get_engine(dash_db)
    with session_scope(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture(scope="module")
def raw(data_dir):
    df = pd.read_excel(next(data_dir.rglob(config.FEEDBACK_FILES["G60"])), sheet_name=config.FEEDBACK_SHEET, dtype=object)
    mapping = {n: code for n, (code, _, _) in vfc_mapping().items()}
    df["vfc_key"] = df["Vfc level2 Name"].map(lambda v: MISSING_VFC if pd.isna(v) else v)
    df["need"] = df["vfc_key"].map(mapping)
    return df


# --- counts -----------------------------------------------------------------------


def test_overview_counts_match_source(s, raw):
    o = dq.overview(s, dq.DashboardFilter())
    assert o.unique_feedback == raw["ID"].nunique() == 4000
    assert o.classification_rows == len(raw) == 5005
    assert o.need_categories_total == 18 and o.vfc_clusters_total == 222
    assert {n.need_code: n.unique_feedback_count for n in o.by_need} == raw.groupby("need")["ID"].nunique().to_dict()
    assert {d.value: d.feedback_ids for d in o.by_market} == raw.drop_duplicates("ID")["Country"].value_counts().to_dict()
    assert {d.value: d.feedback_ids for d in o.by_engine} == raw.drop_duplicates("ID")["Engine Type"].value_counts().to_dict()
    top = raw[~raw["vfc_key"].isin([MISSING_VFC, "no_class_found"])].groupby("vfc_key")["ID"].nunique()
    assert [c.feedback_ids for c in o.top_vfc] == sorted(top.to_list(), reverse=True)[:15]


def test_comments_are_never_counted_as_classifications(s, raw):
    o = dq.overview(s, dq.DashboardFilter(vfc_names=["Touch screen, operation"]))
    rows = raw[raw["vfc_key"] == "Touch screen, operation"]
    assert o.unique_feedback == rows["ID"].nunique() and o.classification_rows == len(rows)
    assert o.unique_feedback < o.classification_rows  # a comment with 2 such rows counts once
    total = dq.overview(s, dq.DashboardFilter())
    assert sum(d.feedback_ids for d in total.by_market) == total.unique_feedback  # markets partition comments


@pytest.mark.parametrize("f, mask", [
    (dq.DashboardFilter(countries=["US"], engine_types=["BEVE"]),
     lambda d: (d["Country"] == "US") & (d["Engine Type"] == "BEVE")),
    (dq.DashboardFilter(need_codes=["CONTROLS"], feedback_types=["Difficult to Use"]),
     lambda d: (d["need"] == "CONTROLS") & (d["Feedback Type"] == "Difficult to Use")),
    (dq.DashboardFilter(countries=["US"], engine_types=["BEVE"], need_codes=["CONTROLS"], feedback_types=["Difficult to Use"]),
     lambda d: (d["Country"] == "US") & (d["Engine Type"] == "BEVE") & (d["need"] == "CONTROLS")
     & (d["Feedback Type"] == "Difficult to Use")),
    (dq.DashboardFilter(need_codes=["CONTROLS"], vfc_names=["Touch screen, operation"]),
     lambda d: d["vfc_key"] == "Touch screen, operation"),
    (dq.DashboardFilter(feedback_types=[None]), lambda d: d["Feedback Type"].isna()),
    (dq.DashboardFilter(vfc_names=[MISSING_VFC]), lambda d: d["vfc_key"] == MISSING_VFC),
])
def test_filters_combine_correctly(s, raw, f, mask):
    subset = raw[mask(raw)]
    o = dq.overview(s, f)
    assert o.unique_feedback == subset["ID"].nunique()
    assert o.classification_rows == len(subset)


def test_contradictory_filters_return_empty_gracefully(s):
    o = dq.overview(s, dq.DashboardFilter(need_codes=["CHARGING"], vfc_names=["Touch screen, operation"]))
    assert o.unique_feedback == 0 and o.by_need == [] and o.top_vfc == []


# --- evidence ---------------------------------------------------------------------


def test_insight_evidence_ids_resolve_to_original_comments(s, raw):
    texts = raw.drop_duplicates("ID").set_index("ID")["Customer Feedback"].to_dict()
    from services.evidence_retrieval import get_feedback_by_ids

    for card in all_insight_cards(s)[:10]:
        records = get_feedback_by_ids(s, card.internal_evidence_ids)
        assert [r.original_feedback_id for r in records] == card.internal_evidence_ids
        assert all(r.customer_feedback == texts[r.original_feedback_id] for r in records)
        assert all(r.source.source_file == config.FEEDBACK_FILES["G60"] and r.source.source_row for r in records)


def test_public_and_internal_evidence_stay_separate(s):
    for need in ("CONTROLS", "INFOTAINMENT", "CHARGING"):
        for item in dq.related_evidence(s, need):
            assert item.tier in ("SUPPORTING", "CONTEXTUAL")
            if item.kind == "public_owner_feedback":
                assert item.source.data_origin == "public_external" and item.source_url and item.tier == "SUPPORTING"
            if item.kind == "competitor_observation":
                assert item.tier == "CONTEXTUAL" and item.source.data_origin == "public_external"
            if item.kind == "customer_study":
                assert item.study_values and all(v.vehicle_code == "G60" for v in item.study_values)
            assert item.source.source_file and item.source.source_row


# --- insights ----------------------------------------------------------------------


def test_template_insights_are_labelled_template(s):
    cards = all_insight_cards(s)
    assert len(cards) == s.query(GeneratedCustomerInsight).count() > 0
    assert {c.status for c in cards} == {STATUS_TEMPLATE}  # no extraction has run in this DB
    assert all(c.text_source == "deterministic_template" for c in cards)


# --- stable identifiers ---------------------------------------------------------------


def test_cluster_and_insight_ids_are_stable_across_rebuilds(s):
    def snapshot():
        clusters = {c.cluster_id: (c.cluster_level, c.need_code, c.vfc_level2_name, c.unique_feedback_count)
                    for c in s.scalars(select(CustomerNeedCluster))}
        insights = {i.insight_id: (i.cluster_id, i.internal_evidence_ids_json)
                    for i in s.scalars(select(GeneratedCustomerInsight))}
        return clusters, insights

    before = snapshot()
    build_clusters(s)
    generate_insights(s, min_unique_feedback=10)
    s.flush()
    assert snapshot() == before
    c = s.scalar(select(CustomerNeedCluster).where(CustomerNeedCluster.vfc_level2_name == "Touch screen, operation"))
    assert c.cluster_id == vfc_cluster_id("Touch screen, operation")
    assert s.get(GeneratedCustomerInsight, f"INS-{c.cluster_id}") is not None
    s.rollback()


def test_known_limitations_are_computed(s):
    lims = dq.known_limitations(s)
    assert len(lims) == 8
    assert lims[-1].title == "No real LLM extraction has run yet"
    assert "US: 3610 of 4000" in lims[2].evidence


# --- Streamlit pages --------------------------------------------------------------------


PAGES = ["ui/pages/discover.py", "ui/pages/define.py", "ui/pages/prioritize.py", "ui/pages/decide.py",
         "ui/pages/customer_intelligence.py", "ui/pages/evidence_explorer.py",
         "ui/pages/customer_insights.py", "ui/pages/data_quality.py"]


@pytest.fixture()
def app(dash_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRODUCTMIND_DB_PATH", str(dash_db))

    def forbid(*a, **k):
        raise AssertionError("The dashboard must not construct an LLM client")

    monkeypatch.setattr("services.llm_client.AnthropicLLMClient.__init__", forbid)
    monkeypatch.setattr("anthropic.Anthropic.__init__", forbid, raising=False)
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    return at


def test_every_page_loads_without_errors(app):
    app.run()
    assert not app.exception
    for page in PAGES[1:]:
        app.switch_page(page).run()
        assert not app.exception, (page, [e.value for e in app.exception])


def test_overview_page_shows_database_counts(app):
    app.run()  # default page is now the main Dashboard
    html = " ".join(m.value for m in app.markdown)
    assert "BMW ProductMind AI" in html and "4,000" in html and "Unique G60 customer comments" in html
    assert "From Customer Feedback to Vehicle Decisions" in html
    app.switch_page("ui/pages/discover.py").run()  # the earlier step view stays available under Advanced
    html = " ".join(m.value for m in app.markdown)
    assert "4,000" in html and "Customer comments analyzed" in html
    assert "Customer needs identified" in html and "Requirements created" in html
    assert "Classification records" not in html


def test_overview_filters_update_counts(app, raw):
    app.switch_page("ui/pages/customer_intelligence.py").run()
    app.multiselect(key="ci_market").set_value(["KR"]).run()
    expected = raw[raw["Country"] == "KR"]["ID"].nunique()
    assert f'class="v">{expected:,}<' in " ".join(m.value for m in app.markdown)


def test_insights_page_labels_templates_and_links_to_explorer(app):
    app.switch_page("ui/pages/customer_insights.py").run()
    text = " ".join(m.value for m in app.markdown) + " ".join(i.value for i in app.info)
    assert "TEMPLATE" in text and "Nothing here is an AI-generated finding" in text
    buttons_shown = [b for b in app.button if b.label == "View Supporting Evidence"]
    # The status legend shows each badge once; every card on the page carries TEMPLATE, none AI_ENRICHED.
    assert text.count("AI_ENRICHED · uses LLM extraction") == 1
    assert text.count("TEMPLATE · from VFC classifications") == 1 + len(buttons_shown)
    buttons = [b for b in app.button if b.label == "View Supporting Evidence"]
    assert buttons
    target = buttons[0].key.removeprefix("view_INS-")
    buttons[0].click().run()
    assert not app.exception
    assert app.session_state["ex_cluster"] == target


def test_explorer_marks_review_items_and_tiers(app):
    app.switch_page("ui/pages/evidence_explorer.py").run()
    app.selectbox(key="ex_need").set_value("CONTROLS").run()
    html = " ".join(m.value for m in app.markdown)
    assert "Direct · internal BMW feedback" in html
    assert "Supporting · public BMW feedback / BMW study" in html
    assert any("requires review" in w.value for w in app.warning)
    assert "pm-src" in html  # source references rendered


def test_dashboard_code_never_references_llm_pipeline():
    """2.0: the UI may START approval-gated paid actions only through
    services.evidence_extraction / ai_evidence_selector (which enforce
    authorisation, approval and budget); it never builds a client or calls the
    extraction pipeline directly."""
    sources = [p.read_text() for p in (ROOT / "ui").rglob("*.py")] + [(ROOT / "app.py").read_text(),
                                                                       (ROOT / "services/dashboard_queries.py").read_text()]
    for src in sources:
        for forbidden in ("AnthropicLLMClient", "run_extraction", "import anthropic", "llm_client"):
            assert forbidden not in src
