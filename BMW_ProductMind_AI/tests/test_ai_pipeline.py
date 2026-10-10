"""Part 2 AI pipeline tests on real G60 records with a mocked LLM (no network, no cost)."""
from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd
import pytest
from sqlalchemy import func, select, text

from core import config
from core.ai_models import (
    AIApiCall,
    AIFeedbackExtraction,
    AIProcessingJob,
    ClusterFeedbackLink,
    CrossSourceEvidenceLink,
    CustomerNeedCluster,
    ExtractedCustomerNeed,
    GeneratedCustomerInsight,
    VfcNeedMapping,
)
from core.config import LLMSettings
from core.database import ensure_ai_schema, get_engine, reset_schema, session_scope
from services import ai_feedback_analysis as ai
from services.cross_source_mapping import build_cross_source_links, comments_for_need, evidence_for_need
from services.insight_generator import InsightText, check_llm_text, build_facts, generate_insights, validate_insight
from services.llm_client import LLMError, LLMResponse, LLMUsage, estimate_cost_usd
from services.need_clustering import build_clusters
from services.need_taxonomy import MISSING_VFC, NEED_CODES, sync_taxonomy, vfc_mapping


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


class FakeLLM:
    """Deterministic stand-in for the Claude API. Builds valid extraction JSON
    from the comments in the request; behaviours can be injected per call."""

    def __init__(self, model="fake-model", *, two_needs=True, extra_ids=(), drop_first_call=False,
                 invalid_first=False, rate_limit_first=False, interrupt_on_call=None, polarity="unknown",
                 bad_polarity_first=False, insight_text=None):
        self.model = model
        self.calls: list[list[str]] = []
        self.two_needs = two_needs
        self.extra_ids = list(extra_ids)
        self.drop_first_call = drop_first_call
        self.invalid_first = invalid_first
        self.rate_limit_first = rate_limit_first
        self.interrupt_on_call = interrupt_on_call
        self.polarity = polarity
        self.bad_polarity_first = bad_polarity_first
        self.insight_text = insight_text

    def complete_json(self, *, system, user, schema, max_tokens):
        usage = LLMUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=800)
        if "facts" in user and self.insight_text is not None:  # insight request
            self.calls.append(["insight"])
            return LLMResponse(text=json.dumps(self.insight_text), stop_reason="end_turn", model=self.model, usage=usage)
        payload = json.loads(user.split("\n\n", 1)[1])
        ids = [p["feedback_id"] for p in payload]
        self.calls.append(ids)
        n = len(self.calls)
        if self.interrupt_on_call == n:
            raise KeyboardInterrupt
        if self.rate_limit_first and n == 1:
            raise LLMError("429", kind="rate_limited", retryable=True, retry_after_s=7.0)
        if self.invalid_first and n == 1:
            return LLMResponse(text="{not json", stop_reason="end_turn", model=self.model, usage=usage)
        results = []
        for i, p in enumerate(payload):
            if self.drop_first_call and n == 1 and i == 0:
                continue
            comment = p["comment"]
            needs = [{
                "problem": "", "customer_need": f"Need A for {p['feedback_id']}", "category": "CONTROLS",
                "polarity": "sarcastic" if (self.bad_polarity_first and n == 1) else self.polarity,
                "vehicle_system": "unspecified", "confidence": "medium", "supporting_quote": comment[:25],
            }]
            if self.two_needs:
                needs.append({
                    "problem": "Example problem", "customer_need": "Shared need statement", "category": "INFOTAINMENT",
                    "polarity": "negative", "vehicle_system": "Display", "confidence": "low",
                    "supporting_quote": "this text is not in the comment",
                })
            results.append({"feedback_id": p["feedback_id"], "summary": "Factual summary.", "extracted_needs": needs})
        for fid in self.extra_ids:
            results.append({"feedback_id": fid, "summary": "x", "extracted_needs": []})
        return LLMResponse(text=json.dumps({"results": results}), stop_reason="end_turn", model=self.model,
                           request_id=f"req_{n}", usage=usage)


SETTINGS = LLMSettings(model="fake-model", batch_size=5, max_attempts=3)


# ---------------------------------------------------------------------------
# Fixtures: a copy of the real database per test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ai_template(built, tmp_path_factory):
    path = tmp_path_factory.mktemp("ai") / "template.db"
    shutil.copy2(built["db_path"], path)
    engine = get_engine(path)
    ensure_ai_schema(engine)
    with session_scope(engine) as s:
        sync_taxonomy(s)
        build_cross_source_links(s)
    engine.dispose()
    return path


@pytest.fixture()
def engine(ai_template, tmp_path):
    path = tmp_path / "ai.db"
    shutil.copy2(ai_template, path)
    eng = get_engine(path)
    yield eng
    eng.dispose()


@pytest.fixture()
def s(engine):
    with session_scope(engine) as session:
        yield session


@pytest.fixture(scope="module")
def g60_raw(data_dir):
    path = next(data_dir.rglob(config.FEEDBACK_FILES["G60"]))
    return pd.read_excel(path, sheet_name=config.FEEDBACK_SHEET, dtype=object)


def sample_ids(s, n=12, need_code=None):
    return ai.select_sample(s, n, "G60", need_code=need_code)


# ---------------------------------------------------------------------------
# Taxonomy and mappings
# ---------------------------------------------------------------------------


def test_every_g60_vfc_class_is_mapped_and_none_forced(s, g60_raw):
    mapping = vfc_mapping()
    names = {MISSING_VFC if pd.isna(v) else v for v in g60_raw["Vfc level2 Name"]}
    assert names == set(mapping)
    assert {n for n, (code, _, _) in mapping.items() if code == "OTHER"} == {"no_class_found", MISSING_VFC}
    assert all(reason for _, status, reason in mapping.values() if status == "review_required")
    assert {code for code, _, _ in mapping.values()} <= NEED_CODES


def test_original_vfc_values_unchanged_by_pipeline(s):
    def fingerprint():
        rows = s.execute(text("SELECT record_id, feedback_type, vfc_level2_3_name, vfc_level2_name, vfc_level3_name "
                              "FROM feedback_classifications ORDER BY record_id")).all()
        return hashlib.sha256(repr(rows).encode()).hexdigest()

    before = fingerprint()
    ai.run_extraction(s, FakeLLM(), SETTINGS, sample_ids(s, 10))
    build_clusters(s)
    generate_insights(s, min_unique_feedback=5)
    s.commit()
    assert fingerprint() == before


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_original_ids_preserved_and_multiple_needs_per_comment(s):
    ids = sample_ids(s, 7)
    job = ai.run_extraction(s, FakeLLM(), SETTINGS, ids)
    assert job.status == "completed" and job.processed_items == 7
    rows = s.scalars(select(AIFeedbackExtraction)).all()
    assert sorted(r.original_feedback_id for r in rows) == sorted(ids)
    record_ids = dict(s.execute(text("SELECT original_feedback_id, record_id FROM feedback WHERE vehicle_code='G60'")).all())
    assert all(r.feedback_record_id == record_ids[r.original_feedback_id] for r in rows)
    needs = s.scalars(select(ExtractedCustomerNeed)).all()
    assert len(needs) == 14 and all(r.need_count == 2 for r in rows)
    per_comment = {}
    for n in needs:
        per_comment.setdefault(n.original_feedback_id, set()).add(n.need_code)
    assert all(v == {"CONTROLS", "INFOTAINMENT"} for v in per_comment.values())


def test_quotes_are_verified_against_original_text(s):
    ai.run_extraction(s, FakeLLM(), SETTINGS, sample_ids(s, 5))
    needs = s.scalars(select(ExtractedCustomerNeed)).all()
    assert {n.quote_verified for n in needs if n.need_code == "CONTROLS"} == {True}
    assert {n.quote_verified for n in needs if n.need_code == "INFOTAINMENT"} == {False}


def test_unknown_polarity_is_stored_and_invalid_polarity_is_rejected(s):
    ai.run_extraction(s, FakeLLM(polarity="unknown", bad_polarity_first=True), SETTINGS, sample_ids(s, 3))
    calls = s.scalars(select(AIApiCall).order_by(AIApiCall.call_id)).all()
    assert calls[0].status == "invalid_output" and calls[1].status == "succeeded"
    polarities = {n.polarity for n in s.scalars(select(ExtractedCustomerNeed)) if n.need_code == "CONTROLS"}
    assert polarities == {"unknown"}


def test_feedback_type_is_not_sent_to_the_model(s):
    comments = ai.load_comments(s, sample_ids(s, 3))
    message = ai._user_message(comments)
    assert "feedback_type" not in message.lower()
    assert all(c.vfc_labels is not None for c in comments)


def test_fabricated_and_foreign_ids_are_never_stored(s):
    ids = sample_ids(s, 4)
    llm = FakeLLM(extra_ids=["G60-9999", "F70-0001", ids[0] + "X"])
    ai.run_extraction(s, llm, SETTINGS, ids)
    stored = set(s.scalars(select(AIFeedbackExtraction.original_feedback_id)))
    assert stored == set(ids)
    assert not s.scalar(select(func.count()).select_from(ExtractedCustomerNeed)
                        .where(ExtractedCustomerNeed.original_feedback_id.not_in(ids)))


def test_missing_results_are_retried_only_for_missing_ids(s):
    ids = sample_ids(s, 5)
    llm = FakeLLM(drop_first_call=True)
    job = ai.run_extraction(s, llm, SETTINGS, ids)
    assert job.processed_items == 5 and job.failed_items == 0
    assert llm.calls[1] == [ids[0]]  # only the dropped comment is re-sent


def test_invalid_json_is_retried(s):
    llm = FakeLLM(invalid_first=True)
    job = ai.run_extraction(s, llm, SETTINGS, sample_ids(s, 3))
    assert job.status == "completed" and len(llm.calls) == 2


def test_rate_limit_waits_and_retries(s):
    waits = []
    llm = FakeLLM(rate_limit_first=True)
    job = ai.run_extraction(s, llm, SETTINGS, sample_ids(s, 3), sleep=waits.append)
    assert waits == [7.0] and job.status == "completed"
    statuses = [c.status for c in s.scalars(select(AIApiCall).order_by(AIApiCall.call_id))]
    assert statuses == ["rate_limited", "succeeded"]


def test_usage_and_cost_are_logged(s):
    job = ai.run_extraction(s, FakeLLM(), SETTINGS, sample_ids(s, 10))
    calls = s.scalars(select(AIApiCall)).all()
    assert len(calls) == 2 == job.api_calls
    assert job.input_tokens == 2000 and job.output_tokens == 1000 and job.cache_read_tokens == 1600
    assert job.estimated_cost_usd == pytest.approx(sum(c.estimated_cost_usd for c in calls))
    assert estimate_cost_usd("claude-opus-5-5", LLMUsage(input_tokens=1_000_000)) == 4.0


# ---------------------------------------------------------------------------
# Caching, checkpoints, resume, guards
# ---------------------------------------------------------------------------


def test_cached_comments_are_not_reprocessed(s):
    ids = sample_ids(s, 10)
    first = FakeLLM()
    ai.run_extraction(s, first, SETTINGS, ids)
    second = FakeLLM()
    job = ai.run_extraction(s, second, SETTINGS, ids)
    assert second.calls == [] and job.cached_items == 10 and job.api_calls == 0
    other_model = FakeLLM(model="another-model")  # cache key includes the model
    ai.run_extraction(s, other_model, SETTINGS, ids[:5])
    assert len(other_model.calls) == 1


def test_cache_key_changes_with_text_model_and_prompt():
    base = ai.cache_key("G60-0001", "h1", "m", "p")
    assert base == ai.cache_key("G60-0001", "h1", "m", "p")
    assert len({base, ai.cache_key("G60-0001", "h2", "m", "p"), ai.cache_key("G60-0001", "h1", "m2", "p"),
                ai.cache_key("G60-0001", "h1", "m", "p2"), ai.cache_key("G60-0002", "h1", "m", "p")}) == 5


def test_interrupted_job_resumes_without_repeating_finished_batches(s):
    ids = sample_ids(s, 15)  # 3 batches of 5
    llm = FakeLLM(interrupt_on_call=2)
    with pytest.raises(KeyboardInterrupt):
        ai.run_extraction(s, llm, SETTINGS, ids, job_id="JOB-TEST")
    job = s.get(AIProcessingJob, "JOB-TEST")
    assert job.status == "interrupted"
    assert s.scalar(select(func.count()).select_from(AIFeedbackExtraction)) == 5  # batch 1 checkpointed

    resumed_llm = FakeLLM()
    job = ai.resume_extraction(s, resumed_llm, SETTINGS, "JOB-TEST")
    assert job.status == "completed" and job.processed_items == 15 and job.cached_items == 5
    assert sorted(i for call in resumed_llm.calls for i in call) == sorted(ids[5:])


def test_full_run_requires_explicit_confirmation(s):
    ids = ai.all_feedback_ids(s)
    assert len(ids) == 4000
    llm = FakeLLM()
    with pytest.raises(ai.FullRunNotConfirmed):
        ai.run_extraction(s, llm, SETTINGS, ids)
    assert llm.calls == []


def test_f70_and_g70_records_cannot_enter_g60_pipeline(s):
    with pytest.raises(ValueError):
        ai.load_comments(s, ["G60-0001", "F70-0001"])
    with pytest.raises(ValueError):
        ai.load_comments(s, ["G70-0001"])
    assert all(i.startswith("G60-") for i in ai.select_sample(s, 500))
    assert all(i.startswith("G60-") for i in comments_for_need(s, "CONTROLS"))


# ---------------------------------------------------------------------------
# Clustering and counts
# ---------------------------------------------------------------------------


def test_cluster_counts_match_source_and_distinguish_comments_from_rows(s, g60_raw):
    build_clusters(s)
    c = s.scalar(select(CustomerNeedCluster).where(CustomerNeedCluster.cluster_level == "vfc_category",
                                                    CustomerNeedCluster.vfc_level2_name == "Touch screen, operation"))
    rows = g60_raw[g60_raw["Vfc level2 Name"] == "Touch screen, operation"]
    assert c.unique_feedback_count == rows["ID"].nunique() and c.classification_count == len(rows)
    assert c.classification_count > c.unique_feedback_count

    mapping = {n: code for n, (code, _, _) in vfc_mapping().items()}
    raw = g60_raw.assign(need=g60_raw["Vfc level2 Name"].map(lambda v: mapping[MISSING_VFC if pd.isna(v) else v]))
    for need_cluster in s.scalars(select(CustomerNeedCluster).where(CustomerNeedCluster.cluster_level == "need_category")):
        subset = raw[raw["need"] == need_cluster.need_code]
        assert need_cluster.unique_feedback_count == subset["ID"].nunique(), need_cluster.need_code
        assert need_cluster.classification_count == len(subset), need_cluster.need_code
    total_need_links = s.scalar(select(func.count(func.distinct(ClusterFeedbackLink.original_feedback_id))))
    assert total_need_links == 4000  # every G60 comment lands in >=1 need-category cluster


def test_extracted_need_clusters_link_back_to_comments(s):
    ids = sample_ids(s, 10)
    ai.run_extraction(s, FakeLLM(), SETTINGS, ids)
    build_clusters(s)
    shared = s.scalar(select(CustomerNeedCluster).where(CustomerNeedCluster.cluster_level == "extracted_need",
                                                         CustomerNeedCluster.customer_need == "Shared need statement"))
    assert shared.unique_feedback_count == 10 and shared.extracted_need_count == 10
    linked = set(s.scalars(select(ClusterFeedbackLink.original_feedback_id).where(ClusterFeedbackLink.cluster_id == shared.cluster_id)))
    assert linked == set(ids)
    singles = s.scalars(select(CustomerNeedCluster).where(CustomerNeedCluster.cluster_level == "extracted_need",
                                                          CustomerNeedCluster.need_code == "CONTROLS")).all()
    assert len(singles) == 10 and all(c.unique_feedback_count == 1 for c in singles)  # distinct statements not merged


# ---------------------------------------------------------------------------
# Cross-source mapping
# ---------------------------------------------------------------------------


def test_cross_source_mappings_preserve_source_distinctions(s):
    links = s.scalars(select(CrossSourceEvidenceLink)).all()
    for l in links:
        if l.evidence_tier == "direct":
            assert l.source_table == "feedback_classifications" and l.data_origin == "internal_bmw"
        if l.source_table == "public_feedback":
            assert l.evidence_tier == "supporting" and l.data_origin == "public_external"
        if l.evidence_tier == "contextual":
            assert l.source_table == "competitor_evidence"
    brand = dict(s.execute(text("SELECT source_row, brand FROM competitor_evidence")).all())
    for l in (l for l in links if l.source_table == "competitor_evidence"):
        row = int(l.source_key.removeprefix("COMP-R"))
        assert (l.evidence_tier == "supporting") == (brand[row] == "BMW")
    ev = evidence_for_need(s, "CONTROLS")
    assert {e.source_key for e in ev.contextual} == {"COMP-R7"}
    assert all(e.record_ids for e in ev.supporting)


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


def test_insights_reference_only_existing_evidence(s):
    ai.run_extraction(s, FakeLLM(), SETTINGS, sample_ids(s, 10, "CONTROLS"))
    build_clusters(s)
    stats = generate_insights(s, min_unique_feedback=5)
    assert stats["insights"] > 0 and stats["invalid"] == 0
    for insight in s.scalars(select(GeneratedCustomerInsight)):
        assert insight.validated and validate_insight(s, insight) == []
        ids = json.loads(insight.internal_evidence_ids_json)
        assert len(ids) == insight.unique_feedback_count and all(i.startswith("G60-") for i in ids)
        assert insight.need_code != "OTHER"


def test_tampered_insight_fails_validation(s):
    build_clusters(s)
    generate_insights(s, min_unique_feedback=20)
    insight = s.scalars(select(GeneratedCustomerInsight)).first()
    insight.internal_evidence_ids_json = json.dumps(json.loads(insight.internal_evidence_ids_json) + ["G60-9999"])
    insight.public_evidence_ids_json = json.dumps(["P999"])
    insight.unique_feedback_count += 1
    errors = validate_insight(s, insight)
    assert any("internal" in e for e in errors) and any("P999" in e for e in errors)
    assert any("unique_feedback_count" in e for e in errors)


def test_llm_insight_text_with_invented_numbers_is_rejected(s):
    build_clusters(s)
    fabricated = {"title": "Touchscreen issues", "customer_problem": "73% of customers dislike it (G60-9999).",
                  "customer_need": "Physical buttons", "evidence_summary": "Based on 999 comments and P042."}
    stats = generate_insights(s, need_code="CONTROLS", min_unique_feedback=40, llm=FakeLLM(insight_text=fabricated))
    assert stats["llm_rejected"] == stats["insights"] > 0 and stats["llm_text"] == 0
    for insight in s.scalars(select(GeneratedCustomerInsight)):
        assert insight.text_source == "deterministic_template" and "999" not in insight.evidence_summary
        assert any("number not in facts" in e for e in json.loads(insight.validation_errors_json))


def test_llm_insight_text_using_only_facts_is_accepted(s):
    build_clusters(s)
    cluster = s.scalar(select(CustomerNeedCluster).where(CustomerNeedCluster.vfc_level2_name == "Touch screen, operation"))
    facts = build_facts(s, cluster)
    ok = InsightText(title="Touchscreen operation", customer_problem="Customers describe usability issues.",
                     customer_need="Easy touchscreen operation.",
                     evidence_summary=f"{facts.unique_feedback_count} comments; see {facts.internal_evidence_ids[0]}.")
    assert check_llm_text(ok, facts) == []
    bad = ok.model_copy(update={"evidence_summary": "Half of all owners (50%) complain."})
    assert check_llm_text(bad, facts)


# ---------------------------------------------------------------------------
# Schema safety
# ---------------------------------------------------------------------------


def test_ingestion_rebuild_preserves_ai_results(engine):
    with session_scope(engine) as s:
        ai.run_extraction(s, FakeLLM(), SETTINGS, ai.select_sample(s, 5))
    reset_schema(engine)  # what initialize_database does
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(AIFeedbackExtraction)) == 5
        assert s.scalar(select(func.count()).select_from(VfcNeedMapping)) == 222
    assert ensure_ai_schema(engine) == []  # idempotent
