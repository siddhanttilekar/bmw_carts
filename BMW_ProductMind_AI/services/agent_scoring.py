"""Registry of supported scoring methods (Part 5).

Agent behaviour is defined ONLY by its scoring method and compatible
resources - never by its name. Every method is deterministic, returns a score
in [0, 100] or ``None`` with status UNSCORED, and records inputs, source
references, rationale, limitations and its version. Missing evidence never
becomes a neutral score.

2.0 (evidence-first): AI_ANALYSIS methods never let a model choose a number.
The LLM only identifies evidence (validated observations + requirement
evidence selection); these Python methods count distinct validated feedback
IDs and apply a fixed, versioned normalisation:

    score = 100 * ln(1 + n) / ln(1 + N),  N = 4000 (fixed reference)

The score is NOT a percentage of customers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.ai_models import ClusterFeedbackLink
from core.models import Feedback
from core.prioritization_models import PriorityAssessment

SCORED, UNSCORED = "SCORED", "UNSCORED"
AI_UNAVAILABLE = "AI analysis unavailable. Use Data Analysis or Human Assessment."

REFERENCE_POPULATION_N = 4000  # fixed: distinct G60 comments in the supplied export when the method was registered
NORMALIZATION = {"version": "log-feedback-v1", "formula": "100 * ln(1 + n) / ln(1 + N)", "N": REFERENCE_POPULATION_N,
                 "N_definition": "fixed reference population: 4,000 distinct G60 comments in the supplied export",
                 "interpretation": "relative evidence strength on a logarithmic scale - not a percentage of customers"}


def log_normalize(n: int, population: int = REFERENCE_POPULATION_N) -> float:
    """Deterministic log normalisation; n is capped at N."""
    if n < 0 or population < 1:
        raise ValueError("n must be >= 0 and N >= 1")
    return round(100 * math.log1p(min(n, population)) / math.log1p(population), 2)

# Market-relevance prerequisites. These are facts about the current data and
# can only be changed by a documented human validation (not from the UI).
MARKET_VALIDATION = {
    "country_to_sales_region_mapping_validated": False,
    "g60_only_sales_available": False,  # sales exist only as G60+G68 combined
}


class ScoreResult(BaseModel):
    status: str
    score: float | None
    method: str
    method_version: str
    inputs: dict = Field(default_factory=dict)
    source_refs: list[dict] = Field(default_factory=list)  # {"table", "key", "record_id"}
    rationale: str
    facts: list[str] = Field(default_factory=list)
    assessments: list[dict] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ScoringMethod:
    method_id: str
    version: str
    label: str
    execution_methods: tuple[str, ...]
    description: str
    required_resources: frozenset[str]
    any_of_resources: frozenset[str] = frozenset()  # at least one of these
    score_label: str = "score"
    limitations: tuple[str, ...] = ()
    fn: Callable | None = field(default=None, compare=False)
    formula: str = ""
    selectable: bool = True  # False for legacy/disabled methods kept only for historical runs

    @property
    def execution_method(self) -> str:  # primary execution method (backwards compatible)
        return self.execution_methods[0]


def _unscored(method: ScoringMethod, reason: str, **kw) -> ScoreResult:
    return ScoreResult(status=UNSCORED, score=None, method=method.method_id, method_version=method.version,
                       rationale=reason, limitations=list(method.limitations) + kw.pop("limitations", []), **kw)


# ---------------------------------------------------------------------------
# METHOD 1 - log-normalised unique feedback count
# ---------------------------------------------------------------------------


def _log_feedback(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    keys = sorted({e["source_key"] for e in req["evidence"] if e["tier"] == "DIRECT" and e["source_table"] == "feedback"})
    if not keys:
        return _unscored(method, "The requirement has no direct customer-feedback evidence.")
    found = dict(session.execute(select(Feedback.original_feedback_id, Feedback.vehicle_code)
                                 .where(Feedback.original_feedback_id.in_(keys))).all())
    missing = [k for k in keys if k not in found]
    foreign = [k for k in keys if found.get(k) not in (None, "G60")]
    if missing or foreign:
        return _unscored(method, "Evidence contains IDs that are not valid G60 feedback; fix the requirement evidence "
                                 "before scoring.",
                         inputs={"invalid_ids": missing[:20], "non_g60_ids": foreign[:20]},
                         limitations=["F70/G70 or unknown IDs are never counted in G60 demand."])
    n = len(keys)  # distinct original feedback IDs (classification rows are not counted)
    population = session.scalar(select(func.count(func.distinct(Feedback.original_feedback_id)))
                                .where(Feedback.vehicle_code == "G60")) or 0
    if population < 1:
        return _unscored(method, "No G60 reference population available.")
    score = round(100 * math.log1p(n) / math.log1p(population), 2)
    cluster_now = session.scalar(select(func.count(func.distinct(ClusterFeedbackLink.feedback_record_id)))
                                 .where(ClusterFeedbackLink.cluster_id == req["source_cluster_id"]))
    return ScoreResult(
        status=SCORED, score=min(score, 100.0), method=method.method_id, method_version=method.version,
        inputs={"unique_g60_feedback_ids": n, "reference_population": population,
                "reference_population_definition": "all distinct G60 feedback IDs in the database",
                "formula": "100 * ln(1 + n) / ln(1 + N)", "source_cluster_id": req["source_cluster_id"],
                "source_cluster_unique_feedback_now": cluster_now},
        source_refs=[{"table": "feedback", "key": k, "record_id": None} for k in keys],
        rationale=f"{n} distinct G60 comments are linked as direct evidence; log-normalised against all "
                  f"{population} distinct G60 comments.",
        facts=[f"{n} distinct G60 feedback IDs are linked to this requirement as direct evidence.",
               f"Reference population: {population} distinct G60 feedback IDs.",
               f"Source cluster {req['source_cluster_id']} currently has {cluster_now} distinct comments."],
        assumptions=["Logarithmic normalisation (diminishing weight of additional comments) is a prototype "
                     "methodology, not a validated BMW standard."],
        limitations=list(method.limitations),
    )


# ---------------------------------------------------------------------------
# METHOD 2 - human score
# ---------------------------------------------------------------------------


def _human(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    a = session.scalars(select(PriorityAssessment).where(
        PriorityAssessment.requirement_id == req["requirement_id"], PriorityAssessment.agent_id == agent["agent_id"],
        PriorityAssessment.superseded.is_(False)).order_by(PriorityAssessment.assessment_timestamp.desc())).first()
    if a is None:
        return _unscored(method, f"No human assessment recorded for agent '{agent['name']}'.")
    import json

    refs = json.loads(a.evidence_refs_json)
    return ScoreResult(
        status=SCORED, score=float(a.score), method=method.method_id, method_version=method.version,
        inputs={"assessment_id": a.assessment_id, "assessed_with_agent_version": a.agent_version},
        source_refs=[{"table": "priority_assessments", "key": a.assessment_id, "record_id": a.assessment_id}]
                    + [{"table": "evidence", "key": r, "record_id": None} for r in refs],
        rationale=f"Human assessment by {a.assessor_id}: {a.rationale}",
        assessments=[{"assessment_id": a.assessment_id, "assessor_id": a.assessor_id, "score": a.score,
                      "rationale": a.rationale, "timestamp": a.assessment_timestamp.isoformat(),
                      "evidence_references": refs, "agent_version": a.agent_version}],
        assumptions=["Score reflects the assessor's documented judgment."],
        limitations=list(method.limitations),
    )


# ---------------------------------------------------------------------------
# METHOD 3 - evidence coverage
# ---------------------------------------------------------------------------

COVERAGE_TYPES = {  # resource -> (label, predicate on evidence-link dict)
    "RES_G60_FEEDBACK": ("direct G60 feedback", lambda e: e["tier"] == "DIRECT" and e["source_table"] == "feedback"),
    "RES_PUBLIC_FEEDBACK": ("public BMW owner evidence",
                            lambda e: e["tier"] == "SUPPORTING" and e["source_table"] in ("public_feedback", "competitor_evidence")),
    "RES_CUSTOMER_STUDIES": ("customer-study items", lambda e: e["source_table"] == "customer_studies"),
    "RES_COMPETITOR": ("competitor observations",
                       lambda e: e["tier"] == "CONTEXTUAL" and e["source_table"] == "competitor_evidence"),
    "RES_EQUIPMENT": ("G60 option-list entries", lambda e: e["source_table"] == "equipment"),
}


EXTRA_COVERAGE_TYPES = {"RES_AI_OBSERVATIONS": "AI-validated observations in the evidence selection",
                        "RES_UPLOADED_DOCUMENTS": "passages in this agent's uploaded documents"}


def _extra_coverage(session: Session, r: str, req: dict, agent: dict) -> tuple[int, list[dict]]:
    if r == "RES_AI_OBSERVATIONS":
        sel = current_evidence(session, req, agent if agent.get("execution_method") == "AI_ANALYSIS" else None)
        if sel is None:
            return 0, []
        ids = sorted(set(sel.combined_demand_ids) | set(sel.positive_feedback_ids))
        return len(ids), [{"table": "evidence_selection_runs", "key": sel.selection_id, "record_id": sel.selection_id}]
    if agent.get("execution_method") == "AI_ANALYSIS":  # passages the agent's AI analysis judged relevant
        from services.agent_analysis import current_analysis, evidence_view

        row = current_analysis(session, agent, req)
        if row is not None:
            refs = evidence_view(session, row).document_refs
            return len(refs), [{"table": "agent_document_chunks", "key": d["chunk_id"], "record_id": d["document_id"]}
                               for d in refs]
    from services.agent_documents import keywords, search_agent_documents

    _, hits = search_agent_documents(session, agent["agent_id"], keywords(req["title"], req["customer_need"]))
    return len(hits), [{"table": "agent_document_chunks", "key": h["chunk_id"], "record_id": h["document_id"]} for h in hits]


def _coverage(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    from services.requirement_validator import _resolve

    types = [r for r in agent["resources"] if r in COVERAGE_TYPES or r in EXTRA_COVERAGE_TYPES]
    if not types:
        return _unscored(method, "No evidence resource is assigned to this agent.")
    members = set(session.scalars(select(ClusterFeedbackLink.original_feedback_id)
                                  .where(ClusterFeedbackLink.cluster_id == req["source_cluster_id"])))
    per_type, refs, invalid = {}, [], []
    for r in types:
        if r in EXTRA_COVERAGE_TYPES:
            n, extra_refs = _extra_coverage(session, r, req, agent)
            per_type[r] = {"label": EXTRA_COVERAGE_TYPES[r], "valid_links": n}
            refs += extra_refs if n else []
            continue
        label, pred = COVERAGE_TYPES[r]
        links = [e for e in req["evidence"] if pred(e)]
        valid = []
        for e in links:
            problem = _resolve(session, e, req["source_cluster_id"], members)
            (invalid if problem else valid).append(problem or e)
        per_type[r] = {"label": label, "valid_links": len(valid)}
        refs += [{"table": e["source_table"], "key": e["source_key"], "record_id": e.get("source_record_id")} for e in valid]
    if invalid:
        return _unscored(method, "Some evidence links do not resolve to database records; fix them before scoring.",
                         inputs={"invalid": invalid[:10]})
    covered = sum(1 for v in per_type.values() if v["valid_links"])
    score = round(100 * covered / len(types), 2)
    return ScoreResult(
        status=SCORED, score=score, method=method.method_id, method_version=method.version,
        inputs={"assigned_evidence_types": len(types), "covered_evidence_types": covered, "per_type": per_type,
                "formula": "100 * covered evidence types / assigned evidence types"},
        source_refs=refs,
        rationale=f"EVIDENCE COVERAGE (not customer impact or business importance): {covered} of {len(types)} "
                  f"assigned evidence types have at least one valid, traceable link.",
        facts=[f"{v['label']}: {v['valid_links']} valid links" for v in per_type.values()],
        assumptions=["Each assigned evidence type counts equally (prototype methodology)."],
        limitations=list(method.limitations),
    )


# ---------------------------------------------------------------------------
# METHOD 4 - validated market relevance
# ---------------------------------------------------------------------------


def _market(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    unmet = [k for k, ok in MARKET_VALIDATION.items() if not ok]
    if unmet:
        return _unscored(method, "Market relevance cannot be scored: prerequisites are not validated ("
                         + ", ".join(unmet) + ").",
                         inputs={"prerequisites": dict(MARKET_VALIDATION)},
                         limitations=["Customer feedback is about 90% US; sales are G60+G68 combined; market share "
                                      "is not inferred from these datasets."])
    return _unscored(method, "Validated market-relevance calculation is not implemented in this prototype.")


# ---------------------------------------------------------------------------
# 2.0 AI_ANALYSIS methods - AI identifies evidence, Python counts
# ---------------------------------------------------------------------------

FREQUENCY_KINDS = {  # method -> (selection attribute, label of what is counted)
    "COMPLAINT_FREQUENCY": ("supporting_complaint_ids", "validated complaints"),
    "CUSTOMER_REQUEST_FREQUENCY": ("improvement_request_ids", "validated improvement requests"),
    "COMBINED_CUSTOMER_DEMAND": ("combined_demand_ids", "comments with a validated complaint or improvement request "
                                                        "(union - a comment with both counts once)"),
}


def current_evidence(session: Session, req: dict, agent: dict | None = None):
    """Evidence for scoring. For an AI_ANALYSIS agent with a completed analysis
    matching its CURRENT instructions / resources / evidence version, the
    agent-specific validated lists (Step 2); otherwise the deterministic
    taxonomy selection for the requirement version (None if none exists).
    Never calls a model."""
    from core.config import get_llm_settings
    from services.ai_evidence_selector import latest_selection
    from services.ai_feedback_analysis import current_observation_version_id

    if agent is not None:
        from services.agent_analysis import current_analysis, evidence_view

        row = current_analysis(session, agent, req)
        if row is not None:
            return evidence_view(session, row)
    return latest_selection(session, req["requirement_id"], current_observation_version_id(get_llm_settings()),
                            requirement_version=req.get("current_version"))


def _ai_frequency(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    attr, what = FREQUENCY_KINDS[method.method_id]
    sel = current_evidence(session, req, agent)
    if sel is None:
        return _unscored(method, "No AI evidence selection exists for this requirement version. Extract observations "
                                 "and run evidence selection first.")
    cov = sel.coverage
    if not sel.coverage_complete:
        return _unscored(method, f"Extraction coverage is incomplete: {cov.get('extracted_comments', 0)} of "
                                 f"{cov.get('direct_comments', 0)} direct comments have validated observations.",
                         inputs={"selection_id": sel.selection_id, "coverage": cov},
                         limitations=["A partial count would understate demand, so no score is produced."])
    ids = list(getattr(sel, attr))
    n = len(ids)
    score = log_normalize(n)
    snap = sel.snapshot()
    basis = snap.get("evidence_basis", "taxonomy_selection")
    extra_refs = []
    if basis == "agent_ai_analysis":
        extra_refs = [{"table": "agent_evidence_analyses", "key": snap["analysis_id"], "record_id": snap["analysis_id"]}] + \
            [{"table": "agent_document_chunks", "key": d["chunk_id"], "record_id": d["document_id"]}
             for d in snap["document_refs"]]
    return ScoreResult(
        status=SCORED, score=score, method=method.method_id, method_version=method.version,
        inputs={"n": n, "N": REFERENCE_POPULATION_N, "counted": what, "normalization": NORMALIZATION,
                "calculation": f"100 * ln(1 + {n}) / ln(1 + {REFERENCE_POPULATION_N}) = {score}",
                "selection_id": sel.selection_id, "observation_version_id": sel.observation_version_id,
                "selector_version": sel.selector_version, "selection_method": sel.method, "coverage": cov,
                "evidence_counts": sel.counts(), "counter_evidence_ids": sel.counter_evidence_ids,
                "uncertain_evidence_ids": sel.uncertain_evidence_ids, "counted_feedback_ids": ids,
                "human_reviewed_matches": sel.human_reviewed, "machine_validated_matches": sel.machine_validated,
                "evidence_basis": basis, "analysis_id": snap.get("analysis_id"),
                "missing_information": snap.get("missing_information", []),
                "document_refs": snap.get("document_refs", []), "evidence_snapshot": snap},
        source_refs=[{"table": "evidence_selection_runs", "key": sel.selection_id, "record_id": sel.selection_id}]
                    + extra_refs + [{"table": "feedback", "key": k, "record_id": None} for k in ids],
        rationale=("Evidence identified by this agent's AI analysis. " if basis == "agent_ai_analysis" else
                   "Evidence from deterministic taxonomy matching (agent AI analysis not run for the current inputs). ")
                  + f"{n} distinct G60 comments with {what}; log-normalised against the fixed reference N = "
                  f"{REFERENCE_POPULATION_N}. Counter-evidence: {len(sel.counter_evidence_ids)} comments; "
                  f"uncertain: {len(sel.uncertain_evidence_ids)}.",
        facts=[f"{n} distinct G60 feedback IDs counted ({what}).",
               f"{len(sel.counter_evidence_ids)} comments are counter-evidence; "
               f"{len(sel.uncertain_evidence_ids)} are uncertain and not counted.",
               f"Coverage: {cov['extracted_comments']} of {cov['direct_comments']} direct comments extracted."],
        assumptions=["Observations are AI-identified and validated (feedback ID exists, excerpt is verbatim); "
                     f"{sel.human_reviewed} matches are human-reviewed, {sel.machine_validated} machine-validated only.",
                     "Logarithmic normalisation is a prototype methodology, not a validated BMW standard."],
        limitations=list(method.limitations),
    )


# ---------------------------------------------------------------------------
# DOCUMENTED_CUSTOM_METRIC - admin-approved, code-implemented metrics only
# ---------------------------------------------------------------------------


def _custom(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    from services.custom_metrics import evaluate_custom_metric

    return evaluate_custom_metric(session, method, req, agent)


# ---------------------------------------------------------------------------
# AI (disabled interface)
# ---------------------------------------------------------------------------


def _ai(session: Session, method: ScoringMethod, req: dict, agent: dict) -> ScoreResult:
    return _unscored(method, AI_UNAVAILABLE, limitations=["No external LLM calls are made in Part 5."])


METHODS: dict[str, ScoringMethod] = {m.method_id: m for m in [
    ScoringMethod("LOG_NORMALIZED_FEEDBACK_COUNT", "1.0", "Log-normalised unique feedback count", ("DATA_ANALYSIS",),
                  "Customer demand: distinct G60 feedback IDs linked to the requirement, log-normalised against all "
                  "distinct G60 comments.",
                  frozenset({"RES_G60_FEEDBACK", "RES_REQUIREMENTS"}), score_label="customer demand",
                  limitations=("Frequency is not severity or business impact.",
                               "Counts are from the supplied export (about 90% US), not market prevalence."),
                  fn=_log_feedback, formula="100 * ln(1 + n) / ln(1 + N), N = live distinct G60 comments"),
    ScoringMethod("HUMAN_SCORE", "1.0", "Human assessment score", ("HUMAN_ASSESSMENT",),
                  "A product manager scores the requirement 0-100 with a documented rationale.",
                  frozenset({"RES_HUMAN_ASSESSMENTS", "RES_REQUIREMENTS"}), score_label="human assessment",
                  limitations=("Subjective; review the rationale.",), fn=_human,
                  formula="score entered by a person (integer 0-100) with rationale"),
    ScoringMethod("EVIDENCE_COVERAGE", "1.1", "Evidence coverage", ("DATA_ANALYSIS", "AI_ANALYSIS"),
                  "Share of assigned evidence types that have at least one valid, traceable link for the requirement.",
                  frozenset({"RES_REQUIREMENTS"}), any_of_resources=frozenset(COVERAGE_TYPES) | frozenset(EXTRA_COVERAGE_TYPES),
                  score_label="evidence coverage (not impact)",
                  limitations=("Evidence coverage is not customer impact or business importance.",
                               "Supporting/contextual evidence is linked at need-category level."),
                  fn=_coverage, formula="100 * covered evidence types / assigned evidence types"),
    ScoringMethod("VALIDATED_MARKET_RELEVANCE", "1.0", "Validated market relevance", ("DATA_ANALYSIS",),
                  "Market-derived relevance - only when market mappings and data are validated; otherwise UNSCORED.",
                  frozenset({"RES_G60_FEEDBACK", "RES_SALES", "RES_REQUIREMENTS"}), score_label="market relevance",
                  limitations=("G60/G68 sales are combined.", "Country-to-sales-region mappings are not validated."),
                  fn=_market, formula="not implemented - always UNSCORED until prerequisites are validated"),
    *[ScoringMethod(mid, "1.0", label, ("AI_ANALYSIS",), desc, frozenset({"RES_AI_OBSERVATIONS", "RES_REQUIREMENTS"}),
                    score_label=label.lower(), fn=_ai_frequency, formula=NORMALIZATION["formula"] + " with N = 4000",
                    limitations=("Frequency is not severity or business impact.",
                                 "Counts come from AI observations validated against the database; review uncertain "
                                 "and counter-evidence before deciding.",
                                 "Counts are from the supplied export (about 90% US), not market prevalence."))
      for mid, label, desc in [
          ("COMPLAINT_FREQUENCY", "Complaint frequency",
           "Distinct G60 comments with a validated complaint matched to the requirement, log-normalised (N = 4000)."),
          ("CUSTOMER_REQUEST_FREQUENCY", "Customer request frequency",
           "Distinct G60 comments with a validated improvement request matched to the requirement, log-normalised."),
          ("COMBINED_CUSTOMER_DEMAND", "Combined customer demand",
           "Distinct G60 comments with a validated complaint OR improvement request (union, no double counting), "
           "log-normalised."),
      ]],
    ScoringMethod("DOCUMENTED_CUSTOM_METRIC", "1.0", "Documented custom metric", ("DATA_ANALYSIS",),
                  "A code-implemented metric that an administrator has documented and approved. Unapproved -> UNSCORED.",
                  frozenset({"RES_REQUIREMENTS"}), score_label="custom metric", fn=_custom,
                  formula="defined by the approved metric (see scoring_method_versions)",
                  limitations=("Only administrator-approved metrics are computed; free-form formulas are never executed.",)),
    ScoringMethod("AI_ASSESSMENT", "0.1", "AI assessment (disabled)", ("AI_ANALYSIS",),
                  "Future LLM-based assessment. Disabled: requires explicit authorisation, API key, cost estimate, "
                  "confirmation and evidence validation.",
                  frozenset({"RES_REQUIREMENTS"}), score_label="AI assessment", limitations=(AI_UNAVAILABLE,), fn=_ai,
                  selectable=False, formula="disabled - a model never chooses a numerical score"),
]}


def get_method(method_id: str) -> ScoringMethod:
    if method_id not in METHODS:
        raise ValueError(f"Unsupported scoring method {method_id!r}")
    return METHODS[method_id]


def check_compatibility(execution_method: str, scoring_method: str, resources: list[str]) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors make the configuration invalid."""
    from services.agent_resources import RESOURCES

    errors, warnings = [], []
    if scoring_method not in METHODS:
        return [f"Unsupported scoring method '{scoring_method}'."], warnings
    m = METHODS[scoring_method]
    if execution_method not in m.execution_methods:
        errors.append(f"Scoring method '{m.label}' requires execution method "
                      f"{' or '.join(m.execution_methods)}, not {execution_method}.")
    unknown = [r for r in resources if r not in RESOURCES]
    errors += [f"Unknown resource '{r}'." for r in unknown]
    missing = sorted(m.required_resources - set(resources))
    if missing:
        errors.append(f"'{m.label}' requires resources: {', '.join(RESOURCES[r].name for r in missing)}.")
    if m.any_of_resources and not (set(resources) & m.any_of_resources):
        errors.append(f"'{m.label}' needs at least one of: "
                      + ", ".join(RESOURCES[r].name for r in sorted(m.any_of_resources)) + ".")
    for r in resources:
        if r in RESOURCES and scoring_method not in RESOURCES[r].supported_analysis_methods:
            errors.append(f"Resource '{RESOURCES[r].name}' cannot be used by '{m.label}'.")
    if scoring_method == "AI_ASSESSMENT":
        warnings.append(AI_UNAVAILABLE)
    if scoring_method in FREQUENCY_KINDS:
        warnings.append("The AI identifies evidence only; the score is calculated in Python from validated counts.")
    if scoring_method == "VALIDATED_MARKET_RELEVANCE" and not all(MARKET_VALIDATION.values()):
        warnings.append("Market prerequisites are not validated - this agent will return UNSCORED.")
    if scoring_method == "EVIDENCE_COVERAGE":
        warnings.append("This agent measures evidence coverage, not customer impact or business importance.")
    return errors, warnings


def methods_for(execution_method: str) -> list[str]:
    return [mid for mid, m in METHODS.items() if execution_method in m.execution_methods and m.selectable]


def recompute_score(method_id: str, inputs: dict) -> float | None:
    """Recalculate a stored score from its stored inputs only (reproducibility
    check). Returns None for methods whose score is a stored human value."""
    if method_id in FREQUENCY_KINDS:
        return log_normalize(int(inputs["n"]), int(inputs["N"]))
    if method_id == "LOG_NORMALIZED_FEEDBACK_COUNT":
        n, pop = int(inputs["unique_g60_feedback_ids"]), int(inputs["reference_population"])
        return min(round(100 * math.log1p(n) / math.log1p(pop), 2), 100.0)
    if method_id == "EVIDENCE_COVERAGE":
        return round(100 * inputs["covered_evidence_types"] / inputs["assigned_evidence_types"], 2)
    return None


def sync_method_versions(session: Session) -> int:
    """Register every method version in ``scoring_method_versions`` (insert
    only; existing rows are never modified)."""
    import json
    from datetime import datetime, timezone

    from core.evidence_models import ScoringMethodVersion

    added = 0
    for m in METHODS.values():
        if session.get(ScoringMethodVersion, (m.method_id, m.version)) is None:
            session.add(ScoringMethodVersion(
                method_id=m.method_id, version=m.version, label=m.label, execution_methods=",".join(m.execution_methods),
                formula=m.formula or m.description,
                normalization_json=json.dumps(NORMALIZATION if m.method_id in FREQUENCY_KINDS else {}),
                registered_at=datetime.now(timezone.utc).replace(tzinfo=None)))
            added += 1
    session.flush()
    return added


def run_method(session: Session, req: dict, agent: dict) -> ScoreResult:
    """Execute an agent snapshot (dict with agent_id, name, execution_method,
    scoring_method, resources) against a requirement dict."""
    method = get_method(agent["scoring_method"])
    errors, _ = check_compatibility(agent["execution_method"], agent["scoring_method"], agent["resources"])
    if errors:
        return _unscored(method, "Agent configuration is invalid: " + " ".join(errors))
    result = method.fn(session, method, req, agent)
    if result.score is not None and not 0 <= result.score <= 100:
        raise ValueError(f"{method.method_id} produced out-of-range score {result.score}")
    return result
