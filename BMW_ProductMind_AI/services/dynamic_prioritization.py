"""Dynamic, agent-based prioritization engine (Part 5).

    Priority Score = SUM(agent_score * agent_weight / 100)

computed only when (1) the profile's active weights total 100 %, (2) every
active agent returned a SCORED result, (3) every score carries source
references and (4) the configuration is valid. Otherwise the requirement is
INCOMPLETE with final score NULL - missing weight is never redistributed.
Only COMPLETE requirements are ranked (competition ranking: 1, 2, 2, 4).

Each run stores a snapshot of the profile version, agent configuration
versions, weights and method versions, plus every agent-level result, so it
remains reproducible after agents or profiles change. A priority score never
changes a requirement's review status.

2.0: the run also freezes the evidence used by AI_ANALYSIS agents
(``evidence_snapshot_json``: selection ID, observation version with model /
prompt / taxonomy versions, validated feedback-ID lists, normalisation). The
same snapshot always yields the same score (``verify_reproducibility``);
``reanalyze_with_new_evidence`` creates a NEW run and a comparison - the old
run is never modified.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.prioritization_models import (
    AgentExecutionResult,
    PriorityAssessment,
    PrioritizationRun,
    RequirementPriorityScore,
)
from core.requirement_models import VehicleRequirement
from services.agent_configuration import get_agent, version_snapshot
from services.agent_explanations import build_explanation
from services.agent_scoring import METHODS, NORMALIZATION, UNSCORED, ScoreResult, recompute_score, run_method, sync_method_versions
from services.agent_weights import weighted_contribution
from services.critical_issues import flags_for
from services.prioritization_profiles import get_profile, validate_profile, weights_for
from services.requirement_review import requirement_as_dict


class PrioritizationError(ValueError):
    pass


class NoSavedRequirements(PrioritizationError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Human assessments
# ---------------------------------------------------------------------------


def record_human_assessment(session: Session, requirement_id: str, agent_id: str, score, rationale: str,
                            assessor_id: str, evidence_refs: list[str] | None = None) -> PriorityAssessment:
    """Stores a human score (0-100, integer) with rationale. Supersedes the
    assessor-agnostic previous assessment for the same requirement/agent."""
    if session.get(VehicleRequirement, requirement_id) is None:
        raise PrioritizationError(f"Unknown requirement {requirement_id!r}")
    agent = get_agent(session, agent_id)
    if agent.deleted or agent.scoring_method != "HUMAN_SCORE":
        raise PrioritizationError(f"Agent '{agent.name}' does not accept human assessments.")
    try:
        value = Decimal(str(score))
    except Exception as exc:  # noqa: BLE001
        raise PrioritizationError("Score must be a number.") from exc
    if value != value.to_integral_value() or not 0 <= value <= 100:
        raise PrioritizationError("Score must be a whole number between 0 and 100.")
    if not (rationale or "").strip():
        raise PrioritizationError("A rationale is required for every human assessment.")
    if not (assessor_id or "").strip():
        raise PrioritizationError("An assessor identifier is required.")
    refs = evidence_refs or []
    if refs:
        from services.critical_issues import _check_refs

        try:
            _check_refs(session, requirement_id, refs)
        except ValueError as exc:
            raise PrioritizationError(str(exc)) from exc
    for old in session.scalars(select(PriorityAssessment).where(PriorityAssessment.requirement_id == requirement_id,
                                                                PriorityAssessment.agent_id == agent_id,
                                                                PriorityAssessment.superseded.is_(False))):
        old.superseded = True
    a = PriorityAssessment(assessment_id=f"ASM-{uuid.uuid4().hex[:10].upper()}", requirement_id=requirement_id,
                           agent_id=agent_id, agent_version=agent.configuration_version, score=int(value),
                           rationale=rationale.strip(), assessor_id=assessor_id.strip(),
                           evidence_refs_json=json.dumps(refs), superseded=False, assessment_timestamp=_now())
    session.add(a)
    session.flush()
    return a


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _profile_snapshot(session: Session, profile_id: str) -> tuple[dict, list[dict]]:
    p = get_profile(session, profile_id)
    agents = []
    for w in weights_for(session, profile_id):
        snap = version_snapshot(session, w.agent_id, w.agent_version)
        snap.setdefault("context", None)
        snap.setdefault("custom_metric_id", None)
        if "RES_UPLOADED_DOCUMENTS" in snap.get("resources", []):  # uploads present at run time are frozen here
            from services.agent_documents import agent_documents

            snap["documents"] = [{"document_id": d["document_id"], "sha256": d["sha256"], "file_name": d["file_name"]}
                                 for d in agent_documents(session, w.agent_id)]
        agents.append({**snap, "weight_bp": w.weight_bp, "configuration_version": w.agent_version})
    return {"profile_id": p.profile_id, "profile_name": p.profile_name, "profile_version": p.current_version,
            "illustrative": p.is_illustrative, "agents": agents}, agents


def _observation_version_info(session: Session, version_id: str | None) -> dict | None:
    from core.evidence_models import ObservationVersion

    v = session.get(ObservationVersion, version_id) if version_id else None
    return None if v is None else {"version_id": v.version_id, "model": v.model, "model_version": v.model_version,
                                   "prompt_version": v.prompt_version, "taxonomy_version": v.taxonomy_version}


def run_prioritization(session: Session, profile_id: str, created_by: str,
                       requirement_ids: list[str] | None = None, based_on_run_id: str | None = None) -> PrioritizationRun:
    if not (created_by or "").strip():
        raise PrioritizationError("A user identifier is required.")
    check = validate_profile(session, profile_id)
    if not check.valid:
        raise PrioritizationError("Profile cannot be executed: " + " ".join(check.errors))
    stmt = select(VehicleRequirement).order_by(VehicleRequirement.created_at)
    if requirement_ids:
        stmt = stmt.where(VehicleRequirement.requirement_id.in_(requirement_ids))
    requirements = list(session.scalars(stmt))
    if requirement_ids and len(requirements) != len(set(requirement_ids)):
        raise PrioritizationError("Unknown requirement IDs in the selection.")
    if not requirements:
        raise NoSavedRequirements("No saved requirements yet. Create a requirement in the Vehicle Requirements page.")

    sync_method_versions(session)
    snapshot, agents = _profile_snapshot(session, profile_id)
    snapshot["normalization"] = NORMALIZATION
    run = PrioritizationRun(
        run_id=f"RUN-{_now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6].upper()}", profile_id=profile_id,
        profile_version=snapshot["profile_version"], profile_snapshot_json=json.dumps(snapshot, default=str),
        method_versions_json=json.dumps({a["scoring_method"]: METHODS[a["scoring_method"]].version for a in agents}),
        requirement_ids_json=json.dumps([r.requirement_id for r in requirements]), created_by=created_by.strip(),
        created_at=_now(), based_on_run_id=based_on_run_id)
    session.add(run)
    session.flush()

    finals: list[tuple[str, Decimal | None]] = []
    evidence: dict[str, dict] = {}
    for req in requirements:
        rd = requirement_as_dict(session, req)
        total, missing = Decimal("0"), []
        for agent in agents:
            result: ScoreResult = run_method(session, rd, agent)
            frozen = result.inputs.pop("evidence_snapshot", None)
            if frozen is not None:
                evidence[req.requirement_id] = {
                    **frozen, "requirement_version": req.current_version,
                    "observation_version": _observation_version_info(session, frozen.get("observation_version_id"))}
            if result.status != UNSCORED and not result.source_refs:  # provenance is mandatory
                result = result.model_copy(update={"status": UNSCORED, "score": None,
                                                   "rationale": "Score discarded: no source references (provenance)."})
            contribution = weighted_contribution(result.score, agent["weight_bp"]) if result.status != UNSCORED else None
            if contribution is None:
                missing.append(agent["name"])
            else:
                total += contribution
            session.add(AgentExecutionResult(
                run_id=run.run_id, requirement_id=req.requirement_id, agent_id=agent["agent_id"],
                agent_version=agent["configuration_version"], agent_name=agent["name"],
                scoring_method=result.method, scoring_version=result.method_version, status=result.status,
                score=result.score, weight_bp=agent["weight_bp"],
                weighted_contribution=float(contribution) if contribution is not None else None,
                inputs_json=json.dumps(result.inputs, default=str), source_refs_json=json.dumps(result.source_refs),
                rationale=result.rationale, limitations_json=json.dumps(result.limitations),
                explanation_json=json.dumps(build_explanation(result, agent, agent["weight_bp"],
                                                              str(contribution) if contribution is not None else None),
                                            default=str),
                created_at=_now()))
        final = None if missing else total.quantize(Decimal("0.01"), ROUND_HALF_UP)
        finals.append((req.requirement_id, final))
        session.add(RequirementPriorityScore(
            run_id=run.run_id, requirement_id=req.requirement_id, requirement_title=req.title,
            review_status=req.review_status, requirement_version=req.current_version,
            priority_status="INCOMPLETE" if missing else "COMPLETE",
            final_score=float(final) if final is not None else None, rank=None, missing_agents_json=json.dumps(missing),
            critical_flags_json=json.dumps(flags_for(session, req.requirement_id)), created_at=_now()))
    session.flush()

    ranks = competition_ranks(finals)
    rows = {r.requirement_id: r for r in session.scalars(select(RequirementPriorityScore)
                                                         .where(RequirementPriorityScore.run_id == run.run_id))}
    for rid, rank in ranks.items():
        rows[rid].rank = rank
    run.complete_count = len(ranks)
    run.incomplete_count = len(finals) - len(ranks)
    run.evidence_snapshot_json = json.dumps(evidence, default=str)
    session.flush()
    return run


# ---------------------------------------------------------------------------
# Reproducibility and reanalysis (2.0)
# ---------------------------------------------------------------------------


def evidence_snapshot(session: Session, run_id: str) -> dict:
    run = session.get(PrioritizationRun, run_id)
    if run is None:
        raise PrioritizationError(f"Unknown run {run_id!r}")
    return json.loads(run.evidence_snapshot_json or "{}")


def verify_reproducibility(session: Session, run_id: str) -> dict:
    """Recompute every agent score from its stored inputs only, and every final
    score / rank from the stored scores and weights. No live data is read."""
    if session.get(PrioritizationRun, run_id) is None:
        raise PrioritizationError(f"Unknown run {run_id!r}")
    mismatches, checked = [], 0
    totals: dict[str, tuple[Decimal, bool]] = {}
    for r in session.scalars(select(AgentExecutionResult).where(AgentExecutionResult.run_id == run_id)):
        inputs = json.loads(r.inputs_json)
        if r.status != UNSCORED:
            expected = recompute_score(r.scoring_method, inputs)
            if expected is not None:
                checked += 1
                if abs(expected - (r.score or 0)) > 1e-9:
                    mismatches.append({"requirement_id": r.requirement_id, "agent": r.agent_name,
                                       "stored": r.score, "recomputed": expected})
        total, complete = totals.get(r.requirement_id, (Decimal("0"), True))
        if r.status == UNSCORED or r.score is None:
            totals[r.requirement_id] = (total, False)
        else:
            totals[r.requirement_id] = (total + weighted_contribution(r.score, r.weight_bp), complete)
    finals = []
    for row in session.scalars(select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == run_id)):
        total, complete = totals.get(row.requirement_id, (Decimal("0"), False))
        final = total.quantize(Decimal("0.01"), ROUND_HALF_UP) if complete else None
        finals.append((row.requirement_id, final))
        stored = Decimal(str(row.final_score)).quantize(Decimal("0.01")) if row.final_score is not None else None
        if stored != final:
            mismatches.append({"requirement_id": row.requirement_id, "agent": "FINAL", "stored": row.final_score,
                               "recomputed": float(final) if final is not None else None})
    ranks = competition_ranks(finals)
    for row in session.scalars(select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == run_id)):
        if ranks.get(row.requirement_id) != row.rank:
            mismatches.append({"requirement_id": row.requirement_id, "agent": "RANK", "stored": row.rank,
                               "recomputed": ranks.get(row.requirement_id)})
    return {"run_id": run_id, "reproducible": not mismatches, "agent_scores_checked": checked,
            "requirements_checked": len(finals), "mismatches": mismatches}


def compare_runs(session: Session, old_run_id: str, new_run_id: str) -> dict:
    def collect(run_id):
        scores = {r.requirement_id: r for r in session.scalars(
            select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == run_id))}
        agents = {}
        for a in session.scalars(select(AgentExecutionResult).where(AgentExecutionResult.run_id == run_id)):
            agents[(a.requirement_id, a.agent_id)] = a
        return scores, agents, evidence_snapshot(session, run_id)

    old_s, old_a, old_e = collect(old_run_id)
    new_s, new_a, new_e = collect(new_run_id)
    rows = []
    for rid in sorted(set(old_s) | set(new_s)):
        o, n = old_s.get(rid), new_s.get(rid)
        agent_rows = []
        for key in sorted({k for k in set(old_a) | set(new_a) if k[0] == rid}, key=lambda k: k[1]):
            oa, na = old_a.get(key), new_a.get(key)
            oi = json.loads(oa.inputs_json) if oa else {}
            ni = json.loads(na.inputs_json) if na else {}
            agent_rows.append({"agent": (na or oa).agent_name, "old_status": oa.status if oa else None,
                               "new_status": na.status if na else None, "old_score": oa.score if oa else None,
                               "new_score": na.score if na else None, "old_n": oi.get("n"), "new_n": ni.get("n")})
        oe, ne = old_e.get(rid, {}), new_e.get(rid, {})
        rows.append({
            "requirement_id": rid, "title": (n or o).requirement_title,
            "old_status": o.priority_status if o else None, "new_status": n.priority_status if n else None,
            "old_score": o.final_score if o else None, "new_score": n.final_score if n else None,
            "old_rank": o.rank if o else None, "new_rank": n.rank if n else None,
            "old_selection_id": oe.get("selection_id"), "new_selection_id": ne.get("selection_id"),
            "old_evidence_counts": oe.get("counts"), "new_evidence_counts": ne.get("counts"),
            "evidence_changed": oe.get("selection_id") != ne.get("selection_id")
                                or oe.get("counts") != ne.get("counts"),
            "agents": agent_rows})
    return {"old_run_id": old_run_id, "new_run_id": new_run_id, "requirements": rows}


def reanalyze_with_new_evidence(session: Session, run_id: str, actor: str) -> tuple[PrioritizationRun, dict]:
    """Refresh deterministic evidence selections from the latest validated
    observations, then score the same requirements with the same profile in a
    NEW run. Returns (new run, comparison with the old run)."""
    from core.config import get_llm_settings
    from services.ai_evidence_selector import run_selection
    from services.ai_feedback_analysis import current_observation_version_id

    old = session.get(PrioritizationRun, run_id)
    if old is None:
        raise PrioritizationError(f"Unknown run {run_id!r}")
    rids = json.loads(old.requirement_ids_json)
    version_id = current_observation_version_id(get_llm_settings())
    for rid in rids:
        if session.get(VehicleRequirement, rid) is not None:
            run_selection(session, rid, version_id, actor)
    new = run_prioritization(session, old.profile_id, actor,
                             [r for r in rids if session.get(VehicleRequirement, r) is not None], based_on_run_id=run_id)
    return new, compare_runs(session, run_id, new.run_id)


def competition_ranks(finals) -> dict[str, int]:
    """Rank only scored (non-None) entries: 1, 2, 2, 4 ... ties ordered by ID.
    Shared by official runs and Part 6 simulations."""
    ranked = sorted(((rid, f) for rid, f in finals if f is not None), key=lambda x: (-x[1], x[0]))
    out, previous, rank = {}, None, 0
    for position, (rid, f) in enumerate(ranked, start=1):
        if f != previous:
            rank, previous = position, f
        out[rid] = rank
    return out


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class RankingRow(BaseModel):
    requirement_id: str
    title: str
    review_status: str
    priority_status: str
    final_score: float | None
    rank: int | None
    missing_agents: list[str]
    critical_flags: list[dict]


def ranking(session: Session, run_id: str) -> list[RankingRow]:
    rows = session.scalars(select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == run_id)).all()
    out = [RankingRow(requirement_id=r.requirement_id, title=r.requirement_title, review_status=r.review_status,
                      priority_status=r.priority_status, final_score=r.final_score, rank=r.rank,
                      missing_agents=json.loads(r.missing_agents_json),
                      critical_flags=json.loads(r.critical_flags_json)) for r in rows]
    return sorted(out, key=lambda r: (r.rank is None, r.rank or 0, r.requirement_id))


def agent_results(session: Session, run_id: str, requirement_id: str) -> list[dict]:
    return [{"agent_id": r.agent_id, "agent_name": r.agent_name, "agent_version": r.agent_version,
             "scoring_method": r.scoring_method, "scoring_version": r.scoring_version, "status": r.status,
             "score": r.score, "weight_percent": r.weight_bp / 100, "weight_bp": r.weight_bp,
             "weighted_contribution": r.weighted_contribution, "inputs": json.loads(r.inputs_json),
             "source_refs": json.loads(r.source_refs_json), "rationale": r.rationale,
             "limitations": json.loads(r.limitations_json), "explanation": json.loads(r.explanation_json)}
            for r in session.scalars(select(AgentExecutionResult).where(
                AgentExecutionResult.run_id == run_id, AgentExecutionResult.requirement_id == requirement_id)
                .order_by(AgentExecutionResult.result_id))]


def list_runs(session: Session, profile_id: str | None = None) -> list[dict]:
    stmt = select(PrioritizationRun).order_by(PrioritizationRun.created_at.desc())
    if profile_id:
        stmt = stmt.where(PrioritizationRun.profile_id == profile_id)
    return [{"run_id": r.run_id, "profile_id": r.profile_id, "profile_version": r.profile_version,
             "profile_name": json.loads(r.profile_snapshot_json)["profile_name"], "complete": r.complete_count,
             "incomplete": r.incomplete_count, "requirements": len(json.loads(r.requirement_ids_json)),
             "created_by": r.created_by, "created_at": r.created_at} for r in session.scalars(stmt)]


def latest_run_for(session: Session, requirement_id: str, profile_id: str | None = None) -> str | None:
    stmt = (select(PrioritizationRun.run_id).join(RequirementPriorityScore, RequirementPriorityScore.run_id == PrioritizationRun.run_id)
            .where(RequirementPriorityScore.requirement_id == requirement_id))
    if profile_id:
        stmt = stmt.where(PrioritizationRun.profile_id == profile_id)
    return session.scalar(stmt.order_by(PrioritizationRun.created_at.desc()).limit(1))
