"""What-if simulator (Part 6).

Recomputes a saved run's whole requirement population under an alternative
configuration (change weights, drop agents, add already-configured agents,
switch to another profile's weights). Reuses Part 5 pieces: stored agent
results, ``run_method`` for agents that were not in the run, exact
basis-point arithmetic and the shared competition ranking.

Nothing is written unless ``save_scenario`` is called explicitly; saving never
touches profiles, runs or assessments. Results are SIMULATIONS, not official
priorities.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.decision_models import PrioritizationScenario
from core.prioritization_models import RequirementPriorityScore
from services.agent_configuration import agent_snapshot, agent_resources, get_agent
from services.agent_scoring import UNSCORED, run_method
from services.agent_weights import WeightEntry, percent_to_bp, validate_weights, weighted_contribution
from services.critical_issues import flags_for
from services.decision_grounding import bind_run, run_agents
from services.dynamic_prioritization import agent_results, competition_ranks
from services.prioritization_profiles import weights_for
from services.requirement_review import get_requirement, requirement_as_dict

SIM_NOTE = "SIMULATION - hypothetical result, not a saved official priority."


class SimulationError(ValueError):
    pass


class SimRow(BaseModel):
    requirement_id: str
    title: str
    original_status: str
    original_final: float | None
    original_rank: int | None
    simulated_status: str
    simulated_final: float | None
    simulated_rank: int | None
    rank_change: int | None
    contributions: list[dict]  # agent, original, simulated, basis
    missing_agents: list[str]
    critical_flags: list[dict]


class SimulationResult(BaseModel):
    base_run_id: str
    base_profile_id: str
    base_profile_version: int
    configuration: list[dict]  # agent_id, name, weight_bp, basis
    rows: list[SimRow]
    notes: list[str]


def simulate(session: Session, base_run_id: str, weights: dict[str, object]) -> SimulationResult:
    """``weights``: {agent_id: percent} for the hypothetical active agents."""
    run = bind_run(session, base_run_id)
    in_run = {a["agent_id"]: a for a in run_agents(run)}
    bp = {}
    for aid, pct in weights.items():
        try:
            bp[aid] = percent_to_bp(pct)
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc
    entries = []
    config = []
    for aid, value in bp.items():
        if aid in in_run:
            config.append({"agent_id": aid, "name": in_run[aid]["name"], "weight_bp": value,
                           "basis": f"stored result from run {base_run_id} (agent v{in_run[aid]['configuration_version']})"})
            entries.append(WeightEntry(agent_id=in_run[aid]["name"], weight_bp=value))
        else:
            agent = get_agent(session, aid)
            if agent.deleted:
                raise SimulationError(f"Agent '{agent.name}' has been removed and cannot be added.")
            config.append({"agent_id": aid, "name": agent.name, "weight_bp": value,
                           "basis": f"computed now from current data (agent v{agent.configuration_version}; not in base run)"})
            entries.append(WeightEntry(agent_id=agent.name, weight_bp=value, enabled=agent.enabled))
    check = validate_weights(entries)
    if not check.valid:
        raise SimulationError(" ".join(check.errors))

    population = json.loads(run.requirement_ids_json)
    originals = {r.requirement_id: r for r in session.scalars(
        select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == base_run_id))}
    added = {c["agent_id"]: {**agent_snapshot(get_agent(session, c["agent_id"]), agent_resources(session, c["agent_id"]))}
             for c in config if c["agent_id"] not in in_run}

    computed: dict[str, dict] = {}
    finals = []
    for rid in population:
        stored = {r["agent_id"]: r for r in agent_results(session, base_run_id, rid)}
        req_dict = requirement_as_dict(session, get_requirement(session, rid)) if added else None
        contributions, total, missing = [], Decimal("0"), []
        for c in config:
            aid = c["agent_id"]
            if aid in stored:
                status, score = stored[aid]["status"], stored[aid]["score"]
            else:
                result = run_method(session, req_dict, added[aid])
                status, score = (result.status, result.score) if result.source_refs or result.status == UNSCORED \
                    else (UNSCORED, None)
            original = stored.get(aid, {}).get("weighted_contribution")
            if status == UNSCORED or score is None:
                missing.append(c["name"])
                sim = None
            else:
                sim = weighted_contribution(score, c["weight_bp"])
                total += sim
            contributions.append({"agent_id": aid, "agent": c["name"], "score": score, "original_contribution": original,
                                  "simulated_contribution": float(sim) if sim is not None else None, "basis": c["basis"]})
        final = None if missing else total.quantize(Decimal("0.01"), ROUND_HALF_UP)
        finals.append((rid, final))
        computed[rid] = {"contributions": contributions, "missing": missing, "final": final}
    ranks = competition_ranks(finals)
    rows = []
    for rid in population:
        o = originals[rid]
        sim_rank = ranks.get(rid)
        rows.append(SimRow(
            requirement_id=rid, title=o.requirement_title, original_status=o.priority_status, original_final=o.final_score,
            original_rank=o.rank, simulated_status="INCOMPLETE" if computed[rid]["missing"] else "COMPLETE",
            simulated_final=float(computed[rid]["final"]) if computed[rid]["final"] is not None else None,
            simulated_rank=sim_rank,
            rank_change=(o.rank - sim_rank) if (o.rank is not None and sim_rank is not None) else None,
            contributions=computed[rid]["contributions"], missing_agents=computed[rid]["missing"],
            critical_flags=flags_for(session, rid)))
    notes = [SIM_NOTE, "All requirements of the base run were re-ranked together under this configuration."]
    if added:
        notes.append("Agents not in the base run were scored now from current data; their inputs may differ from the "
                     "run's point in time.")
    return SimulationResult(base_run_id=base_run_id, base_profile_id=run.profile_id, base_profile_version=run.profile_version,
                            configuration=config, rows=sorted(rows, key=lambda r: (r.simulated_rank is None,
                                                                                   r.simulated_rank or 0, r.requirement_id)),
                            notes=notes)


def simulate_profile(session: Session, base_run_id: str, profile_id: str) -> SimulationResult:
    """Apply another profile's current weights to the base run's population."""
    from services.agent_weights import bp_to_percent

    weights = {w.agent_id: bp_to_percent(w.weight_bp) for w in weights_for(session, profile_id)}
    result = simulate(session, base_run_id, weights)
    result.notes.append(f"Weights taken from profile {profile_id} (current version).")
    return result


def current_weights(session: Session, run_id: str) -> dict[str, float]:
    run = bind_run(session, run_id)
    return {a["agent_id"]: a["weight_bp"] / 100 for a in run_agents(run)}


# ---------------------------------------------------------------------------
# Scenarios (explicit save only)
# ---------------------------------------------------------------------------


def save_scenario(session: Session, result: SimulationResult, name: str, created_by: str,
                  description: str = "") -> PrioritizationScenario:
    if not name.strip() or not (created_by or "").strip():
        raise SimulationError("Scenario name and user identifier are required.")
    scenario = PrioritizationScenario(
        scenario_id=f"SCN-{uuid.uuid4().hex[:10].upper()}", name=name.strip(), description=description.strip(),
        base_run_id=result.base_run_id, base_profile_id=result.base_profile_id,
        base_profile_version=result.base_profile_version, configuration_json=json.dumps(result.configuration),
        results_json=json.dumps([r.model_dump() for r in result.rows]), created_by=created_by.strip(),
        created_at=datetime.now(timezone.utc).replace(tzinfo=None))
    session.add(scenario)
    session.flush()
    return scenario


def list_scenarios(session: Session, base_run_id: str | None = None) -> list[dict]:
    stmt = select(PrioritizationScenario).order_by(PrioritizationScenario.created_at.desc())
    if base_run_id:
        stmt = stmt.where(PrioritizationScenario.base_run_id == base_run_id)
    return [{"scenario_id": s.scenario_id, "name": s.name, "base_run_id": s.base_run_id,
             "configuration": json.loads(s.configuration_json), "created_by": s.created_by, "created_at": s.created_at}
            for s in session.scalars(stmt)]


def load_scenario(session: Session, scenario_id: str) -> SimulationResult:
    s = session.get(PrioritizationScenario, scenario_id)
    if s is None:
        raise SimulationError(f"Unknown scenario {scenario_id!r}")
    return SimulationResult(base_run_id=s.base_run_id, base_profile_id=s.base_profile_id,
                            base_profile_version=s.base_profile_version, configuration=json.loads(s.configuration_json),
                            rows=[SimRow.model_validate(r) for r in json.loads(s.results_json)],
                            notes=[SIM_NOTE, f"Saved scenario '{s.name}' by {s.created_by} on {s.created_at:%Y-%m-%d %H:%M}."])
