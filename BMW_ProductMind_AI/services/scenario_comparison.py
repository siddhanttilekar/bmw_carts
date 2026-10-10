"""Scenario comparison (Part 6).

Compares two of: a saved run, a saved scenario, or an unsaved simulation.
Discloses differences in the underlying inputs (agent scores/assessments,
agent set and versions, requirement versions, review status) and only
attributes rank changes to weights when those inputs are held constant.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.decision_models import ScenarioComparison
from core.prioritization_models import RequirementPriorityScore
from services.critical_issues import flags_for
from services.decision_grounding import bind_run, run_agents
from services.dynamic_prioritization import agent_results
from services.what_if_simulator import SimulationResult, load_scenario


class Side(BaseModel):
    kind: str  # run | scenario | simulation
    ref: str
    label: str
    agents: dict[str, dict]  # agent_id -> {name, weight_bp, version}
    requirements: dict[str, dict]  # rid -> {title, status, final, rank, requirement_version, review_status, scores{aid: score}}


def side_from_run(session: Session, run_id: str) -> Side:
    run = bind_run(session, run_id)
    agents = {a["agent_id"]: {"name": a["name"], "weight_bp": a["weight_bp"], "version": a["configuration_version"]}
              for a in run_agents(run)}
    reqs = {}
    for r in session.scalars(select(RequirementPriorityScore).where(RequirementPriorityScore.run_id == run_id)):
        reqs[r.requirement_id] = {"title": r.requirement_title, "status": r.priority_status, "final": r.final_score,
                                  "rank": r.rank, "requirement_version": r.requirement_version,
                                  "review_status": r.review_status,
                                  "scores": {x["agent_id"]: x["score"] for x in agent_results(session, run_id, r.requirement_id)}}
    return Side(kind="run", ref=run_id, label=f"Run {run_id} (profile {run.profile_id} v{run.profile_version})",
                agents=agents, requirements=reqs)


def side_from_simulation(session: Session, sim: SimulationResult, kind: str = "simulation", ref: str = "unsaved",
                         label: str | None = None) -> Side:
    base = side_from_run(session, sim.base_run_id)
    agents = {c["agent_id"]: {"name": c["name"], "weight_bp": c["weight_bp"],
                              "version": base.agents.get(c["agent_id"], {}).get("version"), "basis": c["basis"]}
              for c in sim.configuration}
    reqs = {}
    for row in sim.rows:
        b = base.requirements.get(row.requirement_id, {})
        reqs[row.requirement_id] = {"title": row.title, "status": row.simulated_status, "final": row.simulated_final,
                                    "rank": row.simulated_rank, "requirement_version": b.get("requirement_version"),
                                    "review_status": b.get("review_status"),
                                    "scores": {c["agent_id"]: c["score"] for c in row.contributions}}
    return Side(kind=kind, ref=ref, label=label or f"Simulation on run {sim.base_run_id}", agents=agents, requirements=reqs)


def side_from_scenario(session: Session, scenario_id: str) -> Side:
    sim = load_scenario(session, scenario_id)
    return side_from_simulation(session, sim, kind="scenario", ref=scenario_id, label=f"Scenario {scenario_id}")


def compare_sides(session: Session, left: Side, right: Side) -> dict:
    shared_agents = set(left.agents) & set(right.agents)
    config_diff = {
        "only_left": [left.agents[a]["name"] for a in sorted(set(left.agents) - set(right.agents))],
        "only_right": [right.agents[a]["name"] for a in sorted(set(right.agents) - set(left.agents))],
        "weight_changes": [{"agent": left.agents[a]["name"], "left_percent": left.agents[a]["weight_bp"] / 100,
                            "right_percent": right.agents[a]["weight_bp"] / 100}
                           for a in sorted(shared_agents) if left.agents[a]["weight_bp"] != right.agents[a]["weight_bp"]],
        "version_changes": [{"agent": left.agents[a]["name"], "left": left.agents[a]["version"], "right": right.agents[a]["version"]}
                            for a in sorted(shared_agents)
                            if None not in (left.agents[a]["version"], right.agents[a]["version"])
                            and left.agents[a]["version"] != right.agents[a]["version"]],
    }
    shared_reqs = sorted(set(left.requirements) & set(right.requirements))
    rows, input_diffs, newly_incomplete = [], [], []
    for rid in shared_reqs:
        l, r = left.requirements[rid], right.requirements[rid]
        rows.append({"requirement_id": rid, "title": l["title"], "left_status": l["status"], "left_final": l["final"],
                     "left_rank": l["rank"], "right_status": r["status"], "right_final": r["final"], "right_rank": r["rank"],
                     "rank_change": (l["rank"] - r["rank"]) if None not in (l["rank"], r["rank"]) else None,
                     "critical_flags": flags_for(session, rid)})
        if l["status"] == "COMPLETE" and r["status"] != "COMPLETE":
            newly_incomplete.append(rid)
        for a in shared_agents:
            if l["scores"].get(a) != r["scores"].get(a):
                input_diffs.append(f"{rid}: '{left.agents[a]['name']}' score {l['scores'].get(a)} vs {r['scores'].get(a)}")
        if None not in (l["requirement_version"], r["requirement_version"]) and l["requirement_version"] != r["requirement_version"]:
            input_diffs.append(f"{rid}: requirement version v{l['requirement_version']} vs v{r['requirement_version']}")
        if l["review_status"] != r["review_status"]:
            input_diffs.append(f"{rid}: review status {l['review_status']} vs {r['review_status']}")
    if input_diffs:
        attribution = ("Rank changes cannot be attributed to weights alone: the underlying inputs differ "
                       f"({len(input_diffs)} differences disclosed).")
    elif config_diff["only_left"] or config_diff["only_right"] or config_diff["version_changes"]:
        attribution = "Differences reflect a different agent set or agent versions, not only weights."
    elif config_diff["weight_changes"]:
        attribution = "Agent scores and requirement inputs are identical; differences are attributable to weight changes."
    else:
        attribution = "The two configurations are equivalent for the compared requirements."
    limitations = []
    for side in (left, right):
        inc = [rid for rid, v in side.requirements.items() if v["status"] != "COMPLETE"]
        if inc:
            limitations.append(f"{side.label}: {len(inc)} incomplete (unranked) requirements.")
    if set(left.requirements) != set(right.requirements):
        limitations.append("The two sides cover different requirement sets; only shared requirements are compared.")
    return {"left": {"kind": left.kind, "ref": left.ref, "label": left.label},
            "right": {"kind": right.kind, "ref": right.ref, "label": right.label},
            "requirements_compared": shared_reqs, "agent_configuration_differences": config_diff, "rows": rows,
            "newly_incomplete": newly_incomplete, "input_differences": input_diffs, "attribution": attribution,
            "limitations": limitations}


def save_comparison(session: Session, result: dict, created_by: str) -> str:
    if not (created_by or "").strip():
        raise ValueError("A user identifier is required.")
    cid = f"CMP-{uuid.uuid4().hex[:10].upper()}"
    session.add(ScenarioComparison(comparison_id=cid, left_kind=result["left"]["kind"], left_id=result["left"]["ref"],
                                   right_kind=result["right"]["kind"], right_id=result["right"]["ref"],
                                   result_json=json.dumps(result, default=str), created_by=created_by.strip(),
                                   created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
    session.flush()
    return cid
