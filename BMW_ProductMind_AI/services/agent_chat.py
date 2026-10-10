"""Agent-specific discussion (Part 6).

The assistant adopts one agent's perspective using ONLY that agent's
configuration as stored in the selected run snapshot (method, version,
resources) and its stored result. Agents are discovered dynamically from the
run - custom agents get the same treatment - and nothing is inferred from an
agent's name.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from core.prioritization_models import PriorityAssessment
from services.agent_resources import RESOURCES, list_resources
from services.agent_scoring import METHODS
from services.decision_grounding import AssistantResponse, bind_run, finalize, historical_notices, run_agents
from services.dynamic_prioritization import agent_results

TOPICS = ("why_score", "method", "evidence", "limitations")


def run_agent_options(session: Session, run_id: str) -> list[dict]:
    run = bind_run(session, run_id)
    return [{"agent_id": a["agent_id"], "name": a["name"], "scoring_method": a["scoring_method"],
             "execution_method": a["execution_method"], "weight_percent": a["weight_bp"] / 100,
             "configuration_version": a["configuration_version"]} for a in run_agents(run)]


def discuss_agent(session: Session, run_id: str, requirement_id: str, agent_id: str, topic: str = "why_score"
                  ) -> AssistantResponse:
    if topic not in TOPICS:
        raise ValueError(f"Topic must be one of {TOPICS}")
    run = bind_run(session, run_id, requirement_id)
    agent = next((a for a in run_agents(run) if a["agent_id"] == agent_id), None)
    resp = AssistantResponse(intent="agent_discussion", title="", requirement_id=requirement_id, run_id=run_id,
                             profile_id=run.profile_id, profile_version=run.profile_version, agent_id=agent_id,
                             notices=historical_notices(session, run, requirement_id))
    if agent is None:
        resp.title = "Agent not part of this run"
        resp.needs_clarification = True
        resp.add("LIMITATION", f"Agent {agent_id} was not used in run {run_id}. Select one of the run's agents.")
        resp.suggestions = [a["name"] for a in run_agents(run)]
        return resp
    result = next(r for r in agent_results(session, run_id, requirement_id) if r["agent_id"] == agent_id)
    method = METHODS[agent["scoring_method"]]
    resp.title = f"{agent['name']} (v{agent['configuration_version']}) on {requirement_id}"

    if topic in ("why_score", "method"):
        resp.add("FACT", f"This agent is configured with the scoring method '{method.label}' "
                         f"(v{result['scoring_version']}, execution {agent['execution_method']}); its behaviour comes from "
                         f"this configuration, not from its name. Objective as configured: {agent['objective']}")
        resp.add("FACT", "Assigned resources: " + ", ".join(RESOURCES[r].name for r in agent["resources"] if r in RESOURCES) + ".")
        counts = {r.resource_id: r.available_records for r in list_resources(session)}
        resp.add("FACT", "Resource availability (current data, not the run snapshot): "
                 + "; ".join(f"{RESOURCES[r].name}: {counts.get(r)} records" for r in agent["resources"] if r in RESOURCES) + ".")
        resp.add("FACT", f"Method: {method.description}")

    if topic == "why_score":
        if result["status"] == "SCORED":
            resp.add("FACT", f"Stored score {result['score']} × weight {result['weight_percent']}% = weighted contribution "
                             f"{result['weighted_contribution']} points (run {run_id}).")
            inputs = result["inputs"]
            if "unique_g60_feedback_ids" in inputs:
                resp.add("FACT", f"{inputs['unique_g60_feedback_ids']} distinct G60 comments linked as direct evidence; "
                                 f"reference population {inputs['reference_population']} distinct G60 comments; formula "
                                 f"{inputs['formula']}.")
            if "covered_evidence_types" in inputs:
                per = "; ".join(f"{v['label']}: {v['valid_links']}" for v in inputs["per_type"].values())
                resp.add("FACT", f"Evidence coverage {inputs['covered_evidence_types']} of {inputs['assigned_evidence_types']} "
                                 f"assigned types ({per}). This measures coverage, not customer impact.")
            if "assessment_id" in inputs:
                a = session.get(PriorityAssessment, inputs["assessment_id"])
                if a is None:
                    resp.add("LIMITATION", "The stored assessment record could not be found.")
                else:
                    resp.add("ASSESSMENT", f"{a.assessor_id} scored {a.score}/100 on {a.assessment_timestamp:%Y-%m-%d %H:%M}. "
                                           f"Rationale: {a.rationale.strip() or 'No rationale was provided.'}",
                             refs=[a.assessment_id])
                    if a.superseded:
                        resp.add("LIMITATION", "This assessment has since been superseded by a newer one; it was the "
                                               "current assessment when the run was executed.")
        else:
            resp.add("LIMITATION", f"UNSCORED: {result['rationale']}")
            if agent["scoring_method"] == "HUMAN_SCORE":
                resp.add("LIMITATION", "Missing: a human assessment (score 0-100 with rationale) for this agent and "
                                       "requirement. Enter it in Agent Prioritization → Execution, then re-run.")
            elif result["inputs"].get("invalid_ids") or result["inputs"].get("non_g60_ids"):
                resp.add("LIMITATION", "Missing: valid G60 evidence. The requirement's evidence contains invalid or non-G60 IDs.")
            elif "prerequisites" in result["inputs"]:
                resp.add("LIMITATION", "Missing: validated market prerequisites "
                                       + ", ".join(k for k, ok in result["inputs"]["prerequisites"].items() if not ok) + ".")

    if topic in ("why_score", "evidence"):
        refs = [{"table": r["table"], "key": r["key"]} for r in result["source_refs"] if r["table"] != "evidence"]
        resp.evidence_refs = refs
        if refs:
            resp.add("FACT", f"{len(refs)} source references were stored with this result.", refs=[r["key"] for r in refs[:15]])
        else:
            resp.add("LIMITATION", "No source references were stored with this result.")

    if topic in ("why_score", "limitations"):
        for item in result["explanation"]["items"]:
            if item["kind"] == "ASSUMPTION":
                resp.add("ASSUMPTION", item["text"])
        for lim in result["limitations"]:
            resp.add("LIMITATION", lim)
    return finalize(session, resp)
