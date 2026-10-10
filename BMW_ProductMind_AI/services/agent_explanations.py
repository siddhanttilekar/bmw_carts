"""Structured, explainable agent results (Part 5).

Every statement is typed:
  FACT        directly supported by source data / calculation inputs
  ASSESSMENT  human-entered judgment
  ASSUMPTION  unverified input or prototype methodology
  LIMITATION  missing or unreliable evidence
Explanations are built only from the scoring result and stored evidence - no
customer opinions are generated. Stored as JSON for the Part 6 assistant.
"""
from __future__ import annotations

from services.agent_scoring import ScoreResult, get_method


def build_explanation(result: ScoreResult, agent: dict, weight_bp: int, contribution: str | None) -> dict:
    method = get_method(result.method)
    items: list[dict] = []
    for fact in result.facts:
        items.append({"kind": "FACT", "text": fact})
    if result.source_refs:
        tables = sorted({r["table"] for r in result.source_refs})
        items.append({"kind": "FACT", "text": f"{len(result.source_refs)} source references ({', '.join(tables)}).",
                      "refs": [r["key"] for r in result.source_refs[:50]]})
    for a in result.assessments:
        items.append({"kind": "ASSESSMENT", "text": f"{a['assessor_id']} scored {a['score']}/100 on "
                                                    f"{a['timestamp'][:16]}: {a['rationale']}",
                      "refs": a.get("evidence_references", [])})
    for a in result.assumptions:
        items.append({"kind": "ASSUMPTION", "text": a})
    if result.status == "UNSCORED":
        items.append({"kind": "LIMITATION", "text": result.rationale})
    for lim in result.limitations + list(agent.get("limitations") or []):
        if {"kind": "LIMITATION", "text": lim} not in items:
            items.append({"kind": "LIMITATION", "text": lim})
    return {
        "agent_id": agent["agent_id"], "agent_name": agent["name"], "agent_version": agent["configuration_version"],
        "objective": agent["objective"], "status": result.status, "score": result.score,
        "score_label": method.score_label, "weight_percent": weight_bp / 100, "weighted_contribution": contribution,
        "methodology": {"scoring_method": method.method_id, "label": method.label, "version": method.version,
                        "execution_method": method.execution_method, "description": method.description,
                        "inputs": result.inputs},
        "rationale": result.rationale, "items": items,
    }


def summarize(explanations: list[dict]) -> dict[str, list[str]]:
    """Group all typed statements of a requirement's agent explanations."""
    out: dict[str, list[str]] = {"FACT": [], "ASSESSMENT": [], "ASSUMPTION": [], "LIMITATION": []}
    for e in explanations:
        for item in e["items"]:
            text = f"[{e['agent_name']}] {item['text']}"
            if text not in out[item["kind"]]:
                out[item["kind"]].append(text)
    return out
