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
                        "execution_method": agent.get("execution_method", method.execution_method),
                        "description": method.description,
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


# ---------------------------------------------------------------------------
# 2.0: "show the calculation behind every score" (stored values only)
# ---------------------------------------------------------------------------


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:g}" if isinstance(x, float) else str(x))


def calculation_lines(result: dict) -> list[str]:
    """Human-readable calculation for one stored agent result (dict from
    ``dynamic_prioritization.agent_results``). Every number comes from the
    stored inputs / score / weight - nothing is recomputed by a model."""
    inp, score, w = result.get("inputs") or {}, result.get("score"), result.get("weight_percent")
    contribution = result.get("weighted_contribution")
    if result.get("status") != "SCORED":
        return [f"UNSCORED: {result.get('rationale', '')}", f"Weight: {_fmt(w)}% (not redistributed)"]
    lines: list[str] = []
    if "n" in inp and "N" in inp:
        lines += [f"Evidence: {inp['n']} unique comments ({inp.get('counted', 'validated evidence')})",
                  f"Normalization: 100 × ln(1 + {inp['n']}) / ln(1 + {inp['N']}) = {_fmt(score)} "
                  f"({inp.get('normalization', {}).get('version', 'log')}; not a percentage of customers)"]
    elif "unique_g60_feedback_ids" in inp:
        lines += [f"Evidence: {inp['unique_g60_feedback_ids']} unique G60 comments linked as direct evidence",
                  f"Normalization: 100 × ln(1 + {inp['unique_g60_feedback_ids']}) / ln(1 + {inp['reference_population']}) "
                  f"= {_fmt(score)}"]
    elif "covered_evidence_types" in inp:
        lines += [f"Coverage: 100 × {inp['covered_evidence_types']} / {inp['assigned_evidence_types']} evidence types "
                  f"= {_fmt(score)} (coverage, not impact)"]
    elif "assessment_id" in inp:
        lines += [f"Human assessment: {_fmt(score)} / 100 (assessment {inp['assessment_id']})"]
    elif "calculation" in inp:
        lines += [f"Custom metric {inp.get('custom_metric_id', '')}: {inp['calculation']}"]
    else:
        lines += [f"Score: {_fmt(score)}"]
    lines += [f"Score calculated by Python: {_fmt(score)}", f"Weight: {_fmt(w)}%",
              f"Weighted contribution: {_fmt(score)} × {_fmt(w)} / 100 = {_fmt(contribution)} (Python)"]
    return lines


def evidence_lines(result: dict) -> list[str]:
    """Validated evidence counts for AI_ANALYSIS results (empty otherwise)."""
    inp = result.get("inputs") or {}
    counts, cov = inp.get("evidence_counts"), inp.get("coverage")
    out = []
    if counts:
        out.append(f"Validated complaints: {counts['complaints']}; improvement requests: {counts['improvement_requests']}; "
                   f"unique complaint/request comments: {counts['combined_demand']}; positive: {counts['positive']}.")
        out.append(f"Counter-evidence: {counts['counter_evidence']} comments; uncertain (not counted): "
                   f"{counts['uncertain']}; rejected by reviewers: {counts['rejected_by_reviewer']}.")
    if cov:
        out.append(f"Extraction coverage: {cov.get('extracted_comments')} of {cov.get('direct_comments')} direct comments.")
    if "human_reviewed_matches" in inp:
        out.append(f"Evidence selection {inp.get('selection_id')}: {inp['human_reviewed_matches']} matches human-reviewed, "
                   f"{inp['machine_validated_matches']} machine-validated only.")
    return out
