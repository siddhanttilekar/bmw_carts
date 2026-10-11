"""DOCUMENTED_CUSTOM_METRIC (ProductMind AI 2.0).

A custom metric is usable only if BOTH hold:

1. it is implemented here in reviewed Python (``CUSTOM_METRICS``) - users can
   never supply a formula, SQL or code through the UI or agent instructions;
2. an administrator has approved it, recorded in ``scoring_method_versions``
   as ``CUSTOM:<metric_id>`` with approver, note and timestamp.

Otherwise the agent returns UNSCORED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.evidence_models import ScoringMethodVersion


@dataclass(frozen=True)
class CustomMetric:
    metric_id: str
    version: str
    label: str
    formula: str
    required_resources: frozenset[str]
    interpretation: str
    fn: Callable


def _document_passage_share(session: Session, metric: CustomMetric, req: dict, agent: dict):
    from services.agent_documents import keywords, search_agent_documents

    terms = keywords(req["title"], req["customer_need"])
    total, hits = search_agent_documents(session, agent["agent_id"], terms)
    if total == 0:
        return None, {"reason": "The agent has no active uploaded documents."}, []
    score = round(100 * len(hits) / total, 2)
    return score, {"matching_passages": len(hits), "total_passages": total, "search_terms": terms,
                   "calculation": f"100 * {len(hits)} / {total} = {score}"}, \
        [{"table": "agent_document_chunks", "key": h["chunk_id"], "record_id": h["document_id"]} for h in hits] \
        or [{"table": "agent_documents", "key": agent["agent_id"], "record_id": None}]


CUSTOM_METRICS: dict[str, CustomMetric] = {m.metric_id: m for m in [
    CustomMetric("DOCUMENT_PASSAGE_SHARE", "1.0", "Uploaded-document passage share",
                 "100 * passages mentioning >= 2 requirement key terms / all passages in the agent's active uploads",
                 frozenset({"RES_UPLOADED_DOCUMENTS"}),
                 "How much of the agent's own reference material discusses this requirement's topic. It is document "
                 "relevance, not customer demand or business value.", _document_passage_share),
]}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def approval(session: Session, metric_id: str) -> ScoringMethodVersion | None:
    m = CUSTOM_METRICS.get(metric_id)
    if m is None:
        return None
    row = session.get(ScoringMethodVersion, (f"CUSTOM:{metric_id}", m.version))
    return row if row is not None and row.approved_by else None


def approve_custom_metric(session: Session, metric_id: str, admin_id: str, note: str) -> ScoringMethodVersion:
    """Administrator approval (CLI / settings). Insert-only per version."""
    if metric_id not in CUSTOM_METRICS:
        raise ValueError(f"Unknown custom metric {metric_id!r}; only implemented metrics can be approved.")
    if not (admin_id or "").strip() or not (note or "").strip():
        raise ValueError("Administrator ID and an approval note are required.")
    m = CUSTOM_METRICS[metric_id]
    row = session.get(ScoringMethodVersion, (f"CUSTOM:{metric_id}", m.version))
    if row is None:
        row = ScoringMethodVersion(method_id=f"CUSTOM:{metric_id}", version=m.version, label=m.label,
                                   execution_methods="DATA_ANALYSIS", formula=m.formula,
                                   normalization_json=json.dumps({"interpretation": m.interpretation}),
                                   registered_at=_now())
        session.add(row)
    if row.approved_by is None:
        row.approved_by, row.approval_note, row.approved_at = admin_id.strip(), note.strip(), _now()
    session.flush()
    return row


def list_custom_metrics(session: Session) -> list[dict]:
    return [{"metric_id": m.metric_id, "version": m.version, "label": m.label, "formula": m.formula,
             "interpretation": m.interpretation, "required_resources": sorted(m.required_resources),
             "approved": approval(session, m.metric_id) is not None} for m in CUSTOM_METRICS.values()]


def evaluate_custom_metric(session: Session, method, req: dict, agent: dict):
    from services.agent_scoring import SCORED, ScoreResult, _unscored

    metric_id = agent.get("custom_metric_id")
    if not metric_id or metric_id not in CUSTOM_METRICS:
        return _unscored(method, "No documented custom metric is selected for this agent.")
    m = CUSTOM_METRICS[metric_id]
    appr = approval(session, metric_id)
    if appr is None:
        return _unscored(method, f"Custom metric '{m.label}' v{m.version} is not approved by an administrator.")
    missing = sorted(m.required_resources - set(agent["resources"]))
    if missing:
        return _unscored(method, f"Custom metric '{m.label}' requires resources: {', '.join(missing)}.")
    score, inputs, refs = m.fn(session, m, req, agent)
    if score is None:
        return _unscored(method, inputs.get("reason", "Metric could not be computed."), inputs=inputs)
    return ScoreResult(
        status=SCORED, score=score, method=method.method_id, method_version=method.version,
        inputs={**inputs, "custom_metric_id": metric_id, "custom_metric_version": m.version, "formula": m.formula,
                "approved_by": appr.approved_by, "approved_at": str(appr.approved_at)},
        source_refs=refs, rationale=f"{m.label}: {inputs.get('calculation', '')}. {m.interpretation}",
        facts=[inputs.get("calculation", "")], assumptions=[m.interpretation],
        limitations=list(method.limitations) + ["Uploaded documents are untrusted; only keyword matches are counted."])
