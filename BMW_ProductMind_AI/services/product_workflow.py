"""Dashboard and continuous Product Prioritization workflow (read models +
orchestration). Streamlit-free so every step is testable.

Reuses the existing services only:
* domains   = the normalised customer-need taxonomy (18 categories)
* features  = original VFC level-2 topics mapped to each domain (the VFC
              clusters that requirements are generated from)
* evidence  = validated AI observations (counted as DISTINCT feedback IDs)
* scoring   = agent_scoring / dynamic_prioritization (unchanged formulas)
* AI        = agent_analysis (cached, approval-gated) via the shared client

Nothing here changes historical runs, profiles or scoring formulas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.ai_models import AIFeedbackExtraction, ClusterFeedbackLink, CustomerNeedCluster
from core.config import LLMSettings
from core.evidence_models import AIFeedbackObservation
from core.models import Feedback
from core.prioritization_models import PriorityAssessment
from core.requirement_models import VehicleRequirement
from services import ai_feedback_analysis as ai
from services.agent_weights import BP_TOTAL, percent_to_bp
from services.need_taxonomy import NEED_NAMES, TAXONOMY

COUNTED_TYPES = ("COMPLAINT", "IMPROVEMENT_REQUEST", "POSITIVE")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _version(settings: LLMSettings) -> str:
    return ai.current_observation_version_id(settings)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def dashboard_metrics(session: Session, settings: LLMSettings) -> dict:
    vid = _version(settings)
    unique = session.scalar(select(func.count(func.distinct(Feedback.original_feedback_id)))
                            .where(Feedback.vehicle_code == "G60")) or 0
    processed = session.scalar(select(func.count(func.distinct(AIFeedbackExtraction.original_feedback_id))).where(
        AIFeedbackExtraction.observation_version_id == vid, AIFeedbackExtraction.status == "succeeded")) or 0
    by_type = dict(session.execute(
        select(AIFeedbackObservation.observation_type, func.count(func.distinct(AIFeedbackObservation.original_feedback_id)))
        .where(AIFeedbackObservation.version_id == vid).group_by(AIFeedbackObservation.observation_type)).all())
    observations = session.scalar(select(func.count()).select_from(AIFeedbackObservation)
                                  .where(AIFeedbackObservation.version_id == vid)) or 0
    return {"unique_comments": unique, "ai_processed_comments": processed,
            "complaint_comments": by_type.get("COMPLAINT", 0), "request_comments": by_type.get("IMPROVEMENT_REQUEST", 0),
            "positive_comments": by_type.get("POSITIVE", 0), "observations": observations, "version_id": vid}


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def top_topics(session: Session, settings: LLMSettings, observation_type: str, limit: int = 6,
               domains: list[str] | None = None, examples: int = 3) -> list[dict]:
    """Vehicle systems (as described in the validated observations) ranked by
    DISTINCT comments with the given observation type. Each row carries real
    excerpts for traceability. 'unspecified' is excluded."""
    stmt = select(AIFeedbackObservation).where(AIFeedbackObservation.version_id == _version(settings),
                                               AIFeedbackObservation.observation_type == observation_type)
    if domains:
        stmt = stmt.where(AIFeedbackObservation.taxonomy_category.in_(domains))
    groups: dict[str, dict] = {}
    for o in session.scalars(stmt.order_by(AIFeedbackObservation.original_feedback_id)):
        key = _norm(o.vehicle_system)
        if key in ("", "unspecified", "none", "n/a"):
            continue
        g = groups.setdefault(key, {"label": o.vehicle_system.strip(), "comments": set(), "domains": {}, "examples": []})
        g["comments"].add(o.original_feedback_id)
        g["domains"][o.taxonomy_category] = g["domains"].get(o.taxonomy_category, 0) + 1
        if len(g["examples"]) < examples and o.original_feedback_id not in {e["feedback_id"] for e in g["examples"]}:
            g["examples"].append({"feedback_id": o.original_feedback_id, "excerpt": o.evidence_excerpt,
                                  "customer_need": o.customer_need})
    rows = sorted(groups.values(), key=lambda g: (-len(g["comments"]), g["label"].lower()))[:limit]
    return [{"topic": g["label"][:1].upper() + g["label"][1:], "comments": len(g["comments"]),
             "domain": NEED_NAMES.get(max(g["domains"], key=g["domains"].get), "Other"), "examples": g["examples"]}
            for g in rows]


def needs_by_domain(session: Session, settings: LLMSettings) -> list[dict]:
    """Distinct comments per taxonomy domain and observation type."""
    out: dict[str, dict] = {c.code: {"domain_code": c.code, "domain": c.name, "COMPLAINT": 0,
                                     "IMPROVEMENT_REQUEST": 0, "POSITIVE": 0} for c in TAXONOMY}
    for code, otype, n in session.execute(
        select(AIFeedbackObservation.taxonomy_category, AIFeedbackObservation.observation_type,
               func.count(func.distinct(AIFeedbackObservation.original_feedback_id)))
        .where(AIFeedbackObservation.version_id == _version(settings),
               AIFeedbackObservation.observation_type.in_(COUNTED_TYPES))
        .group_by(AIFeedbackObservation.taxonomy_category, AIFeedbackObservation.observation_type)):
        if code in out:
            out[code][otype] = n
    return [r for r in out.values() if r["COMPLAINT"] or r["IMPROVEMENT_REQUEST"] or r["POSITIVE"]]


# ---------------------------------------------------------------------------
# Step 1 - domains and features
# ---------------------------------------------------------------------------


def domains(session: Session, settings: LLMSettings) -> list[dict]:
    counts = {r["domain_code"]: r for r in needs_by_domain(session, settings)}
    out = []
    for c in TAXONOMY:
        r = counts.get(c.code, {})
        out.append({"code": c.code, "name": c.name, "description": c.description,
                    "complaint_comments": r.get("COMPLAINT", 0), "request_comments": r.get("IMPROVEMENT_REQUEST", 0)})
    return out


def features(session: Session, domain_codes: list[str]) -> list[dict]:
    """Original VFC level-2 topics mapped to the domains (existing clusters)."""
    if not domain_codes:
        return []
    rows = session.scalars(select(CustomerNeedCluster).where(
        CustomerNeedCluster.cluster_level == "vfc_category", CustomerNeedCluster.need_code.in_(domain_codes),
        CustomerNeedCluster.unique_feedback_count > 0)
        .order_by(CustomerNeedCluster.need_code, CustomerNeedCluster.unique_feedback_count.desc()))
    return [{"feature_id": c.cluster_id, "label": c.vfc_level2_name or c.label, "domain_code": c.need_code,
             "domain": NEED_NAMES.get(c.need_code, c.need_code), "unique_comments": c.unique_feedback_count,
             "requires_review": c.requires_review} for c in rows]


def scope_comment_ids(session: Session, domain_codes: list[str], feature_ids: list[str] | None) -> set[str]:
    """Distinct G60 feedback IDs in scope: the selected features' comments, or
    every comment of the selected domains when no feature is selected."""
    clusters = list(feature_ids or []) or [f"NEED-{d}" for d in domain_codes]
    if not clusters:
        return set()
    return set(session.scalars(select(ClusterFeedbackLink.original_feedback_id)
                               .where(ClusterFeedbackLink.cluster_id.in_(clusters))))


def evidence_preview(session: Session, settings: LLMSettings, domain_codes: list[str],
                     feature_ids: list[str] | None = None) -> dict:
    ids = scope_comment_ids(session, domain_codes, feature_ids)
    out = {"relevant_comments": len(ids), "complaint_comments": 0, "request_comments": 0, "positive_comments": 0}
    if not ids or not domain_codes:
        return out
    for otype, n in session.execute(
        select(AIFeedbackObservation.observation_type, func.count(func.distinct(AIFeedbackObservation.original_feedback_id)))
        .where(AIFeedbackObservation.version_id == _version(settings),
               AIFeedbackObservation.original_feedback_id.in_(ids),
               AIFeedbackObservation.taxonomy_category.in_(domain_codes))
        .group_by(AIFeedbackObservation.observation_type)):
        key = {"COMPLAINT": "complaint_comments", "IMPROVEMENT_REQUEST": "request_comments",
               "POSITIVE": "positive_comments"}.get(otype)
        if key:
            out[key] = n
    return out


# ---------------------------------------------------------------------------
# Step 2 - requirements and candidates
# ---------------------------------------------------------------------------


def _cluster_scope(domain_codes: list[str], feature_ids: list[str] | None) -> list[str]:
    return list(feature_ids or []) or [f"NEED-{d}" for d in domain_codes]


def requirements_in_scope(session: Session, domain_codes: list[str], feature_ids: list[str] | None) -> list[dict]:
    """Saved requirements whose need is in the selected domains (restricted to
    the selected features' clusters when features are chosen)."""
    stmt = select(VehicleRequirement).where(VehicleRequirement.need_code.in_(domain_codes or ["-"]))
    if feature_ids:
        stmt = stmt.where(VehicleRequirement.source_cluster_id.in_(feature_ids))
    labels = dict(session.execute(select(CustomerNeedCluster.cluster_id, CustomerNeedCluster.label)).all())
    return [{"requirement_id": r.requirement_id, "title": r.title, "customer_problem": r.customer_problem,
             "feature": labels.get(r.source_cluster_id, r.source_cluster_id), "domain": NEED_NAMES.get(r.need_code),
             "evidence_summary": r.evidence_summary, "status": r.review_status, "version": r.current_version,
             "cluster_id": r.source_cluster_id}
            for r in session.scalars(stmt.order_by(VehicleRequirement.updated_at.desc()))]


def candidate_needs(session: Session, settings: LLMSettings, domain_codes: list[str], feature_ids: list[str] | None,
                    limit: int = 12) -> list[dict]:
    """Customer needs in scope that have NO saved requirement yet - derived from
    existing clusters and validated observations (nothing is invented)."""
    saved_clusters = set(session.scalars(select(VehicleRequirement.source_cluster_id)))
    cluster_ids = _cluster_scope(domain_codes, feature_ids)
    if not feature_ids:  # domain level: also offer the domain's largest feature clusters
        cluster_ids += [f["feature_id"] for f in features(session, domain_codes)[:limit]]
    clusters = {c.cluster_id: c for c in session.scalars(select(CustomerNeedCluster).where(
        CustomerNeedCluster.cluster_id.in_(cluster_ids)))}
    vid = _version(settings)
    out = []
    for cid in dict.fromkeys(cluster_ids):
        c = clusters.get(cid)
        if c is None or cid in saved_clusters or c.unique_feedback_count == 0:
            continue
        ids = set(session.scalars(select(ClusterFeedbackLink.original_feedback_id).where(ClusterFeedbackLink.cluster_id == cid)))
        obs = list(session.scalars(select(AIFeedbackObservation).where(
            AIFeedbackObservation.version_id == vid, AIFeedbackObservation.original_feedback_id.in_(ids),
            AIFeedbackObservation.taxonomy_category == c.need_code,
            AIFeedbackObservation.observation_type.in_(("COMPLAINT", "IMPROVEMENT_REQUEST")))))
        needs: dict[str, set[str]] = {}
        for o in obs:
            needs.setdefault(_norm(o.customer_need), set()).add(o.original_feedback_id)
        top = sorted(needs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:3]
        out.append({"cluster_id": cid, "label": c.vfc_level2_name or c.label, "domain": NEED_NAMES.get(c.need_code),
                    "unique_comments": c.unique_feedback_count,
                    "complaint_comments": len({o.original_feedback_id for o in obs if o.observation_type == "COMPLAINT"}),
                    "request_comments": len({o.original_feedback_id for o in obs
                                             if o.observation_type == "IMPROVEMENT_REQUEST"}),
                    "top_needs": [k for k, _ in top]})
    return out[:limit]


def generate_draft(session: Session, cluster_id: str):
    """Rule-based draft from the cluster's evidence package (existing generator).
    The draft is NOT saved; raises requirement_generator.NotGeneratable."""
    from services.requirement_evidence import build_evidence_package
    from services.requirement_generator import generate_rule_based

    return generate_rule_based(build_evidence_package(session, cluster_id=cluster_id))


def save_draft(session: Session, draft, actor: str) -> str:
    """Persist through the existing review workflow (status DRAFT, never
    approved automatically). Returns the existing requirement instead of
    creating a duplicate for the same cluster."""
    from services import requirement_review as rr

    existing = session.scalar(select(VehicleRequirement.requirement_id).where(
        VehicleRequirement.source_cluster_id == draft.source_cluster_id).limit(1))
    if existing:
        return existing
    return rr.create_requirement(session, draft, actor).requirement_id


# ---------------------------------------------------------------------------
# Step 3 - agents, weights, readiness
# ---------------------------------------------------------------------------


def available_agents(session: Session) -> list[dict]:
    from services.agent_configuration import list_agents
    from services.agent_resources import RESOURCES

    out = []
    for a in list_agents(session):
        if a["status"] != "ACTIVE":
            continue
        out.append({**a, "resource_names": [RESOURCES[r].name for r in a["resources"] if r in RESOURCES]})
    return out


@dataclass
class Readiness:
    errors: list[str] = field(default_factory=list)
    missing_human: list[dict] = field(default_factory=list)  # {agent_id, agent_name, requirement_id, title}
    ai_plans: list = field(default_factory=list)  # agent_analysis.AnalysisPlan (uncached ones need approval)
    ai_cached: int = 0
    estimated_usd: float = 0.0
    upper_usd: float = 0.0
    total_bp: int = 0

    @property
    def needs_paid_ai(self) -> bool:
        return any(p.cached is None and p.requests for p in self.ai_plans)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing_human


def weight_total_bp(weights: dict[str, object]) -> int:
    return sum(percent_to_bp(v) for v in weights.values())


def check_readiness(session: Session, settings: LLMSettings, requirement_ids: list[str], weights: dict[str, object],
                    actor: str) -> Readiness:
    """All Run-Prioritization validations, without side effects except the free,
    deterministic evidence selections AI agents need."""
    from services import agent_analysis as aa
    from services.agent_configuration import agent_resources, get_agent
    from services.agent_scoring import check_compatibility

    r = Readiness()
    if not requirement_ids:
        r.errors.append("Select at least one requirement.")
    if not weights:
        r.errors.append("Select at least one agent.")
    try:
        r.total_bp = weight_total_bp(weights)
    except ValueError as exc:
        r.errors.append(str(exc))
        return r
    if weights and r.total_bp != BP_TOTAL:
        r.errors.append("Please adjust the selected agent weights to 100%.")
    reqs = {q.requirement_id: q for q in session.scalars(select(VehicleRequirement).where(
        VehicleRequirement.requirement_id.in_(requirement_ids or ["-"])))}
    for agent_id in weights:
        agent = get_agent(session, agent_id)
        res = agent_resources(session, agent_id)
        errs, _ = check_compatibility(agent.execution_method, agent.scoring_method, res)
        if errs:
            r.errors.append(f"{agent.name}: " + " ".join(errs))
            continue
        if agent.scoring_method == "HUMAN_SCORE":
            for rid in requirement_ids:
                has = session.scalar(select(PriorityAssessment.assessment_id).where(
                    PriorityAssessment.requirement_id == rid, PriorityAssessment.agent_id == agent_id,
                    PriorityAssessment.superseded.is_(False)).limit(1))
                if not has and rid in reqs:
                    r.missing_human.append({"agent_id": agent_id, "agent_name": agent.name, "requirement_id": rid,
                                            "title": reqs[rid].title})
        if agent.execution_method == "AI_ANALYSIS" and set(res) & set(aa.ANALYSABLE_RESOURCES):
            for rid in requirement_ids:
                plan = aa.plan_analysis(session, settings, agent_id, rid, actor or "system")
                r.ai_plans.append(plan)
                if plan.cached is not None:
                    r.ai_cached += 1
                elif plan.requests:
                    r.estimated_usd += plan.estimate.estimated_usd
                    r.upper_usd += plan.estimate.upper_bound_usd
    r.estimated_usd, r.upper_usd = round(r.estimated_usd, 4), round(r.upper_usd, 4)
    return r


def execute(session: Session, settings: LLMSettings, requirement_ids: list[str], weights: dict[str, object],
            actor: str, *, llm=None, approved_max_usd: float | None = None) -> str:
    """Run uncached AI analyses (approval-gated), then a NEW versioned run
    profile and prioritization run. Existing profiles and runs are untouched.
    Returns the run ID."""
    from services import agent_analysis as aa
    from services import dynamic_prioritization as dp
    from services import prioritization_profiles as pp

    if not (actor or "").strip():
        raise ValueError("Enter your user ID first.")
    ready = check_readiness(session, settings, requirement_ids, weights, actor)
    if ready.errors:
        raise ValueError(" ".join(ready.errors))
    if ready.missing_human:
        raise ValueError("Human assessments are missing for: " + ", ".join(
            f"{m['agent_name']} / {m['title']}" for m in ready.missing_human))
    if ready.needs_paid_ai:
        if llm is None:
            raise aa.ApprovalRequired("AI unavailable — configure OpenAI API in application settings.")
        remaining = approved_max_usd
        for plan in ready.ai_plans:
            if plan.cached is None and plan.requests:
                row = aa.run_analysis(session, llm, settings, plan, actor, approved_max_usd=remaining)
                if row.status != "completed":
                    raise RuntimeError(f"AI analysis for agent '{plan.agent['name']}' failed: {row.error}")
                remaining = None if remaining is None else round(remaining - row.cost_usd, 6)
    name = f"Prioritization {_now():%Y-%m-%d %H:%M} ({len(weights)} agents)"
    profile = pp.create_profile(session, name, {a: str(Decimal(str(w))) for a, w in weights.items()}, actor,
                                description="Created by the Product Prioritization workflow.")
    run = dp.run_prioritization(session, profile.profile_id, actor, list(requirement_ids))
    return run.run_id


# ---------------------------------------------------------------------------
# Step 4 - WHY (offline, from the saved run)
# ---------------------------------------------------------------------------


def _level(score: float | None) -> str:
    if score is None:
        return "could not assess"
    return "strong" if score >= 60 else "moderate" if score >= 35 else "limited"


def why(session: Session, run_id: str, requirement_id: str, max_quotes: int = 5) -> dict:
    """Business explanation of one saved result. Built only from the saved run,
    its frozen evidence and the requirement's evidence links; no LLM call."""
    from services import dynamic_prioritization as dp
    from services import requirement_review as rr
    from services.agent_explanations import calculation_lines

    row = next((r for r in dp.ranking(session, run_id) if r.requirement_id == requirement_id), None)
    if row is None:
        raise ValueError("Requirement is not part of this run.")
    req = rr.requirement_as_dict(session, rr.get_requirement(session, requirement_id))
    results = dp.agent_results(session, run_id, requirement_id)
    frozen = dp.evidence_snapshot(session, run_id).get(requirement_id) or {}
    # customer evidence: prefer the evidence of the AI agent with the largest contribution
    agent_ev = {}
    for res in sorted(results, key=lambda x: -(x["weighted_contribution"] or 0)):
        ev = (frozen.get("agents") or {}).get(res["agent_id"])
        if ev:
            agent_ev = ev
            break
    agent_ev = agent_ev or frozen
    counts = agent_ev.get("counts") or {}
    obs_ids = []
    for role in ("COMPLAINT", "IMPROVEMENT_REQUEST"):
        obs_ids += (agent_ev.get("observation_ids") or {}).get(role, [])
    evidence_basis = "the run's validated AI evidence" if obs_ids else None
    if not obs_ids:  # no AI agent in the run: use the requirement's linked direct comments (validated observations)
        linked = [e["source_key"] for e in req["evidence"] if e["tier"] == "DIRECT" and e["source_table"] == "feedback"]
        rows_ = list(session.scalars(select(AIFeedbackObservation).where(
            AIFeedbackObservation.original_feedback_id.in_(linked or ["-"]),
            AIFeedbackObservation.taxonomy_category == req["need_code"],
            AIFeedbackObservation.observation_type.in_(("COMPLAINT", "IMPROVEMENT_REQUEST")))))
        obs_ids = [o.observation_id for o in rows_]
        if rows_:
            evidence_basis = "the requirement's linked customer comments"
            counts = {"complaints": len({o.original_feedback_id for o in rows_ if o.observation_type == "COMPLAINT"}),
                      "improvement_requests": len({o.original_feedback_id for o in rows_
                                                   if o.observation_type == "IMPROVEMENT_REQUEST"}),
                      "combined_demand": len({o.original_feedback_id for o in rows_})}
    quotes = []
    if obs_ids:
        for o in session.scalars(select(AIFeedbackObservation).where(AIFeedbackObservation.observation_id.in_(obs_ids))
                                 .order_by(AIFeedbackObservation.original_feedback_id).limit(200)):
            if len(quotes) < max_quotes and o.original_feedback_id not in {q["feedback_id"] for q in quotes}:
                quotes.append({"feedback_id": o.original_feedback_id, "type": o.observation_type,
                               "customer_need": o.customer_need, "excerpt": o.evidence_excerpt})
    needs: dict[str, set[str]] = {}
    if obs_ids:
        for o in session.scalars(select(AIFeedbackObservation).where(AIFeedbackObservation.observation_id.in_(obs_ids))):
            needs.setdefault(_norm(o.customer_need), set()).add(o.original_feedback_id)
    top_needs = [k for k, _ in sorted(needs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:4]]

    matters = []
    if row.priority_status == "COMPLETE":
        lead = max(results, key=lambda x: x["weighted_contribution"] or 0)
        matters.append(f"Ranked {row.rank} with a priority score of {row.final_score} out of 100. The largest share "
                       f"comes from {lead['agent_name']}.")
    else:
        matters.append("This requirement has no formal rank yet because some agents could not assess it: "
                       + ", ".join(row.missing_agents) + ".")
    if counts:
        matters.append(f"{counts.get('combined_demand', 0)} distinct customers raised a related complaint or "
                       f"improvement request ({counts.get('complaints', 0)} complaints, "
                       f"{counts.get('improvement_requests', 0)} requests).")
    perspectives = [{"agent": r["agent_name"], "view": _level(r["score"]) + " support" if r["status"] == "SCORED"
                     else "could not assess", "basis": r["rationale"].split(". ")[0][:220],
                     "weight_percent": r["weight_percent"]} for r in results]
    supporting = [e for e in req["evidence"] if e["tier"] in ("SUPPORTING", "CONTEXTUAL")]
    studies = sorted({e["source_key"].split("::")[0] for e in supporting if e["source_table"] == "customer_studies"})
    competitor = [e["source_key"] for e in supporting if e["source_table"] == "competitor_evidence"]
    uncertain = []
    if counts.get("counter_evidence"):
        uncertain.append(f"{counts['counter_evidence']} customers point the other way (counter-evidence).")
    if counts.get("uncertain"):
        uncertain.append(f"{counts['uncertain']} related comments have uncertain relevance and were not counted.")
    for res in results:
        for m in (res["inputs"].get("missing_information") or [])[:3]:
            uncertain.append(f"{res['agent_name']}: {m}")
        if res["status"] != "SCORED":
            uncertain.append(f"{res['agent_name']}: {res['rationale']}")
    uncertain.append("Comment counts come from the supplied G60 export (mostly US customers); frequency is not severity.")
    return {"requirement_id": requirement_id, "title": row.title, "status": row.priority_status, "rank": row.rank,
            "evidence_basis": evidence_basis,
            "score": row.final_score, "why_it_matters": matters, "top_needs": top_needs, "quotes": quotes,
            "counts": counts, "perspectives": perspectives, "studies": studies, "competitor_refs": competitor,
            "uncertain": uncertain,
            "breakdown": [{"agent": r["agent_name"], "lines": calculation_lines(r)} for r in results]}
