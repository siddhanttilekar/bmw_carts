"""Decision assistant controller (Part 6).

Routes a question to a deterministic handler. Supported intents:
  explain_priority  A  why does the requirement have this priority / rank
  explain_agent     B  why did agent X assign this score        (-> agent_chat)
  evidence          C  which customer comments support it
  compare           D  why is A ranked above B                   (-> agent_debate)
  missing           E  why is it incomplete
  scenario          F  what if agent X receives N% weight        (-> what_if_simulator)
  challenge         G  what if the problem is more serious than the ranking suggests
  reliability       H  how reliable is this recommendation

OFFLINE mode (default) needs no credentials: intents come from predefined
questions or simple keyword rules over the run's configured agent names;
anything ambiguous returns clarification + suggestions, never a guessed
answer. Every answer is bound to one saved run, uses stored calculations,
validates its evidence IDs, and is persisted (IDs only, no verbatim feedback).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.decision_models import DecisionConversation, DecisionMessage
from core.requirement_models import RequirementEvidenceLink
from services.agent_chat import discuss_agent
from services.agent_debate import run_debate
from services.decision_context import get_decision_context
from services.decision_grounding import (
    AssistantResponse,
    bind_run,
    finalize,
    historical_notices,
    requirement_evidence,
    run_agents,
)
from services.dynamic_prioritization import ranking
from services.what_if_simulator import SimulationError, simulate

INTENTS = ("explain_priority", "explain_agent", "evidence", "compare", "missing", "scenario", "challenge", "reliability")

SUGGESTED_QUESTIONS = {
    "explain_priority": "Why does this requirement have this priority?",
    "explain_agent": "Why did the selected agent assign this score?",
    "evidence": "Which customer comments support this requirement?",
    "compare": "Why is this requirement ranked above or below the other selected requirement?",
    "missing": "Why is this requirement incomplete?",
    "scenario": "What happens if the selected agent receives a different weight?",
    "challenge": "What if this customer problem is more serious than its ranking suggests?",
    "reliability": "How reliable is this recommendation?",
}

_RULES = [
    ("scenario", re.compile(r"\bwhat (?:if|happens)\b.*\d+(?:\.\d+)?\s*%|\bweight\b.*\d+(?:\.\d+)?\s*%", re.I)),
    ("missing", re.compile(r"\b(incomplete|missing|unscored|not ranked)\b", re.I)),
    ("evidence", re.compile(r"\b(comments?|evidence|support(?:s|ing)?|sources?)\b", re.I)),
    ("compare", re.compile(r"\b(compare|versus|vs\.?|above|below|instead of|difference between)\b", re.I)),
    ("challenge", re.compile(r"\b(more serious|too low|too high|underestimat|overestimat|challenge|disagree|safety critical)\b", re.I)),
    ("reliability", re.compile(r"\b(reliab|trust|confiden|robust|limitation)", re.I)),
    ("explain_priority", re.compile(r"\b(why|explain)\b.*\b(rank|ranked|priority|first|score|top)\b", re.I)),
]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def classify(question: str, agent_names: list[str]) -> tuple[str | None, dict]:
    """Deterministic intent detection. Returns (intent or None, extracted params)."""
    q = question.strip()
    params: dict = {}
    mentioned = [n for n in agent_names if n and n.lower() in q.lower()]
    if mentioned:
        params["agent_name"] = max(mentioned, key=len)
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if pct:
        params["percent"] = pct.group(1)
    hits = [intent for intent, rx in _RULES if rx.search(q)]
    if "scenario" in hits and "agent_name" in params:
        return "scenario", params
    if mentioned and re.search(r"\b(why|how|explain)\b", q, re.I) and not {"evidence", "missing", "compare"} & set(hits):
        return "explain_agent", params
    specific = [h for h in hits if h != "explain_priority"]
    if len(specific) == 1:
        return specific[0], params
    if not specific and hits:
        return "explain_priority", params
    return None, params


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _base(session, intent, title, requirement_id, run) -> AssistantResponse:
    return AssistantResponse(intent=intent, title=title, requirement_id=requirement_id, run_id=run.run_id,
                             profile_id=run.profile_id, profile_version=run.profile_version,
                             notices=historical_notices(session, run, requirement_id))


def explain_priority(session, requirement_id, run) -> AssistantResponse:
    ctx = get_decision_context(session, requirement_id, run_id=run.run_id)
    resp = _base(session, "explain_priority", "Why this priority", requirement_id, run)
    if ctx["priority_status"] != "COMPLETE":
        resp.add("FACT", f"Status INCOMPLETE in run {run.run_id}: no final score and no formal rank.")
        resp.add("LIMITATION", "Missing agent results: " + ", ".join(ctx["missing_assessments"]) + ".")
        return finalize(session, resp)
    rows = ranking(session, run.run_id)
    complete = [r for r in rows if r.priority_status == "COMPLETE"]
    resp.add("FACT", f"Final score {ctx['final_score']}, rank {ctx['rank']} of {len(complete)} complete requirements "
                     f"(profile {run.profile_id} v{run.profile_version}, run {run.run_id}).")
    for c in ctx["agent_contributions"]:
        resp.add("FACT", f"{c['agent_name']}: score {c['score']} × weight {c['weight_percent']}% = "
                         f"{c['weighted_contribution']} points ({c['scoring_method']} v{c['scoring_version']}).")
    tops = ctx.get("top_contributors") or []
    if len(tops) == 1:
        resp.add("FACT", f"Largest weighted contribution: {tops[0]}.")
    elif tops:
        points = ctx["agent_contributions"][0]["weighted_contribution"]
        resp.add("FACT", f"Largest weighted contribution is shared ({points} points each): {', '.join(tops)}.")
    neighbour = next((r for r in complete if r.rank == ctx["rank"] + 1), None) if ctx["rank"] == 1 else \
        next((r for r in reversed(complete) if r.rank is not None and r.rank < ctx["rank"]), None)
    if neighbour:
        debate = run_debate(session, run.run_id, requirement_id, neighbour.requirement_id)
        relation = "ahead of" if ctx["rank"] < neighbour.rank else "behind"
        resp.add("FACT", f"It is {relation} {neighbour.requirement_id} ('{neighbour.title}', score {neighbour.final_score}).")
        for d in debate.round3_drivers[:3]:
            resp.add("FACT", f"Contribution difference from {d['agent']}: {d['difference']:+} points.")
    for a in ctx["assumptions"][:3]:
        resp.add("ASSUMPTION", a)
    for l in ctx["limitations"][:3]:
        resp.add("LIMITATION", l)
    resp.add("LIMITATION", "Scoring methodology is a prototype, not an official BMW framework; priority does not imply approval.")
    return finalize(session, resp)


def evidence(session, requirement_id, run) -> AssistantResponse:
    resp = _base(session, "evidence", "Supporting evidence", requirement_id, run)
    resp.data_basis = "requirement evidence links (saved with the requirement)"
    groups = requirement_evidence(session, requirement_id)
    direct = groups["DIRECT"]
    resp.add("FACT", f"{len(direct)} distinct G60 customer comments are linked as direct evidence.",
             refs=[d["key"] for d in direct[:20]])
    by_table: dict[str, int] = {}
    for g in groups["SUPPORTING"]:
        by_table[g["table"]] = by_table.get(g["table"], 0) + 1
    if groups["SUPPORTING"]:
        resp.add("FACT", "Supporting evidence (linked at need-category level, not proof of individual complaints): "
                 + ", ".join(f"{n} {t}" for t, n in sorted(by_table.items())) + ".",
                 refs=[g["key"] for g in groups["SUPPORTING"][:20]])
    if groups["CONTEXTUAL"]:
        resp.add("FACT", f"{len(groups['CONTEXTUAL'])} contextual items (competitor observations / option list) - "
                         "context only, never evidence about BMW customers.", refs=[g["key"] for g in groups["CONTEXTUAL"][:20]])
    review = sum(g["requires_review"] for gs in groups.values() for g in gs)
    if review:
        resp.add("LIMITATION", f"{review} linked items have mappings marked requires_review.")
    resp.evidence_refs = [{"table": g["table"], "key": g["key"]} for gs in groups.values() for g in gs]
    return finalize(session, resp)


def missing(session, requirement_id, run) -> AssistantResponse:
    ctx = get_decision_context(session, requirement_id, run_id=run.run_id)
    resp = _base(session, "missing", "Missing assessments", requirement_id, run)
    unscored = [c for c in ctx["agent_contributions"] if c["status"] != "SCORED"]
    if not unscored:
        resp.add("FACT", f"Nothing is missing: all {len(ctx['agent_contributions'])} agents scored this requirement in run "
                         f"{run.run_id} (status {ctx['priority_status']}).")
    for c in unscored:
        resp.add("LIMITATION", f"{c['agent_name']} ({c['scoring_method']}): {c['rationale']}")
    if unscored:
        resp.add("FACT", "Missing weight is not redistributed, so the requirement has no final score and no rank.")
    return finalize(session, resp)


def scenario(session, requirement_id, run, agent_id: str | None, percent) -> AssistantResponse:
    resp = _base(session, "scenario", "What-if simulation", requirement_id, run)
    agents = {a["agent_id"]: a for a in run_agents(run)}
    if agent_id not in agents or percent is None:
        resp.needs_clarification = True
        resp.add("LIMITATION", "Name one of the run's agents and a weight, e.g. "
                 f"'What happens if {next(iter(agents.values()))['name']} receives 40%?'")
        resp.suggestions = [a["name"] for a in agents.values()]
        return resp
    target_bp = int(Decimal(str(percent)) * 100)
    if not 0 <= target_bp <= 10000:
        raise SimulationError("Weight must be between 0 and 100%.")
    others = {aid: a["weight_bp"] for aid, a in agents.items() if aid != agent_id}
    rest, total_others = 10000 - target_bp, sum(others.values())
    weights_bp = {agent_id: target_bp}
    if others:
        if total_others == 0:
            raise SimulationError("Other agents have zero weight; set weights explicitly in the simulator.")
        scaled = {aid: rest * w // total_others for aid, w in others.items()}
        remainder = rest - sum(scaled.values())
        for aid in sorted(others, key=lambda a: -others[a])[:remainder]:
            scaled[aid] += 1
        weights_bp.update(scaled)
    elif target_bp != 10000:
        raise SimulationError("A single-agent profile must keep 100% weight.")
    sim = simulate(session, run.run_id, {aid: Decimal(bp) / 100 for aid, bp in weights_bp.items()})
    resp.add("ASSUMPTION", "Other agents' weights were scaled proportionally so the total stays 100%: "
             + ", ".join(f"{agents[a]['name']} {Decimal(bp) / 100}%" for a, bp in weights_bp.items()) + ".")
    row = next(r for r in sim.rows if r.requirement_id == requirement_id)
    if row.simulated_status == "COMPLETE":
        resp.add("SIMULATION", f"Simulated score {row.simulated_final} (official {row.original_final}); simulated rank "
                               f"{row.simulated_rank} (official {row.original_rank}).")
    else:
        resp.add("SIMULATION", "Simulated status INCOMPLETE (missing: " + ", ".join(row.missing_agents) + "); no simulated rank.")
    movers = [r for r in sim.rows if r.rank_change]
    if movers:
        resp.add("SIMULATION", "Rank changes across the run: " + "; ".join(
            f"{r.requirement_id} {r.original_rank}→{r.simulated_rank}" for r in movers) + ".")
    resp.add("LIMITATION", "Hypothetical result; not saved and not an official priority.")
    return finalize(session, resp)


def challenge(session, requirement_id, run) -> AssistantResponse:
    ctx = get_decision_context(session, requirement_id, run_id=run.run_id)
    resp = _base(session, "challenge", "Challenging the priority", requirement_id, run)
    for c in ctx["agent_contributions"]:
        method = c["scoring_method"]
        if method == "HUMAN_SCORE":
            a = next((h for h in ctx["human_assessments"] if h["agent"] == c["agent_name"]), None)
            if a:
                resp.add("ASSESSMENT", f"{c['agent_name']}: {a['assessor_id']} scored {a['score']} - \"{a['rationale']}\"",
                         refs=[a["assessment_id"]])
            else:
                resp.add("LIMITATION", f"{c['agent_name']}: no human assessment recorded.")
        elif method == "LOG_NORMALIZED_FEEDBACK_COUNT":
            resp.add("LIMITATION", f"{c['agent_name']} measures how often the need appears, not how serious it is; "
                                   "no data agent in this run measures severity.")
        elif method == "EVIDENCE_COVERAGE":
            resp.add("LIMITATION", f"{c['agent_name']} measures evidence coverage, not customer impact.")
    resp.add("FACT", "Scores change only through the existing workflow: record a new human assessment (or add evidence) "
                     "in Agent Prioritization and execute a new run. A challenge records the dispute and its review.")
    if ctx["critical_flags"]:
        resp.add("FACT", "Critical-issue flags on this requirement: " + "; ".join(
            f"{f['category']} ({f['status']})" for f in ctx["critical_flags"]) + ".")
    else:
        resp.add("FACT", "No critical-issue flag is recorded; one can be raised and must be confirmed by a person.")
    return finalize(session, resp)


def reliability(session, requirement_id, run) -> AssistantResponse:
    ctx = get_decision_context(session, requirement_id, run_id=run.run_id)
    resp = _base(session, "reliability", "Reliability indicators", requirement_id, run)
    contributions = ctx["agent_contributions"]
    human_w = sum(c["weight_percent"] for c in contributions if c["scoring_method"] == "HUMAN_SCORE")
    data_w = sum(c["weight_percent"] for c in contributions if c["scoring_method"] != "HUMAN_SCORE")
    resp.add("FACT", f"Weight from human assessments {human_w}% vs data-derived agents {data_w}%.")
    resp.add("FACT", f"Priority status {ctx['priority_status']}; review status {ctx['review_status']} "
                     "(a priority score is not an approval).")
    review_links = session.scalar(select(RequirementEvidenceLink.link_id).where(
        RequirementEvidenceLink.requirement_id == requirement_id, RequirementEvidenceLink.requires_review.is_(True)).limit(1))
    if review_links:
        resp.add("LIMITATION", "Some evidence mappings for this requirement are marked requires_review.")
    for l in ctx["limitations"]:
        resp.add("LIMITATION", l)
    for a in ctx["assumptions"][:4]:
        resp.add("ASSUMPTION", a)
    resp.add("LIMITATION", "The prototype scoring methodology has not been validated as a BMW framework.")
    return finalize(session, resp)


# ---------------------------------------------------------------------------
# Controller + persistence
# ---------------------------------------------------------------------------


def answer(session: Session, requirement_id: str, run_id: str, *, intent: str | None = None, question: str = "",
           agent_id: str | None = None, other_requirement_id: str | None = None, percent=None) -> AssistantResponse:
    run = bind_run(session, run_id, requirement_id)
    agents = run_agents(run)
    params: dict = {}
    if intent is None:
        intent, params = classify(question, [a["name"] for a in agents])
    if intent is None:
        resp = _base(session, "clarification", "Please clarify", requirement_id, run)
        resp.needs_clarification = True
        resp.add("LIMITATION", "The question could not be interpreted reliably offline. Choose one of the suggested questions.")
        resp.suggestions = list(SUGGESTED_QUESTIONS.values())
        return resp
    if intent not in INTENTS:
        raise ValueError(f"Unknown intent {intent}")
    if "agent_name" in params and agent_id is None:
        agent_id = next(a["agent_id"] for a in agents if a["name"] == params["agent_name"])
    percent = percent if percent is not None else params.get("percent")
    if intent == "explain_agent":
        if not agent_id:
            resp = _base(session, intent, "Select an agent", requirement_id, run)
            resp.needs_clarification = True
            resp.suggestions = [a["name"] for a in agents]
            resp.add("LIMITATION", "Select one of the run's agents.")
            return resp
        return discuss_agent(session, run_id, requirement_id, agent_id)
    if intent == "compare":
        if not other_requirement_id:
            resp = _base(session, intent, "Select a second requirement", requirement_id, run)
            resp.needs_clarification = True
            resp.add("LIMITATION", "Select the second requirement to compare (from the same run).")
            return resp
        d = run_debate(session, run_id, requirement_id, other_requirement_id)
        resp = _base(session, intent, f"{requirement_id} vs {other_requirement_id}", requirement_id, run)
        resp.add("FACT", d.round4_synthesis["ranking_under_profile"])
        resp.add("FACT", d.round4_synthesis["agreement"])
        for dr in d.round3_drivers[:4]:
            resp.add("FACT", f"{dr['agent']}: {dr['contribution_a']} vs {dr['contribution_b']} points ({dr['difference']:+}).")
        sens = d.round4_synthesis["sensitivity"]
        if sens["single_agent_shifts_to_tie"]:
            s0 = sens["single_agent_shifts_to_tie"][0]
            resp.add("SIMULATION", f"The order would tie if {s0['agent']}'s weight changed by {s0['weight_change_points']:+} "
                                   f"points (others scaled proportionally) - {sens['classification']}.")
        for m in d.round4_synthesis["missing_evidence"]:
            resp.add("LIMITATION", f"Unable to conclude: {m}")
        return finalize(session, resp)
    if intent == "scenario":
        return scenario(session, requirement_id, run, agent_id, percent)
    return {"explain_priority": explain_priority, "evidence": evidence, "missing": missing, "challenge": challenge,
            "reliability": reliability}[intent](session, requirement_id, run)


def ask(session: Session, requirement_id: str, run_id: str, user: str, *, question: str | None = None,
        intent: str | None = None, conversation_id: str | None = None, **kwargs) -> tuple[str, AssistantResponse]:
    """Answer and persist (question + structured response + evidence IDs)."""
    if not (user or "").strip():
        raise ValueError("A user identifier is required.")
    question = question or SUGGESTED_QUESTIONS.get(intent or "", "")
    resp = answer(session, requirement_id, run_id, intent=intent, question=question, **kwargs)
    run = bind_run(session, run_id)
    conv = session.get(DecisionConversation, conversation_id) if conversation_id else None
    if conv is None or conv.run_id != run_id or conv.requirement_id != requirement_id:
        conv = DecisionConversation(conversation_id=f"CNV-{uuid.uuid4().hex[:10].upper()}", requirement_id=requirement_id,
                                    run_id=run_id, profile_id=run.profile_id, profile_version=run.profile_version,
                                    agent_id=kwargs.get("agent_id"), created_by=user.strip(), created_at=_now())
        session.add(conv)
        session.flush()
    session.add(DecisionMessage(conversation_id=conv.conversation_id, requirement_id=requirement_id, run_id=run_id,
                                agent_id=resp.agent_id or kwargs.get("agent_id"), intent=resp.intent, question=question,
                                response_json=resp.model_dump_json(), evidence_refs_json=json.dumps(resp.evidence_refs),
                                response_mode=resp.mode, created_at=_now()))
    session.flush()
    return conv.conversation_id, resp


def conversation_history(session: Session, conversation_id: str) -> dict:
    conv = session.get(DecisionConversation, conversation_id)
    if conv is None:
        raise ValueError(f"Unknown conversation {conversation_id!r}")
    run = bind_run(session, conv.run_id)
    messages = [{"question": m.question, "intent": m.intent, "response": AssistantResponse.model_validate_json(m.response_json),
                 "mode": m.response_mode, "at": m.created_at}
                for m in session.scalars(select(DecisionMessage).where(DecisionMessage.conversation_id == conversation_id)
                                         .order_by(DecisionMessage.message_id))]
    return {"conversation_id": conversation_id, "requirement_id": conv.requirement_id, "run_id": conv.run_id,
            "profile_id": conv.profile_id, "profile_version": conv.profile_version,
            "notices": historical_notices(session, run, conv.requirement_id), "messages": messages}


def list_conversations(session: Session, requirement_id: str | None = None) -> list[dict]:
    stmt = select(DecisionConversation).order_by(DecisionConversation.created_at.desc())
    if requirement_id:
        stmt = stmt.where(DecisionConversation.requirement_id == requirement_id)
    return [{"conversation_id": c.conversation_id, "requirement_id": c.requirement_id, "run_id": c.run_id,
             "agent_id": c.agent_id, "created_by": c.created_by, "created_at": c.created_at} for c in session.scalars(stmt)]
