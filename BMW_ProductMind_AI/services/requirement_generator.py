"""Requirement draft generation (Part 4).

MODE A - RULE_BASED (default, offline, no API key)
    Category-specific, solution-neutral templates filled ONLY with values from
    the evidence package. Framing is decided from the distinct-comment counts
    of the original VFC level-3 label groups:
      * improve  - issue/request labels are a substantial share
      * retain   - comments are predominantly positive labels
      * refine   - selection is broad/ambiguous or evidence is thin: a
                   problem-oriented draft is produced, marked as requiring
                   refinement, with the missing information listed. No
                   technical solution is proposed.
    Acceptance criteria name a measurement method; targets are always
    "proposed - requires validation". No BMW specifications are invented.

MODE B - LLM_ASSISTED (optional, disabled by default)
    Only runs when ALL hold: PRODUCTMIND_ALLOW_EXTERNAL_LLM is set, an API key
    is present, the caller passes ``confirmed=True`` and acknowledges the
    displayed cost estimate. Server-side model fallback is forced off.
    The model writes text fields only; evidence IDs and counts are attached by
    code and the result goes through the same validator.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from core.config import LLMSettings, get_llm_settings
from services.need_taxonomy import NEED_NAMES
from services.requirement_evidence import CATCH_ALL_VFC, EvidencePackage

VEHICLE_MODEL = "BMW 5 Series (G60)"
TARGET_GENERATION = "Next-generation BMW 5 Series (successor to G60; launch timing not assumed)"
PROPOSED_TARGET = "Proposed target - requires validation (no target derivable from the available evidence)"
MIN_COMMENTS_FOR_SPECIFIC = 5


# ---------------------------------------------------------------------------
# Draft model
# ---------------------------------------------------------------------------


class AcceptanceCriterion(BaseModel):
    criterion: str
    measurement: str
    target: str = PROPOSED_TARGET
    target_status: str = "proposed_requires_validation"  # | evidence_baseline


class EvidenceRef(BaseModel):
    tier: str  # DIRECT | SUPPORTING | CONTEXTUAL
    source_table: str
    source_key: str
    source_record_id: str | None
    evidence_scope: str  # cluster_member | need_category | vehicle_catalogue
    requires_review: bool = False
    note: str | None = None


class RequirementDraft(BaseModel):
    title: str
    source_cluster_id: str
    source_insight_id: str | None
    need_code: str
    customer_need: str
    customer_problem: str
    proposed_requirement: str
    requirement_type: str
    vehicle_model: str = VEHICLE_MODEL
    source_vehicle_code: str = "G60"
    target_generation: str = TARGET_GENERATION
    evidence_summary: str
    assumptions: list[str]
    acceptance_criteria: list[AcceptanceCriterion]
    generation_method: str = "RULE_BASED"
    template_id: str | None = None
    framing: str  # improve | retain | refine
    requires_refinement: bool = False
    refinement_notes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef]
    allowed_numbers: list[str] = Field(default_factory=list)  # numbers that come from the evidence package

    def supporting_evidence_ids(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"direct": [], "supporting": [], "contextual": []}
        for e in self.evidence:
            out[e.tier.lower()].append(e.source_key)
        return out


# ---------------------------------------------------------------------------
# Category templates (solution-neutral)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    requirement_type: str
    value: str        # what customers value, as a need phrase
    issue_kind: str   # how issues in this category are described
    measurement: str  # how fulfilment would be measured (method, not target)


TEMPLATES: dict[str, Template] = {
    "INFOTAINMENT": Template("functional_usability", "dependable, easy-to-use infotainment and digital functions",
                             "function and usability issues",
                             "task success and error rates for core tasks in moderated usability tests with target customers"),
    "CONTROLS": Template("usability", "intuitive, low-effort operation",
                         "operation and usability issues",
                         "task success rate and time on task for frequent tasks in usability tests, including a driving-simulator condition"),
    "DRIVING_DYNAMICS": Template("performance", "the ride and handling character expected of a BMW 5 Series",
                                 "ride, handling or chassis issues",
                                 "ride and handling ratings in structured customer drive clinics against the G60"),
    "POWERTRAIN": Template("performance", "responsive and refined powertrain behaviour",
                           "powertrain behaviour or refinement issues",
                           "customer ratings of power delivery and refinement in drive clinics against the G60"),
    "CHARGING": Template("functional", "reliable and convenient charging",
                         "charging reliability or convenience issues",
                         "charging-session success rate and customer-reported charging issues in field tests"),
    "RANGE_EFFICIENCY": Template("performance", "dependable electric range and trustworthy range information",
                                 "range or range-information issues",
                                 "agreement between displayed and achieved range on defined real-world test routes"),
    "SOFTWARE": Template("reliability", "stable software behaviour and smooth updates",
                         "software stability or update issues",
                         "software-related fault reports per vehicle in field monitoring"),
    "DRIVER_ASSISTANCE": Template("functional_safety_related", "predictable, trustworthy driver-assistance behaviour",
                                  "assistance behaviour or usability issues",
                                  "rate of unexpected or unexplained assistance interventions in structured test drives, and trust ratings"),
    "SAFETY_SECURITY": Template("safety", "a high perceived and actual level of safety and security",
                                "safety or security concerns",
                                "perceived-safety ratings in customer studies and issue reports in field monitoring"),
    "CONNECTIVITY": Template("functional", "reliable phone, app and remote-service connectivity",
                             "connection or app reliability issues",
                             "success rates for pairing, app and remote functions in field tests"),
    "ACCESS": Template("functional", "reliable and convenient vehicle access",
                       "access, key or locking issues",
                       "success rate of locking, unlocking and key/Digital Key operations in field tests"),
    "CLIMATE": Template("comfort", "effective, easy-to-control climate comfort",
                        "climate performance or climate-control issues",
                        "time-to-comfort measurements and climate-control usability ratings in customer clinics"),
    "INTERIOR_COMFORT": Template("comfort", "long-distance seating and cabin comfort",
                                 "seating, comfort or cabin-noise issues",
                                 "seating-comfort and cabin-noise ratings in customer drive clinics against the G60"),
    "INTERIOR_QUALITY": Template("quality", "high perceived interior quality and design",
                                 "perceived-quality or interior design issues",
                                 "perceived-quality ratings in customer clinics against the G60"),
    "PRACTICALITY": Template("functional", "practical space and storage for everyday use",
                             "space or storage issues",
                             "ratings of space and storage usefulness in customer clinics against the G60"),
    "EXTERIOR": Template("design", "an attractive, high-quality exterior design and lighting",
                         "exterior design, part or lighting issues",
                         "exterior design and lighting ratings in customer clinics against the G60"),
    "OVERALL": Template("overall", "a convincing overall vehicle", "overall issues",
                        "overall product satisfaction in customer studies against the G60"),
}


class NotGeneratable(ValueError):
    pass


def _focus(pkg: EvidencePackage) -> str:
    if pkg.cluster_level == "vfc_category":
        return f"'{pkg.cluster_label}'"
    if pkg.cluster_level == "extracted_need":
        return f"the need '{pkg.cluster_label}'"
    return f"the {NEED_NAMES[pkg.need_code]} category"


def decide_framing(pkg: EvidencePackage) -> tuple[str, list[str]]:
    """Returns (framing, reasons why refinement is needed)."""
    reasons: list[str] = []
    n = pkg.unique_feedback_count
    if pkg.need_code == "OVERALL":
        reasons.append("The need category 'Overall Vehicle Impression' is too broad for a vehicle requirement.")
    if pkg.cluster_level == "need_category":
        reasons.append("A whole need category spans several VFC classes; select a VFC cluster to identify the affected function.")
    vfc = pkg.vfc_level2_name or ""
    if vfc in CATCH_ALL_VFC or "overall technology" in vfc:
        reasons.append(f"The VFC class '{pkg.cluster_label}' is a broad catch-all and does not identify a specific function.")
    if n < MIN_COMMENTS_FOR_SPECIFIC:
        reasons.append(f"Only {n} direct comments; more evidence is needed before specifying a requirement.")
    g = pkg.label_groups
    if n and g.get("unlabelled", 0) / n > 0.5:
        reasons.append("Most comments have no VFC level-3 label, so the nature of the need is unclear.")
    if reasons:
        return "refine", reasons
    issues, requests, positive = g.get("issue", 0), g.get("request", 0), g.get("positive", 0)
    if issues + requests == 0 or (positive / n >= 0.6 and issues / n < 0.15):
        return "retain", []
    return "improve", []


def _labels_text(pkg: EvidencePackage, group: str, k: int = 3) -> str:
    from services.requirement_evidence import label_group

    items = [l for l in pkg.vfc_level3_labels if label_group(l["label"]) == group][:k]
    return ", ".join(f"'{l['label']}' ({l['unique_feedback_count']} comments)" for l in items)


def _evidence_refs(pkg: EvidencePackage) -> list[EvidenceRef]:
    refs = [EvidenceRef(tier="DIRECT", source_table="feedback", source_key=fid, source_record_id=None,
                        evidence_scope="cluster_member") for fid in pkg.direct_feedback_ids]
    for i in pkg.supporting_evidence + pkg.contextual_evidence:
        refs.append(EvidenceRef(tier=i.tier, source_table={"customer_study": "customer_studies",
                                                           "public_owner_feedback": "public_feedback"}.get(i.kind, "competitor_evidence"),
                                source_key=i.key, source_record_id=i.source.record_id or None, evidence_scope="need_category",
                                requires_review=i.requires_review, note=i.mapping_note))
    for e in pkg.equipment_context:
        refs.append(EvidenceRef(tier="CONTEXTUAL", source_table="equipment", source_key=f"{e.option_code}@p{e.source_page}",
                                source_record_id=e.record_id, evidence_scope="vehicle_catalogue",
                                note="Current G60 option-list offering; not customer evidence."))
    return refs


def allowed_numbers(pkg: EvidencePackage) -> set[str]:
    nums = {str(pkg.unique_feedback_count), str(pkg.classification_count)}
    nums |= {str(v) for v in pkg.label_groups.values()}
    nums |= {str(m["unique_feedback_count"]) for m in pkg.markets}
    nums |= {str(l["unique_feedback_count"]) for l in pkg.vfc_level3_labels}
    nums |= {str(len(pkg.supporting_evidence)), str(len(pkg.contextual_evidence)), str(len(pkg.equipment_context))}
    nums.add(str(sum(i.kind in ("public_owner_feedback", "public_bmw_observation") for i in pkg.supporting_evidence)))
    nums.add(str(sum(i.kind == "customer_study" for i in pkg.supporting_evidence)))
    import re

    nums |= set(re.findall(r"\d+", pkg.cluster_label + " " + " ".join(l["label"] for l in pkg.vfc_level3_labels)
                           + " " + (pkg.mapping_note or "")))  # mapping note is generated from database counts
    nums |= {"5", "60"}  # "5 Series", "G60"
    return nums


def evidence_summary(pkg: EvidencePackage) -> str:
    markets = ", ".join(f"{m['country']} {m['unique_feedback_count']}" for m in pkg.markets[:4])
    g = pkg.label_groups
    publics = sum(i.kind in ("public_owner_feedback", "public_bmw_observation") for i in pkg.supporting_evidence)
    studies = sum(i.kind == "customer_study" for i in pkg.supporting_evidence)
    return (f"DIRECT: {pkg.unique_feedback_count} distinct G60 comments ({pkg.classification_count} classification rows; "
            f"markets {markets}). By original VFC level-3 label group (distinct comments; a comment can be in several): "
            f"issue {g.get('issue', 0)}, request {g.get('request', 0)}, positive {g.get('positive', 0)}. "
            f"SUPPORTING (need-category level, not proof of individual complaints): {publics} public BMW observations, "
            f"{studies} customer-study items. CONTEXTUAL: {len(pkg.contextual_evidence)} competitor observations, "
            f"{len(pkg.equipment_context)} G60 option-list entries.")


def base_assumptions(pkg: EvidencePackage) -> list[str]:
    out = [
        "Customer need is derived from the original BMW VFC category and level-3 labels; no AI analysis of the "
        "comment wording has been performed (counts are from the database).",
        "Target is the next-generation BMW 5 Series; no launch date or model year is assumed.",
        "Supporting evidence (public observations, customer studies) is linked at need-category level and does not "
        "prove any individual complaint; competitor observations are context only.",
        "All acceptance-criterion targets are proposals that require validation by BMW engineering and product "
        "management; no BMW specifications were used or invented.",
    ]
    total = sum(m["unique_feedback_count"] for m in pkg.markets) or 1
    if pkg.markets and pkg.markets[0]["unique_feedback_count"] / total >= 0.8:
        out.append(f"Evidence is concentrated in {pkg.markets[0]['country']} ({pkg.markets[0]['unique_feedback_count']} "
                   f"of {pkg.unique_feedback_count} comments); applicability to other markets must be confirmed.")
    if pkg.mapping_requires_review:
        out.append(f"The VFC-to-need mapping for this cluster requires review: {pkg.mapping_note or 'see Data Quality'}.")
    if pkg.equipment_context:
        out.append("Equipment context reflects the current G60 option list (German market) and indicates offering, "
                   "not customer demand.")
    return out


def generate_rule_based(pkg: EvidencePackage) -> RequirementDraft:
    if pkg.need_code == "OTHER":
        raise NotGeneratable("Comments without a usable VFC classification cannot be turned into a requirement.")
    if pkg.unique_feedback_count == 0:
        raise NotGeneratable("The selection has no direct customer evidence.")
    t = TEMPLATES[pkg.need_code]
    focus = _focus(pkg)
    framing, refine_reasons = decide_framing(pkg)
    g = pkg.label_groups
    issue_labels, request_labels, positive_labels = (_labels_text(pkg, k) for k in ("issue", "request", "positive"))
    assumptions = base_assumptions(pkg)

    if framing == "refine":
        title = f"Refine customer need: {pkg.cluster_label}"
        need = f"To be refined - G60 customers comment on {focus}; the specific underlying need is not yet identified."
        problem = (f"{pkg.unique_feedback_count} G60 comments relate to {focus}. "
                   + (f"Issue labels include {issue_labels}. " if issue_labels else "")
                   + (f"Request labels include {request_labels}. " if request_labels else "")
                   + "The evidence does not yet identify a specific function or failure mode.")
        statement = (f"PROBLEM-ORIENTED DRAFT (requires refinement): The next-generation BMW 5 Series shall address the "
                     f"customer concerns reported for {focus} on the G60. No technical solution is proposed until the "
                     f"specific need has been identified.")
        criteria = [AcceptanceCriterion(
            criterion="The specific customer need and affected function are identified and documented before a "
                      "technical requirement is written.",
            measurement="Review of the cluster's original comments (and AI extraction, if authorised) by product management")]
        rtype = "needs_refinement"
        notes = refine_reasons + ["Missing: which specific function or situation is affected.",
                                  "Missing: whether the need concerns usability, reliability, performance or availability.",
                                  "Missing: analysis of customer wording (AI extraction has not been run)."]
        assumptions.append("This is a problem-oriented placeholder, not a vehicle requirement.")
    elif framing == "retain":
        title = f"Retain valued {pkg.cluster_label}"
        need = f"G60 customers value {t.value} regarding {focus}; this must not regress."
        problem = (f"No dominant problem: {g.get('positive', 0)} of {pkg.unique_feedback_count} comments carry positive "
                   f"VFC labels ({positive_labels})."
                   + (f" {g.get('request', 0)} comments carry request labels ({request_labels})." if g.get("request") else "")
                   + (f" {g.get('issue', 0)} comments carry issue labels ({issue_labels})." if g.get("issue") else ""))
        statement = (f"The next-generation BMW 5 Series shall retain {t.value} for {focus}, as valued by G60 customers, "
                     f"with no regression relative to the G60.")
        criteria = [
            AcceptanceCriterion(criterion=f"Customer ratings for {focus} are not lower than for the G60.",
                                measurement=t.measurement, target="No regression vs. G60 (proposed - requires validation)"),
            AcceptanceCriterion(criterion=f"Any change to {focus} is reviewed against the G60 comments that praise it.",
                                measurement="Design review referencing the linked direct evidence IDs",
                                target="Review completed (proposed - requires validation)"),
        ]
        rtype, notes = f"retain_{t.requirement_type}", []
        if g.get("request") or g.get("issue"):
            assumptions.append(f"The {g.get('issue', 0)} issue-labelled and {g.get('request', 0)} request-labelled comments "
                               "in this cluster may justify a separate improvement requirement.")
    else:
        title = f"Improve {pkg.cluster_label}"
        need = f"G60 customers need {t.value} regarding {focus}."
        problem = (f"{g.get('issue', 0)} of {pkg.unique_feedback_count} G60 comments carry issue labels"
                   + (f" ({issue_labels})" if issue_labels else "")
                   + (f" and {g.get('request', 0)} carry request labels ({request_labels})" if g.get("request") else "")
                   + f", indicating {t.issue_kind}.")
        statement = (f"The next-generation BMW 5 Series shall provide {t.value} for {focus}, resolving the {t.issue_kind} "
                     f"reported in G60 customer feedback.")
        criteria = [
            AcceptanceCriterion(criterion=f"Fulfilment of the need for {focus} is demonstrated with target customers.",
                                measurement=t.measurement),
            AcceptanceCriterion(criterion=f"Issue-labelled VFC feedback for {focus} is lower than the G60 baseline in "
                                          f"comparable post-launch feedback.",
                                measurement="Distinct comments with issue labels per comparable feedback volume (VFC tracking)",
                                target=f"Baseline: {g.get('issue', 0)} of {pkg.unique_feedback_count} G60 comments in this "
                                       f"export; reduction target proposed - requires validation",
                                target_status="evidence_baseline"),
        ]
        rtype, notes = t.requirement_type, []
        if g.get("positive", 0) / pkg.unique_feedback_count >= 0.3:
            assumptions.append(f"Feedback is mixed: {g.get('positive', 0)} comments also carry positive labels; valued "
                               "aspects must be retained while resolving the issues.")
            criteria.append(AcceptanceCriterion(criterion=f"Aspects of {focus} that G60 customers praise do not regress.",
                                                measurement=t.measurement, target="No regression vs. G60 (proposed - requires validation)"))

    return RequirementDraft(
        title=title, source_cluster_id=pkg.cluster_id, source_insight_id=pkg.insight_id, need_code=pkg.need_code,
        customer_need=need, customer_problem=problem, proposed_requirement=statement, requirement_type=rtype,
        evidence_summary=evidence_summary(pkg), assumptions=assumptions, acceptance_criteria=criteria,
        generation_method="RULE_BASED", template_id=f"{pkg.need_code}:{framing}:v1", framing=framing,
        requires_refinement=framing == "refine", refinement_notes=notes, evidence=_evidence_refs(pkg),
        allowed_numbers=sorted(allowed_numbers(pkg)),
    )


# ---------------------------------------------------------------------------
# MODE B - LLM-assisted (gated; never runs by default)
# ---------------------------------------------------------------------------


class LLMGateStatus(BaseModel):
    enabled: bool
    reasons: list[str]  # why it is disabled (empty when enabled)


def llm_generation_status(settings: LLMSettings | None = None) -> LLMGateStatus:
    settings = settings or get_llm_settings()
    reasons = []
    if not settings.external_calls_authorized:
        reasons.append("External API use is not authorised (set PRODUCTMIND_ALLOW_EXTERNAL_LLM=true after approval "
                       "to send internal BMW feedback to an external provider).")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        reasons.append("No API key configured (ANTHROPIC_API_KEY).")
    return LLMGateStatus(enabled=not reasons, reasons=reasons)


class LLMCostEstimate(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    note: str = "Rough estimate (characters / 4); actual cost is logged after the call."


_LLM_SYSTEM = """You help BMW product managers phrase a vehicle requirement for the next-generation BMW 5 Series from evidence.
Write solution-neutral text. Use only the facts provided. Do not invent numbers, statistics, specifications, units,
dates or sources. Do not state that competitors prove anything about BMW customers. Acceptance criteria must name a
measurement method; leave targets as proposals requiring validation."""

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "customer_need": {"type": "string"}, "customer_problem": {"type": "string"},
        "proposed_requirement": {"type": "string"}, "assumptions": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {
            "type": "object", "properties": {"criterion": {"type": "string"}, "measurement": {"type": "string"}},
            "required": ["criterion", "measurement"], "additionalProperties": False}},
    },
    "required": ["title", "customer_need", "customer_problem", "proposed_requirement", "assumptions", "acceptance_criteria"],
    "additionalProperties": False,
}


def _llm_payload(pkg: EvidencePackage, base: RequirementDraft) -> str:
    return json.dumps({
        "rule_based_draft": base.model_dump(include={"title", "customer_need", "customer_problem", "proposed_requirement"}),
        "facts": {"need": pkg.need_name, "cluster": pkg.cluster_label, "unique_comments": pkg.unique_feedback_count,
                  "label_groups": pkg.label_groups, "vfc_level3_labels": pkg.vfc_level3_labels[:8]},
        "example_comments": [{"id": d.feedback_id, "text": d.text} for d in pkg.direct_evidence[:10]],
    }, ensure_ascii=False)


def estimate_llm_cost(pkg: EvidencePackage, settings: LLMSettings | None = None) -> LLMCostEstimate:
    from services.llm_client import LLMUsage, estimate_cost_usd

    settings = settings or get_llm_settings()
    base = generate_rule_based(pkg)
    usage = LLMUsage(input_tokens=(len(_LLM_SYSTEM) + len(_llm_payload(pkg, base))) // 4, output_tokens=1500)
    return LLMCostEstimate(model=settings.model, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                           estimated_cost_usd=estimate_cost_usd(settings.model, usage))


class LLMNotAuthorized(PermissionError):
    pass


def generate_llm_assisted(pkg: EvidencePackage, *, confirmed: bool, cost_acknowledged: LLMCostEstimate | None,
                          settings: LLMSettings | None = None, client=None) -> RequirementDraft:
    """Gated LLM rewrite of the rule-based draft. Raises LLMNotAuthorized unless
    every condition holds. ``client`` lets tests inject a fake (no network)."""
    settings = (settings or get_llm_settings()).model_copy(update={"use_fallbacks": False})
    gate = llm_generation_status(settings)
    if client is None and not gate.enabled:
        raise LLMNotAuthorized("; ".join(gate.reasons))
    if not settings.external_calls_authorized:
        raise LLMNotAuthorized("External API use is not authorised.")
    if not confirmed or cost_acknowledged is None:
        raise LLMNotAuthorized("The user must review the cost estimate and explicitly confirm generation.")
    if client is None:
        from services.llm_client import AnthropicLLMClient

        client = AnthropicLLMClient(settings)  # fallbacks forced off above
    base = generate_rule_based(pkg)
    response = client.complete_json(system=_LLM_SYSTEM, user=_llm_payload(pkg, base), schema=_LLM_SCHEMA,
                                    max_tokens=settings.max_tokens)
    if response.stop_reason == "refusal" or not response.text:
        raise RuntimeError("The model returned no usable draft; keep the rule-based draft.")
    data = json.loads(response.text)
    return base.model_copy(update={
        "title": data["title"], "customer_need": data["customer_need"], "customer_problem": data["customer_problem"],
        "proposed_requirement": data["proposed_requirement"],
        "assumptions": base.assumptions + [a for a in data["assumptions"] if a not in base.assumptions]
                       + ["Text drafted with LLM assistance from the facts above; review wording against the evidence."],
        "acceptance_criteria": [AcceptanceCriterion(criterion=c["criterion"], measurement=c["measurement"])
                                for c in data["acceptance_criteria"]] or base.acceptance_criteria,
        "generation_method": "LLM_ASSISTED", "template_id": None,
    })
