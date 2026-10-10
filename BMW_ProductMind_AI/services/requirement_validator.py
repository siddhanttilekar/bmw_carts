"""Requirement quality validation (Part 4).

Checks (severity "error" blocks approval; "warning" is shown to reviewers):

 1 NEED_CLEAR              customer need present and stated as a need
 2 VEHICLE_SCOPE           vehicle model + target generation set; no assumed launch year
 3 EVIDENCE_VALID          >=1 direct comment; every evidence key resolves to a real record;
                           direct IDs are G60 members of the source cluster; category-level
                           evidence is never scoped as an individual complaint
 4 NOT_COPIED_COMPLAINT    requirement statement is not a copy of a customer comment
 5 UNSUPPORTED_CLAIMS      no generalising or proof claims ("most customers", "studies show")
 6 INVENTED_STATISTICS     every number/percentage comes from the evidence package
 7 UNSUPPORTED_SPEC        no numeric technical specifications (units) in the requirement;
                           in acceptance targets only when marked as proposed
 8 ASSUMPTIONS             assumptions present (and refinement notes for refinement drafts)
 + CRITERIA                acceptance criteria present, each with a measurement method
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.ai_models import ClusterFeedbackLink, CrossSourceEvidenceLink
from core.models import CompetitorEvidence, CustomerStudy, Equipment, Feedback, PublicFeedback
from services.requirement_evidence import build_evidence_package
from services.requirement_generator import allowed_numbers

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Digits attached to letters or hyphens ("G60", "F70", "level-3") are identifiers, not numbers.
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.,-])\d+(?:[.,]\d+)?")
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
SPEC_RE = re.compile(
    r"(?<![A-Za-z0-9.,-])\d+(?:[.,]\d+)?\s*(?:kWh|kW|km/h|km|mm|cm|ms|sec(?:onds)?|s\b|min(?:utes)?|hours?|h\b|Nm|dB|°C|Hz|px|"
    r"inch(?:es)?|\"|%|W\b|V\b|A\b|l/100|kg|g\b|lux|nits?)", re.IGNORECASE)
CLAIM_PATTERNS = [
    (re.compile(r"\b(all|most|every|majority of|the majority of)\s+(?:G60\s+)?(customers|owners|drivers|users)\b", re.I),
     "generalises beyond the evidence"),
    (re.compile(r"\bstud(?:y|ies)\s+(show|shows|prove|proves|confirm|confirms)\b", re.I), "claims proof from studies"),
    (re.compile(r"\b(proven|proves|statistically significant|scientifically)\b", re.I), "claims proof"),
    (re.compile(r"\bcompetitors?\b.{0,40}\b(prove|proves|show|shows|demonstrate|confirm)", re.I),
     "uses competitor observations as proof"),
    (re.compile(r"\b(best[- ]in[- ]class|market[- ]leading|industry[- ]leading|world[- ]class)\b", re.I),
     "unsupported superlative"),
]
TEXT_FIELDS = ("title", "customer_need", "customer_problem", "proposed_requirement")


class ValidationIssue(BaseModel):
    code: str
    severity: str  # error | warning
    field: str
    message: str


class ValidationResult(BaseModel):
    passed: bool
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _shares_ngram(a: str, b: str, n: int = 8) -> bool:
    wa, wb = _words(a), _words(b)
    grams = {tuple(wa[i:i + n]) for i in range(len(wa) - n + 1)}
    return any(tuple(wb[i:i + n]) in grams for i in range(len(wb) - n + 1))


def _resolve(session: Session, ref: dict, cluster_id: str, members: set[str]) -> str | None:
    """Return an error message if the evidence reference does not resolve."""
    table, key = ref["source_table"], ref["source_key"]
    if ref["tier"] == "DIRECT":
        if table != "feedback":
            return f"direct evidence must be internal feedback, got {table}:{key}"
        fb = session.scalar(select(Feedback).where(Feedback.original_feedback_id == key))
        if fb is None:
            return f"feedback ID {key} does not exist"
        if fb.vehicle_code != "G60":
            return f"feedback ID {key} is {fb.vehicle_code}, not G60"
        if key not in members:
            return f"feedback ID {key} is not a member of source cluster {cluster_id}"
        if ref.get("evidence_scope") != "cluster_member":
            return f"direct evidence {key} must be scoped as cluster_member"
        return None
    if ref.get("evidence_scope") == "cluster_member":
        return f"{table}:{key} is category-level evidence and cannot be scoped as an individual complaint"
    if table == "public_feedback":
        ok = session.scalar(select(PublicFeedback.record_id).where(PublicFeedback.original_public_id == key))
    elif table == "competitor_evidence":
        row = key.removeprefix("COMP-R")
        ok = row.isdigit() and session.scalar(select(CompetitorEvidence.record_id).where(CompetitorEvidence.source_row == int(row)))
    elif table == "customer_studies":
        study, _, attribute = key.partition("::")
        ok = session.scalar(select(CustomerStudy.record_id).where(CustomerStudy.study_name == study,
                                                                  CustomerStudy.attribute == attribute).limit(1))
    elif table == "equipment":
        ok = ref.get("source_record_id") and session.get(Equipment, ref["source_record_id"])
    else:
        return f"unknown evidence table {table}"
    if not ok:
        return f"{table}:{key} does not resolve to a database record"
    if table in ("public_feedback", "competitor_evidence", "customer_studies"):
        linked = session.scalar(select(CrossSourceEvidenceLink.link_id).where(
            CrossSourceEvidenceLink.source_key == key, CrossSourceEvidenceLink.evidence_tier == ref["tier"].lower()).limit(1))
        if not linked:
            return f"{table}:{key} is not a curated {ref['tier'].lower()} evidence link"
    return None


def validate_requirement(session: Session, req: dict) -> ValidationResult:
    """``req``: requirement fields (see RequirementDraft / review.requirement_as_dict)
    including ``evidence`` (list of evidence-ref dicts) and ``source_cluster_id``."""
    issues: list[ValidationIssue] = []

    def add(code, severity, field, message):
        issues.append(ValidationIssue(code=code, severity=severity, field=field, message=message))

    # 1 need
    need = (req.get("customer_need") or "").strip()
    if len(_words(need)) < 5:
        add("NEED_CLEAR", "error", "customer_need", "Customer need is missing or too short to be clear.")
    # 2 scope
    if not (req.get("vehicle_model") or "").strip() or "5 Series" not in req.get("vehicle_model", ""):
        add("VEHICLE_SCOPE", "error", "vehicle_model", "Vehicle model must be the BMW 5 Series (G60 baseline).")
    target = req.get("target_generation") or ""
    if not target.strip():
        add("VEHICLE_SCOPE", "error", "target_generation", "Target generation is missing.")
    for field in ("target_generation",) + TEXT_FIELDS:
        if YEAR_RE.search(req.get(field) or ""):
            add("VEHICLE_SCOPE", "error", field, "Contains a year; no launch date or model year may be assumed.")

    # 3 evidence
    try:
        pkg = build_evidence_package(session, cluster_id=req["source_cluster_id"])
    except ValueError as exc:
        add("EVIDENCE_VALID", "error", "source_cluster_id", str(exc))
        return ValidationResult(passed=False, issues=issues)
    members = set(session.scalars(select(ClusterFeedbackLink.original_feedback_id)
                                  .where(ClusterFeedbackLink.cluster_id == req["source_cluster_id"])))
    refs = req.get("evidence") or []
    if not any(r["tier"] == "DIRECT" for r in refs):
        add("EVIDENCE_VALID", "error", "evidence", "At least one direct customer comment is required.")
    bad = [m for m in (_resolve(session, r, req["source_cluster_id"], members) for r in refs) if m]
    for message in bad[:10]:
        add("EVIDENCE_VALID", "error", "evidence", message)
    if len(bad) > 10:
        add("EVIDENCE_VALID", "error", "evidence", f"... {len(bad) - 10} more invalid evidence references")

    # 4 copied complaint
    statement = req.get("proposed_requirement") or ""
    direct_ids = [r["source_key"] for r in refs if r["tier"] == "DIRECT"]
    texts = list(session.scalars(select(Feedback.customer_feedback).where(Feedback.original_feedback_id.in_(direct_ids))))
    for text in texts:
        if SequenceMatcher(None, statement.lower(), text.lower()).ratio() >= 0.75 or _shares_ngram(statement, text):
            add("NOT_COPIED_COMPLAINT", "error", "proposed_requirement",
                "The requirement statement reproduces a customer comment; rephrase it as a requirement.")
            break

    # 5 claims, 6 statistics, 7 specifications
    allowed = allowed_numbers(pkg)
    criteria = req.get("acceptance_criteria") or []
    texts_to_check = {f: req.get(f) or "" for f in TEXT_FIELDS + ("evidence_summary",)}
    texts_to_check["assumptions"] = " ".join(req.get("assumptions") or [])
    for i, c in enumerate(criteria):
        texts_to_check[f"acceptance_criteria[{i}].criterion"] = c.get("criterion", "")
        texts_to_check[f"acceptance_criteria[{i}].measurement"] = c.get("measurement", "")
    for field, text in texts_to_check.items():
        for pattern, why in CLAIM_PATTERNS:
            if pattern.search(text):
                add("UNSUPPORTED_CLAIMS", "error", field, f"Statement {why}: '{pattern.search(text).group(0)}'.")
        if PERCENT_RE.search(text):
            add("INVENTED_STATISTICS", "error", field, f"Percentage '{PERCENT_RE.search(text).group(0)}' is not computed from the evidence.")
        unknown = sorted(set(NUMBER_RE.findall(PERCENT_RE.sub(" ", text))) - allowed)
        if unknown:
            add("INVENTED_STATISTICS", "error", field, f"Numbers not found in the evidence package: {', '.join(unknown[:5])}.")
        spec_text = text
        for label in [pkg.cluster_label] + [l["label"] for l in pkg.vfc_level3_labels]:  # quoted source labels
            spec_text = spec_text.replace(label, " ")
        if field in TEXT_FIELDS and SPEC_RE.search(spec_text):
            add("UNSUPPORTED_SPEC", "error", field, f"Technical specification '{SPEC_RE.search(spec_text).group(0)}' is not supported by evidence.")
    for i, c in enumerate(criteria):
        target = c.get("target", "")
        if SPEC_RE.search(target) or (NUMBER_RE.search(target) and c.get("target_status") != "evidence_baseline"):
            if "requires validation" in target.lower():
                add("UNSUPPORTED_SPEC", "warning", f"acceptance_criteria[{i}].target",
                    "Numeric target is a proposal; it must be validated by engineering before approval of the specification.")
            else:
                add("UNSUPPORTED_SPEC", "error", f"acceptance_criteria[{i}].target",
                    "Numeric target is not derived from evidence; mark it 'proposed - requires validation'.")
        if c.get("target_status") == "evidence_baseline":
            unknown = sorted(set(NUMBER_RE.findall(target)) - allowed)
            if unknown:
                add("INVENTED_STATISTICS", "error", f"acceptance_criteria[{i}].target",
                    f"Baseline numbers not in the evidence package: {', '.join(unknown)}.")

    # 8 assumptions + criteria
    if not [a for a in req.get("assumptions") or [] if a.strip()]:
        add("ASSUMPTIONS", "error", "assumptions", "At least one assumption must be documented.")
    if req.get("requires_refinement") and not req.get("refinement_notes"):
        add("ASSUMPTIONS", "error", "refinement_notes", "A draft requiring refinement must list the missing information.")
    if not criteria:
        add("CRITERIA", "error", "acceptance_criteria", "At least one acceptance criterion is required.")
    for i, c in enumerate(criteria):
        if not (c.get("measurement") or "").strip():
            add("CRITERIA", "warning", f"acceptance_criteria[{i}]", "Criterion has no measurement method; it is not measurable yet.")
    if req.get("requires_refinement"):
        add("NEED_CLEAR", "warning", "customer_need", "Draft requires refinement before it can become a vehicle requirement.")
    return ValidationResult(passed=not any(i.severity == "error" for i in issues), issues=issues)
