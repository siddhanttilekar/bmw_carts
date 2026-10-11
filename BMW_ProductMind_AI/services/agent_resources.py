"""Registry of approved data resources for prioritization agents (Part 5).

Resources are fixed descriptions over EXISTING tables and services - no
datasets are copied, and nothing user-supplied is ever turned into SQL or
Python. ``available_records`` counts are computed live.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.models import CompetitorEvidence, CustomerStudy, Equipment, Feedback, PublicFeedback, SalesVolume


@dataclass(frozen=True)
class Resource:
    resource_id: str
    name: str
    description: str
    data_source: str
    available_fields: tuple[str, ...]
    limitations: tuple[str, ...]
    supported_analysis_methods: tuple[str, ...]


RESOURCES: dict[str, Resource] = {r.resource_id: r for r in [
    Resource("RES_G60_FEEDBACK", "G60 Internal Customer Feedback",
             "Original BMW G60 customer comments with VFC classifications (vehicle_code = G60 only).",
             "feedback, feedback_classifications (services.feedback_analysis / evidence_retrieval)",
             ("original_feedback_id", "customer_feedback", "country", "engine_type", "feedback_source",
              "vfc_level2_name", "vfc_level3_name", "feedback_type"),
             ("Heavily US-concentrated (about 90% of comments).",
              "Feedback Type is not sentiment; frequency is not severity.",
              "One comment can carry several VFC classifications - count distinct feedback IDs."),
             ("LOG_NORMALIZED_FEEDBACK_COUNT", "EVIDENCE_COVERAGE", "VALIDATED_MARKET_RELEVANCE")),
    Resource("RES_CUSTOMER_STUDIES", "Customer Studies",
             "BMW 2025 customer-study cells (CN/EU means, US answer proportions) linked to needs.",
             "customer_studies (services.customer_insights / cross_source_mapping)",
             ("study_name", "survey_market", "vehicle_label", "attribute", "measure", "value"),
             ("No sample sizes are available.", "CN/EU mean scale is not stated.", "China study has no G60."),
             ("EVIDENCE_COVERAGE",)),
    Resource("RES_SALES", "Sales Data",
             "Sales volumes by market and year (2030 = planning figure).",
             "sales (services.customer_insights.sales_context)",
             ("sales_group", "vehicle_codes", "market_code", "year", "volume", "is_planning_figure"),
             ("G60 volumes exist only combined with G68.", "2030 is a planning figure.",
              "Country-to-sales-region mapping is not validated."),
             ("VALIDATED_MARKET_RELEVANCE",)),
    Resource("RES_PUBLIC_FEEDBACK", "Public BMW Owner Feedback",
             "15 curated, paraphrased public owner observations (plus BMW rows of the competitor file).",
             "public_feedback, competitor_evidence (BMW rows)",
             ("original_public_id", "model", "market", "theme", "sentiment (as labelled)", "source_url"),
             ("Small, curated, not representative; several rows can share one review.",),
             ("EVIDENCE_COVERAGE",)),
    Resource("RES_COMPETITOR", "Competitor Evidence",
             "Curated public competitor observations (contextual only).",
             "competitor_evidence (non-BMW rows)",
             ("brand", "model", "market", "theme", "extracted_insight", "source_url"),
             ("Qualitative and small; observation counts are not competitor performance.",
              "Never proof of BMW customer dissatisfaction."),
             ("EVIDENCE_COVERAGE",)),
    Resource("RES_EQUIPMENT", "G60 Equipment / Option List",
             "G60 option-list entries (German market, July 2026) linked to needs.",
             "equipment, equipment_availability",
             ("option_code", "name", "section", "price_gross_eur_text", "variant availability"),
             ("Shows current offering and price, not customer demand.",),
             ("EVIDENCE_COVERAGE",)),
    Resource("RES_REQUIREMENTS", "Saved Vehicle Requirements",
             "Saved Part 4 requirements with their evidence links (the items being prioritized).",
             "vehicle_requirements, requirement_evidence_links",
             ("requirement_id", "title", "review_status", "source_cluster_id", "evidence links"),
             ("Evidence links are a snapshot taken when the requirement was created.",),
             ("LOG_NORMALIZED_FEEDBACK_COUNT", "EVIDENCE_COVERAGE", "HUMAN_SCORE", "VALIDATED_MARKET_RELEVANCE",
              "AI_ASSESSMENT", "COMPLAINT_FREQUENCY", "CUSTOMER_REQUEST_FREQUENCY", "COMBINED_CUSTOMER_DEMAND",
              "DOCUMENTED_CUSTOM_METRIC")),
    Resource("RES_HUMAN_ASSESSMENTS", "Human Assessments",
             "Scores (0-100) with rationale entered by product managers per requirement and agent.",
             "priority_assessments",
             ("score", "rationale", "assessor_id", "assessment_timestamp", "evidence_references"),
             ("Subjective judgment; quality depends on the documented rationale.",),
             ("HUMAN_SCORE",)),
    Resource("RES_AI_OBSERVATIONS", "AI-Validated Customer Observations",
             "Typed observations (complaint / improvement request / positive / neutral / unclear) extracted once from "
             "G60 comments; each has a verbatim excerpt validated against the database.",
             "ai_feedback_observations, evidence_selection_runs, requirement_evidence_matches, evidence_match_reviews",
             ("observation_id", "original_feedback_id", "observation_type", "customer_need", "vehicle_system",
              "evidence_excerpt", "confidence", "taxonomy_category"),
             ("AI-identified evidence: validated for existence and verbatim excerpts, not automatically human-reviewed.",
              "Scores are UNSCORED until every direct comment of the requirement has been extracted."),
             ("COMPLAINT_FREQUENCY", "CUSTOMER_REQUEST_FREQUENCY", "COMBINED_CUSTOMER_DEMAND", "EVIDENCE_COVERAGE")),
    Resource("RES_UPLOADED_DOCUMENTS", "Agent Uploaded Documents",
             "PDF / XLSX / CSV / TXT / MD files uploaded to this agent only (parsed to text; never executed).",
             "agent_documents, agent_document_chunks (scoped to the owning agent)",
             ("document_id", "file_name", "passage location", "passage text"),
             ("Untrusted content: only keyword matches are used; instructions inside documents are ignored.",
              "Visible to the agent that owns the upload only."),
             ("EVIDENCE_COVERAGE", "DOCUMENTED_CUSTOM_METRIC")),
]}


class ResourceView(BaseModel):
    resource_id: str
    name: str
    description: str
    data_source: str
    available_fields: list[str]
    limitations: list[str]
    supported_analysis_methods: list[str]
    available_records: int | None


def _count(session: Session, resource_id: str) -> int | None:
    from core.evidence_models import AgentDocument, AIFeedbackObservation
    from core.prioritization_models import PriorityAssessment
    from core.requirement_models import VehicleRequirement

    stmt = {
        "RES_G60_FEEDBACK": select(func.count()).select_from(Feedback).where(Feedback.vehicle_code == "G60"),
        "RES_CUSTOMER_STUDIES": select(func.count()).select_from(CustomerStudy).where(CustomerStudy.vehicle_code == "G60"),
        "RES_SALES": select(func.count()).select_from(SalesVolume),
        "RES_PUBLIC_FEEDBACK": select(func.count()).select_from(PublicFeedback),
        "RES_COMPETITOR": select(func.count()).select_from(CompetitorEvidence).where(CompetitorEvidence.derived_is_bmw.is_(False)),
        "RES_EQUIPMENT": select(func.count()).select_from(Equipment).where(Equipment.vehicle_code == "G60"),
        "RES_REQUIREMENTS": select(func.count()).select_from(VehicleRequirement),
        "RES_HUMAN_ASSESSMENTS": select(func.count()).select_from(PriorityAssessment).where(PriorityAssessment.superseded.is_(False)),
        "RES_AI_OBSERVATIONS": select(func.count()).select_from(AIFeedbackObservation),
        "RES_UPLOADED_DOCUMENTS": select(func.count()).select_from(AgentDocument).where(AgentDocument.active.is_(True)),
    }.get(resource_id)
    return session.scalar(stmt) if stmt is not None else None


def list_resources(session: Session | None = None) -> list[ResourceView]:
    return [ResourceView(resource_id=r.resource_id, name=r.name, description=r.description, data_source=r.data_source,
                         available_fields=list(r.available_fields), limitations=list(r.limitations),
                         supported_analysis_methods=list(r.supported_analysis_methods),
                         available_records=_count(session, r.resource_id) if session is not None else None)
            for r in RESOURCES.values()]


def get_resource(resource_id: str) -> Resource:
    if resource_id not in RESOURCES:
        raise ValueError(f"Unknown resource {resource_id!r}; only registered resources can be assigned")
    return RESOURCES[resource_id]
