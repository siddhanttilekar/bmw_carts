"""SQLAlchemy ORM models.

Every imported record carries provenance columns (``ProvenanceMixin``):
source document, file name, worksheet or PDF page, original row number where
applicable, a deterministic unique record identifier and the import timestamp.

Original values are stored verbatim. Columns prefixed ``derived_`` or named
``*_status`` are computed flags that never overwrite source values.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column


class Base(DeclarativeBase):
    pass


class ProvenanceMixin:
    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    @declared_attr
    def source_document_id(cls) -> Mapped[str]:  # noqa: N805
        return mapped_column(ForeignKey("source_documents.document_id"), nullable=False, index=True)

    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    source_sheet: Mapped[str | None] = mapped_column(String(255))
    source_page: Mapped[int | None] = mapped_column(Integer)
    # 1-based row number as shown in Excel (header row included).
    source_row: Mapped[int | None] = mapped_column(Integer)
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SourceDocument(Base):
    """Registry of every supplied file, including ones not imported row-by-row."""

    __tablename__ = "source_documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # File names / paths of the copies this canonical file was selected from
    # (from data/source_manifest.json; JSON list). NULL if no manifest entry.
    original_file_names: Mapped[str | None] = mapped_column(Text)
    original_source_paths: Mapped[str | None] = mapped_column(Text)
    # e.g. bmw_feedback, customer_studies, sales_volumes, consolidated_workbook
    dataset_type: Mapped[str | None] = mapped_column(String(64))
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # internal_bmw | public_external | briefing
    data_origin: Mapped[str] = mapped_column(String(32), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sheet_or_page_count: Mapped[int | None] = mapped_column(Integer)
    is_source_of_truth: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # imported | partially_extracted | no_text_layer | registered_only | failed
    ingestion_status: Mapped[str] = mapped_column(String(32), nullable=False)
    ingestion_notes: Mapped[str | None] = mapped_column(Text)
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SourceDocumentPage(Base, ProvenanceMixin):
    """Raw text of each PDF page / PPTX slide, with page number preserved."""

    __tablename__ = "source_document_pages"

    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    page_kind: Mapped[str] = mapped_column(String(16), nullable=False)  # pdf_page | pptx_slide
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(64), nullable=False)
    extraction_note: Mapped[str | None] = mapped_column(Text)


class DatasetNote(Base, ProvenanceMixin):
    """README / method / caveat sheets shipped with the workbooks (key-value)."""

    __tablename__ = "dataset_notes"

    note_key: Mapped[str | None] = mapped_column(Text)
    note_value: Mapped[str | None] = mapped_column(Text)


class Feedback(Base, ProvenanceMixin):
    """Internal BMW customer feedback, one row per original feedback ID.

    ``source_row`` is the first worksheet row in which the ID appears; every
    row is retained in ``feedback_classifications``.
    """

    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("source_file", "original_feedback_id"),)

    data_origin: Mapped[str] = mapped_column(String(32), nullable=False, default="internal_bmw")
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    feedback_source: Mapped[str | None] = mapped_column(String(64))  # "Source A".."Source D"
    brand: Mapped[str | None] = mapped_column(String(64))
    sales_description: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(8), index=True)
    engine_type: Mapped[str | None] = mapped_column(String(16))
    customer_feedback: Mapped[str] = mapped_column(Text, nullable=False)
    classification_row_count: Mapped[int] = mapped_column(Integer, nullable=False)


class FeedbackClassification(Base, ProvenanceMixin):
    """One row per source worksheet row: the original BMW VFC classification."""

    __tablename__ = "feedback_classifications"

    feedback_record_id: Mapped[str] = mapped_column(ForeignKey("feedback.record_id"), nullable=False, index=True)
    original_feedback_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    feedback_type: Mapped[str | None] = mapped_column(String(64))
    vfc_level2_3_name: Mapped[str | None] = mapped_column(Text)
    vfc_level2_name: Mapped[str | None] = mapped_column(Text, index=True)
    vfc_level3_name: Mapped[str | None] = mapped_column(Text)
    # classified | no_class_found | missing  (derived from the raw values above)
    derived_classification_status: Mapped[str] = mapped_column(String(32), nullable=False)


class PublicFeedback(Base, ProvenanceMixin):
    """Public owner feedback (paraphrased, curated). Kept apart from BMW data."""

    __tablename__ = "public_feedback"

    data_origin: Mapped[str] = mapped_column(String(32), nullable=False, default="public_external")
    original_public_id: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128))
    model_year: Mapped[str | None] = mapped_column(String(16))
    review_date: Mapped[str | None] = mapped_column(String(64))
    source_name: Mapped[str | None] = mapped_column(String(128))
    source_type: Mapped[str | None] = mapped_column(String(64))
    market: Mapped[str | None] = mapped_column(String(64))
    sentiment: Mapped[str | None] = mapped_column(String(32))
    theme: Mapped[str | None] = mapped_column(String(128))
    paraphrased_feedback: Mapped[str | None] = mapped_column(Text)
    product_implication: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    evidence_note: Mapped[str | None] = mapped_column(Text)
    # Parsed from evidence_note, e.g. "Same review as P001" -> (same_review, P001)
    derived_related_public_id: Mapped[str | None] = mapped_column(String(16))
    derived_relation_type: Mapped[str | None] = mapped_column(String(32))


class CompetitorEvidence(Base, ProvenanceMixin):
    __tablename__ = "competitor_evidence"

    data_origin: Mapped[str] = mapped_column(String(32), nullable=False, default="public_external")
    brand: Mapped[str | None] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(128))
    source_type: Mapped[str | None] = mapped_column(String(64))
    date_or_period: Mapped[str | None] = mapped_column(String(64))
    market: Mapped[str | None] = mapped_column(String(64))
    theme: Mapped[str | None] = mapped_column(String(128))
    sentiment: Mapped[str | None] = mapped_column(String(32))
    extracted_insight: Mapped[str | None] = mapped_column(Text)
    evidence_limitation: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    derived_is_bmw: Mapped[bool] = mapped_column(Boolean, nullable=False)


class CustomerStudy(Base, ProvenanceMixin):
    """Survey cross-tab cells in long format (one row per numeric cell)."""

    __tablename__ = "customer_studies"

    study_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    survey_market: Mapped[str] = mapped_column(String(32), nullable=False)
    vehicle_label: Mapped[str] = mapped_column(String(128), nullable=False)  # header as written
    vehicle_code: Mapped[str | None] = mapped_column(String(8), index=True)
    attribute: Mapped[str] = mapped_column(Text, nullable=False)
    measure: Mapped[str] = mapped_column(String(64), nullable=False)  # "Mean" or answer label
    measure_base: Mapped[str | None] = mapped_column(String(64))  # e.g. "% of column, sample base"
    # mean_score | column_proportion
    value_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source_column: Mapped[str] = mapped_column(String(4), nullable=False)


class SalesVolume(Base, ProvenanceMixin):
    __tablename__ = "sales"

    sales_group: Mapped[str] = mapped_column(String(32), nullable=False)  # sheet name, e.g. G60_G68
    vehicle_codes: Mapped[str] = mapped_column(String(64), nullable=False)  # "G60,G68"
    market: Mapped[str] = mapped_column(String(64), nullable=False)
    market_code: Mapped[str] = mapped_column(String(8), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    volume: Mapped[int | None] = mapped_column(Integer)
    is_planning_figure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # True when one volume covers several vehicles (G60_G68). Such rows must
    # never be presented as single-model volumes.
    is_combined_models: Mapped[bool] = mapped_column(Boolean, nullable=False)
    scope_label: Mapped[str] = mapped_column(String(128), nullable=False)
    source_column: Mapped[str] = mapped_column(String(4), nullable=False)


class Equipment(Base, ProvenanceMixin):
    """Option/equipment entries extracted from the option-list PDFs."""

    __tablename__ = "equipment"

    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    page_title: Mapped[str | None] = mapped_column(String(128))
    section: Mapped[str | None] = mapped_column(String(128))
    subsection: Mapped[str | None] = mapped_column(String(128))
    option_code: Mapped[str] = mapped_column(String(8), nullable=False, index=True)  # SA-Nr.
    parent_option_code: Mapped[str | None] = mapped_column(String(8))
    name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    price_gross_eur_text: Mapped[str | None] = mapped_column(String(64))
    price_net_eur_text: Mapped[str | None] = mapped_column(String(64))
    price_gross_eur: Mapped[float | None] = mapped_column(Float)
    has_variant_matrix: Mapped[bool] = mapped_column(Boolean, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    parse_note: Mapped[str | None] = mapped_column(Text)


class EquipmentAvailability(Base, ProvenanceMixin):
    """Per-variant availability cells (■ standard / □ optional / price text)."""

    __tablename__ = "equipment_availability"

    equipment_record_id: Mapped[str] = mapped_column(ForeignKey("equipment.record_id"), nullable=False, index=True)
    vehicle_code: Mapped[str] = mapped_column(String(8), nullable=False)
    option_code: Mapped[str] = mapped_column(String(8), nullable=False)
    variant: Mapped[str] = mapped_column(String(64), nullable=False)
    condition_text: Mapped[str | None] = mapped_column(Text)  # e.g. "In Verbindung mit M Sportpaket"
    line_price_gross_text: Mapped[str | None] = mapped_column(String(64))
    cell_raw: Mapped[str] = mapped_column(String(64), nullable=False)
    # standard | optional | optional_price_in_cell | no_extra_charge
    availability: Mapped[str] = mapped_column(String(32), nullable=False)


class DatasetOverlap(Base):
    """Comparison of consolidated-workbook sheets against original files."""

    __tablename__ = "dataset_overlaps"

    overlap_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    consolidated_file: Mapped[str] = mapped_column(String(255), nullable=False)
    consolidated_sheet: Mapped[str] = mapped_column(String(255), nullable=False)
    original_file: Mapped[str | None] = mapped_column(String(255))
    original_sheet: Mapped[str | None] = mapped_column(String(255))
    consolidated_rows: Mapped[int | None] = mapped_column(Integer)
    original_rows: Mapped[int | None] = mapped_column(Integer)
    # exact_copy | differs | derived_aggregate | template | metadata | no_original
    comparison_result: Mapped[str] = mapped_column(String(32), nullable=False)
    differing_cells: Mapped[int | None] = mapped_column(Integer)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    checked_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PossibleDuplicate(Base):
    """Candidate duplicates flagged for human review. Never merged automatically."""

    __tablename__ = "possible_duplicates"

    duplicate_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(64), nullable=False)
    rule: Mapped[str] = mapped_column(String(128), nullable=False)
    group_key: Mapped[str] = mapped_column(Text, nullable=False)
    record_ids: Mapped[str] = mapped_column(Text, nullable=False)  # comma-separated
    original_ids: Mapped[str | None] = mapped_column(Text)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, default="unreviewed")


class IngestionAudit(Base):
    """Expected (raw source) vs imported counts per sheet / document."""

    __tablename__ = "ingestion_audit"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    source_sheet: Mapped[str | None] = mapped_column(String(255))
    target_table: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    imported_count: Mapped[int] = mapped_column(Integer, nullable=False)
    count_basis: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # ok | mismatch
    checked_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
