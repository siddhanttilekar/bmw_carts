"""Central configuration for BMW ProductMind AI.

All paths are resolved relative to the project root so scripts and tests work
from any working directory. Values can be overridden with environment variables
(see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseModel):
    project_root: Path = PROJECT_ROOT
    data_dir: Path = Field(default_factory=lambda: _env_path("PRODUCTMIND_DATA_DIR", PROJECT_ROOT / "data"))
    database_path: Path = Field(
        default_factory=lambda: _env_path("PRODUCTMIND_DB_PATH", PROJECT_ROOT / "database" / "productmind.db")
    )
    reports_dir: Path = Field(default_factory=lambda: _env_path("PRODUCTMIND_REPORTS_DIR", PROJECT_ROOT / "reports"))

    @property
    def internal_dir(self) -> Path:
        return self.data_dir / "internal"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def briefing_dir(self) -> Path:
        return self.data_dir / "briefing"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"


# ---------------------------------------------------------------------------
# Dataset registry. File names are the ones supplied for the hackathon.
# ---------------------------------------------------------------------------

# BMW internal feedback exports, one per vehicle (e-code).
FEEDBACK_FILES: dict[str, str] = {
    "F70": "F70_feedback_hackathon.xlsx",
    "G60": "G60_feedback_hackathon.xlsx",
    "G70": "G70_feedback_hackathon.xlsx",
}
FEEDBACK_SHEET = "Feedback_Explorer"
FEEDBACK_COLUMNS = [
    "ID",
    "Source",
    "Brand",
    "Sales Description",
    "Country",
    "Derivate (e-code)",
    "Engine Type",
    "Feedback Type",
    "Customer Feedback",
    "Vfc level2 3 Name",
    "Vfc level2 Name",
    "Vfc level3 Name",
]

CUSTOMER_STUDIES_FILE = "F70_G60_G68_G70_customer_studies.xlsx"
SALES_FILE = "sales_volumes.xlsx"

# Option-list (price/equipment catalogue) PDFs, one per vehicle.
OPTION_LIST_FILES: dict[str, str] = {
    "F70": "F70_OptionList.PDF",
    "G60": "G60_OptionList.PDF",
    "G70": "G70_OptionList.PDF",
}

# Consolidated workbook built from the files above. It is NOT a source of truth.
CONSOLIDATED_FILE = "BMW_G60_Complete_Prototype_Data.xlsx"

# Public (non-BMW) evidence.
PUBLIC_FEEDBACK_FILE = "BMW_5_Series_Public_Owner_Feedback.xlsx"
COMPETITOR_EVIDENCE_FILE = "BMW_G60_Public_Competitor_Evidence.xlsx"

# Briefing material.
BRIEFING_PDF_FILE = "BMW_Hackathon_Briefing_Short.pdf"
KEYNOTE_PPTX_FILE = "261010_BMW_Product_Management_Key_Note.pptx"

# Sales sheet name -> vehicle e-codes the volumes cover.
SALES_SHEET_VEHICLES: dict[str, list[str]] = {
    "F70": ["F70"],
    "G60_G68": ["G60", "G68"],
    "G70": ["G70"],
}
# Per the consolidated workbook's START HERE note: "2030 is future/planning
# figure, not historical sales."
SALES_PLANNING_YEARS = {2030}


def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# AI customer-intelligence layer (Part 2)
# ---------------------------------------------------------------------------


class LLMSettings(BaseModel):
    """LLM configuration. Credentials are resolved by the Anthropic SDK
    (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile)."""

    model: str = Field(default_factory=lambda: os.getenv("PRODUCTMIND_LLM_MODEL", "claude-opus-5-5"))
    effort: str = Field(default_factory=lambda: os.getenv("PRODUCTMIND_LLM_EFFORT", "medium"))
    batch_size: int = Field(default_factory=lambda: int(os.getenv("PRODUCTMIND_LLM_BATCH_SIZE", "10")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("PRODUCTMIND_LLM_MAX_TOKENS", "16000")))
    max_attempts: int = Field(default_factory=lambda: int(os.getenv("PRODUCTMIND_LLM_MAX_ATTEMPTS", "4")))
    max_rate_limit_wait_s: float = 120.0
    # Server-side refusal fallback to another model is OFF by default (project policy: never fall back
    # silently to another paid model). Opt in with PRODUCTMIND_LLM_FALLBACKS=1.
    use_fallbacks: bool = Field(default_factory=lambda: os.getenv("PRODUCTMIND_LLM_FALLBACKS", "0") == "1")
    # External API use must be explicitly authorised (internal BMW data leaves the machine).
    external_calls_authorized: bool = Field(
        default_factory=lambda: os.getenv("PRODUCTMIND_ALLOW_EXTERNAL_LLM", "").lower() in ("1", "true", "yes"))


# USD per million tokens (Anthropic first-party list prices; used for cost estimates only).
LLM_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
}

# Analyses in Part 2 are scoped to this vehicle.
ANALYSIS_VEHICLE = "G60"


def get_llm_settings() -> LLMSettings:
    return LLMSettings()
