# BMW ProductMind AI

AI-powered product decision support for the BMW 5 Series (G60). Hackathon prototype.

**Status: Part 4, evidence-backed vehicle requirement generation, is built on top of the Parts 1–3 data, intelligence and dashboard layers.** There is no prioritisation and no export yet. No real LLM extraction has been run, and external LLM use is disabled by default.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.sync_source_data mydata  # pick canonical copies by content, copy into data/
python -m scripts.initialize_database   # builds database/productmind.db + reports/
python -m scripts.status                # prints table counts and document status
streamlit run app.py                    # dashboard (reads SQLite; makes no LLM calls)
python -m scripts.customer_intelligence setup         # taxonomy, mappings, clusters, insights (no LLM)
python -m scripts.customer_intelligence demo --category CONTROLS
pytest                                  # 122 tests against the real datasets (LLM mocked)
```

`sync_source_data` identifies each file by its content (worksheets, header columns, PDF metadata), not only by name. Copies with identical SHA-256 collapse to one canonical file, and names such as `x(1).xlsx` are normalised. Originals are never moved or deleted, and a destination whose content differs is never overwritten. The original names and paths are recorded in `data/source_manifest.json` and stored in `source_documents.original_file_names` / `original_source_paths`.

`initialize_database` drops and rebuilds every table. Record IDs are deterministic (UUIDv5 of the provenance), so running it twice gives identical records and counts, as the tests check. The script exits non-zero if any file fails, any count check mismatches, or any of the 7 integration checks fails.

## Layout

```
app.py                         entry point (status / --rebuild)
core/config.py                 paths (env-overridable) + dataset registry
core/database.py               engine, schema reset, session scope
core/models.py                 SQLAlchemy ORM models
services/data_ingestion.py     parsers for every supplied file
services/data_validation.py    raw-file recount, integration checks, quality report
services/source_discovery.py   content-based file identification + canonical copy selection
services/feedback_analysis.py  counts, VFC grouping, Feedback Type distribution, filters
services/customer_insights.py  overview, category profiles, study values, sales context
services/evidence_retrieval.py original records by ID / category / text; public evidence
scripts/sync_source_data.py    copy canonical source files into data/
scripts/initialize_database.py build DB + report
data/internal/                 BMW-supplied files (feedback, studies, sales, option lists, consolidated workbook)
data/external/                 public evidence workbooks
data/briefing/                 briefing PDF + keynote PPTX
database/                      SQLite output
reports/                       data_quality_report.md / .json
tests/                         pytest suite
```

The `data/` and `database/` folders are git-ignored because they contain confidential hackathon data.

## Datasets and how they are handled

| File | Origin | Target table(s) |
|---|---|---|
| `F70/G60/G70_feedback_hackathon.xlsx` | internal | `feedback`, `feedback_classifications` |
| `F70_G60_G68_G70_customer_studies.xlsx` | internal | `customer_studies` (long format, one row per cell) |
| `sales_volumes.xlsx` | internal | `sales` (long format; 2030 flagged `is_planning_figure`) |
| `F70/G60/G70_OptionList.PDF` | internal | `source_document_pages`, `equipment`, `equipment_availability` |
| `BMW_G60_Complete_Prototype_Data.xlsx` | internal, consolidated | **not re-imported**; compared against the originals in `dataset_overlaps` |
| `BMW_5_Series_Public_Owner_Feedback.xlsx` | public | `public_feedback` |
| `BMW_G60_Public_Competitor_Evidence.xlsx` | public | `competitor_evidence` |
| `BMW_Hackathon_Briefing_Short.pdf`, keynote `.pptx` | briefing | `source_document_pages` |
| README / caveat sheets | all | `dataset_notes` |

### Data-integrity rules implemented

- **Provenance on every record.** Each record stores `record_id`, `source_document_id`, `source_file`, `source_sheet` or `source_page`, `source_row` (the 1-based Excel row) and `imported_at`. Customer-study and sales rows also store `source_column`, so every value points back to one cell.
- **Values are kept verbatim.** Missing values stay NULL. BMW VFC strings, including `no_class_found` and separator variants such as `Intervention: Malfunction`, are kept exactly as supplied. Columns prefixed `derived_` are computed flags, and they never overwrite source values.
- **Feedback grain.** One source row is one feedback ID combined with one VFC classification. A comment with several labels is stored once in `feedback` and once per row in `feedback_classifications`. Rows that share an ID are not treated as duplicates.
- **Duplicates are flagged, never merged.** Different IDs with identical normalised text, and repeated source URLs, are listed in `possible_duplicates` with `review_status = unreviewed`.
- **Single source of truth.** The original files are authoritative. At import time each sheet of the consolidated workbook is compared cell by cell with its original, and all of them currently match exactly. Its `Feedback Theme Counts` sheet is recomputed from the G60 rows and also matches.
- **Public data stays separate from internal data.** Public evidence is kept in separate tables with `data_origin = public_external`, and its source URLs are preserved.
- **Failures are documented.** A file that cannot be parsed is marked `failed` in `source_documents` with the reason. Its partial rows are rolled back and the other files still load.

## Customer-intelligence services (Part 2 preparation)

All functions are deterministic SQL over the stored data and return Pydantic models (`.model_dump()` for charts).

- **Scoped to G60 by default.** Every feedback function filters on `FeedbackFilter.vehicle_code="G60"`, so F70 and G70 feedback (imported, but for other vehicles) never enter G60 results.
- **Two counting units.** `feedback_ids` counts distinct customer comments. `classification_rows` counts source rows, one per ID × VFC label.
- **Filters.** You can filter by country, engine type (`ICE`/`BEVE`/`PHEV` as supplied), feedback channel, Feedback Type and VFC level 2. `None` selects missing values.
- **Feedback Type is not sentiment.** It labels each classification row, not each comment. 119 G60 comments carry both a `Likes` row and a `Defect` or `Difficult to Use` row. Source D never sets it, Source C never uses `Likes`/`Wants`, and 731 G60 rows have no value. The values are reported as supplied, and no sentiment is derived.
- **Sales for G60 exist only combined with G68.** `sales_context("G60")` returns those rows with `scope_label = "G60 + G68 combined (not G60-only)"` plus a warning, and it never splits volumes.
- **Public evidence is returned as its own record types** with `data_origin = public_external`. Sentiment there is the curator's label from the source file.

## Vehicle requirements (Part 4)

The **Vehicle Requirements** page (fifth in the dashboard) runs this flow: select a need (by insight, VFC cluster or need category) → review the evidence → **Generate Requirement Draft** → edit → Save Draft → Submit for Review → Approve or Reject. You can open the page from Customer Insights ("Draft Requirement"), from the Evidence Explorer ("Draft requirement from this selection") and from Customer Intelligence ("Open in Vehicle Requirements"); the selected insight or cluster is preserved.

| Module | Purpose |
|---|---|
| `services/requirement_evidence.py` | Evidence package per cluster, insight or need, built from real database IDs. DIRECT is the cluster's G60 comments. SUPPORTING is public BMW observations and BMW studies. CONTEXTUAL is competitor rows plus G60 option-list equipment. Supporting and contextual items are linked at need-category level and never treated as proof of an individual complaint. The package also lists missing and uncertain evidence. |
| `services/requirement_generator.py` | **RULE_BASED** (default, offline): solution-neutral templates per category, filled only with package values. Framing comes from the distinct-comment counts of the original VFC level-3 label groups: *improve*, *retain*, or *refine*. *Refine* produces a problem-oriented draft with the missing information listed and no technical solution; it is used for whole categories, catch-all VFC classes, fewer than 5 comments, or OVERALL. **LLM_ASSISTED** is optional and gated: it needs `PRODUCTMIND_ALLOW_EXTERNAL_LLM=true`, an API key, a displayed cost estimate and explicit confirmation. Model fallback is forced off. The model writes text only; evidence is attached by code. |
| `services/requirement_validator.py` | Checks: need clarity; vehicle scope (no assumed launch year); evidence IDs resolve, with direct IDs required to be G60 cluster members; not a copied complaint; no unsupported claims; no invented statistics (every number must exist in the evidence package); no numeric technical specifications; assumptions present; measurable criteria. Errors block submission and approval. |
| `services/requirement_review.py` | State machine: DRAFT → PENDING_REVIEW → APPROVED or REJECTED, with reopen back to DRAFT. There is no REJECTED → APPROVED path. Editing a non-draft returns it to DRAFT with a recorded transition. Approval requires a reason, a passing validation and a different person from the submitter (four-eyes rule). Every decision is stored with timestamp, previous and new status, reviewer and reason. Version 1 is always the original generated draft and is never modified. |

**Tables** (added additively; never dropped by ingestion rebuilds): `vehicle_requirements`, `requirement_evidence_links` (with `evidence_scope` set to cluster_member, need_category or vehicle_catalogue), `requirement_versions` and `requirement_review_history`. Requirement IDs (`REQ-` + 12 hex characters from a UUID) are generated once and stored, so they survive restarts, cluster rebuilds and ingestion rebuilds.

**Targets.** Acceptance criteria name a measurement method. Targets are always "proposed – requires validation" or an explicit G60 baseline taken from the database. No BMW engineering specifications are used or invented, and the target generation is "next-generation BMW 5 Series (launch timing not assumed)".

**LLM policy.** Server-side fallback to another model is now **off by default** for all LLM features (`PRODUCTMIND_LLM_FALLBACKS=1` opts in). External calls require `PRODUCTMIND_ALLOW_EXTERNAL_LLM=true`.

## Dashboard (Part 3)

`streamlit run app.py` opens four pages. They read the SQLite database through the existing services: `services/dashboard_queries.py` composes them, and `ui/data.py` caches the results.

| Page | Content |
|---|---|
| Customer Intelligence | KPIs, plus comments by need category, top VFC topics, market and engine type. Filters for market, engine type, need category, VFC category and Feedback Type all combine. |
| Evidence Explorer | Pick a need category or VFC cluster. Original comments appear as expandable cards showing the VFC labels and a file/sheet/row source reference, with literal text search and pagination. Supporting evidence (public BMW feedback, BMW studies) and contextual evidence (competitors) each have their own tab, with source URLs. |
| Customer Insights | Insight cards showing need, cluster, unique comment count, evidence IDs, limitations and the derived status. "View Supporting Evidence" opens the Explorer on that cluster. |
| Data Quality | Datasets, live record counts, integrity checks, missing data, unreviewed VFC mappings and cross-source links, and the eight known limitations with live figures. |

**Counting and safeguards.**
- **Counts.** Comments are always counted as `COUNT(DISTINCT feedback record)`. Classification rows are shown separately and labelled as such.
- **No sentiment.** Feedback Type appears only as raw counts labelled "not sentiment"; no positive or negative shares are shown.
- **Combined filters.** Need category, VFC category and Feedback Type must all match on the *same* classification row.
- **Cache freshness.** Cached reads are keyed on the database file's modification time and expire after 5 minutes.

**Insight status** is derived from the database, never assumed:
- `TEMPLATE`: text built from VFC classifications.
- `AI_ENRICHED`: real LLM extraction rows or validated LLM text exist for the cluster.
- `PENDING`: the insight failed validation.

On TEMPLATE cards the label list is titled "Original VFC level-3 labels" rather than "Customer problem", because many of those labels are praise.

**Stable identifiers.** Cluster IDs now come from canonical keys:
- `NEED-<code>` for need categories.
- `VFC-<sha1(VFC name)[:12]>` for VFC categories.
- `XN-<sha1(need | statement or member IDs)>` for extracted-need groups.

Insight IDs are `INS-<cluster_id>`, so rebuilding clusters keeps every ID and link unchanged; a test checks this.

**Visual system.** Dark navy surfaces, white type, and a single validated blue for data marks (`#3987E5` on `#111C33`). Evidence tiers and statuses always carry a text label, not just a colour, and every chart has a table view.

## AI Customer Intelligence (Part 2)

```bash
export ANTHROPIC_API_KEY=...            # or `ant auth login`; never commit keys
python -m scripts.customer_intelligence estimate --sample 20 --category CONTROLS   # no API call
python -m scripts.customer_intelligence extract  --sample 20 --category CONTROLS   # small real-data run
python -m scripts.customer_intelligence status
python -m scripts.customer_intelligence cluster  [--llm-grouping]
python -m scripts.customer_intelligence insights [--llm]
python -m scripts.customer_intelligence extract --all --confirm-full-run           # all 4,000: explicit only
python -m scripts.customer_intelligence resume JOB-...                              # after an interruption
```

**Modules**

| Module | Purpose |
|---|---|
| `services/need_taxonomy.py` | Curated taxonomy (18 needs), the VFC → need mapping covering all 222 G60 VFC values, and the cross-source mapping data |
| `services/llm_client.py` | Claude API client using structured JSON output, a cached system prompt, refusal fallback, usage and cost accounting, and classified errors |
| `services/ai_feedback_analysis.py` | Batched extraction with cache, checkpoints and resume, retries and rate-limit waits, a guard against unconfirmed full runs, and validation |
| `services/need_clustering.py` | Clusters at three levels (need category → VFC class → extracted need); counts come from SQL only |
| `services/cross_source_mapping.py` | Direct, supporting and contextual evidence links, each resolved against real records |
| `services/insight_generator.py` | Insights per cluster: facts come from SQL, the LLM writes text only, and every insight is checked against the database |
| `scripts/customer_intelligence.py` | CLI, including `demo` |

**Taxonomy.** The suggested 12 categories were checked against the 221 actual G60 VFC level-2 names. Six more categories were needed: Powertrain, Electric Range & Efficiency, Safety & Security, Vehicle Access & Keys, Climate, and Overall Impression. The mapping has 160 direct entries and 62 marked `review_required`, each with a reason. Broad BMW catch-all classes such as "Body equipment, overall technology" map to OVERALL rather than being forced into a specific need. Only `no_class_found` and empty VFC cells map to OTHER. Original VFC values are never changed.

**Extraction safeguards.**
- **No Feedback Type.** It is never sent to the model, so polarity is judged from the comment wording only, and `unknown` is used when unclear.
- **One comment, several needs.** A comment can produce several needs, each with a supporting quote that is checked to be a verbatim substring of the comment.
- **Unknown IDs are never stored.** Feedback IDs the model returns that weren't in the batch are discarded.
- **Cache key.** Each comment is cached under `sha256(feedback ID | text hash | model | prompt version)`, so an unchanged comment is never sent twice.
- **Checkpoints and logging.** Each batch is committed as it finishes. Every API attempt is logged in `ai_api_calls` with its tokens and estimated cost.
- **Model settings.** The default model is `claude-opus-5-5` at effort `medium`, with server-side refusal fallback enabled. Override with `PRODUCTMIND_LLM_*` (see `core/config.py`).

**Counting.** Every count is computed with SQL. `unique_feedback_count` counts distinct comments, the customer volume. `classification_count` counts VFC rows in the cluster's scope. The LLM never produces counts or evidence IDs. When the LLM writes insight text, that text is rejected if it contains a number, feedback ID or public ID that isn't in the facts.

**Evidence tiers.**
- **Direct:** internal BMW feedback.
- **Supporting:** public BMW owner observations, including the BMW rows of the competitor file, and BMW survey attributes.
- **Contextual:** non-BMW competitor rows, which are never treated as evidence about BMW customers.
- **Review flag:** 106 of the 378 links are marked `requires_review`.

**Database.** These tables are added additively by `core.database.ensure_ai_schema`: `customer_need_taxonomy`, `vfc_need_mappings`, `ai_processing_jobs`, `ai_api_calls`, `ai_feedback_extractions`, `extracted_customer_needs`, `customer_need_clusters`, `cluster_feedback_links`, `cross_source_evidence_links` and `generated_customer_insights`. They sit on a separate SQLAlchemy base with no foreign keys into ingestion tables, so re-running `initialize_database` never deletes AI results.

## Known limitations (details in `reports/data_quality_report.md`)

- **Briefing PDF.** Both pages are images (the PDF was made with "Microsoft Print to PDF") and have no text layer. OCR was not performed, so no text is stored for them.
- **Option lists.** The catalogues are German and AES-encrypted, but text extraction is permitted.
  - Equipment is parsed from text coordinates. Rows carrying an `SA-Nr.` code are extracted, together with the ■ (standard), □ (optional) and price cells for each model variant.
  - Standard-equipment pages and technical-data pages have no codes, so they are kept as page text only.
  - About 15 codes per catalogue have no symbols on their own line (for example colour groups that share one line). They carry a `parse_note`, and their availability is not inferred.
  - Two G70 package names are not in the text layer, so their `name` is NULL.
  - A variant missing from `equipment_availability` means "not listed", not "unavailable".
- **Customer studies.**
  - No sample sizes (n) are given. US values are column proportions, not counts.
  - The CN/EU scale is not stated; values look like 1–10 means.
- **Sales.** `G60_G68` combines G60 and G68 volumes. F70 has explicit 0 volumes in CN and US.
- **Feedback.**
  - `Source A–D` are anonymised channels and their meaning is not documented.
  - 200 G60 rows and 200 G70 rows have no VFC classification at all.
- **Cross-dataset joins.** Vehicle code is the only exact key. Mapping countries to sales or survey regions, and themes to VFC names, are candidate mappings that need confirmation in Part 2.
