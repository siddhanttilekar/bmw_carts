# BMW ProductMind AI

AI-powered product decision support for the BMW 5 Series (G60). Hackathon prototype.

**Status: Parts 1–6, Upgrade A (DISCOVER / DEFINE / PRIORITIZE / DECIDE), the 2.0 evidence-first upgrade and
Stage 1 (OpenAI provider) are complete.** The OpenAI connection has been verified with one test request; the
20-comment extraction sample has not been run yet. External LLM use is disabled unless explicitly enabled.

## Quick start (new teammate)

The BMW datasets are confidential and are **not in this repository**. Get the source files from the project
owner through a private channel and put them in a folder named `mydata/` inside `BMW_ProductMind_AI/`.

```bash
cd BMW_ProductMind_AI
python3 -m venv .venv && source .venv/bin/activate   # always use this env (conda "base" lacks the packages)
pip install -r requirements.txt
python -m scripts.sync_source_data mydata  # pick canonical copies by content, copy into data/
python -m scripts.initialize_database   # builds database/productmind.db + reports/
python -m scripts.customer_intelligence setup         # taxonomy, mappings, clusters, insights (no LLM)
python -m scripts.status                # prints table counts and document status
streamlit run app.py                    # dashboard on http://localhost:8501
pytest                                  # 227 tests against the real datasets (LLM always mocked)
```

Optional AI features (never put a key in a file that is committed):

```bash
export OPENAI_API_KEY=...  PRODUCTMIND_LLM_PROVIDER=openai  PRODUCTMIND_ALLOW_EXTERNAL_LLM=1  PRODUCTMIND_MAX_API_SPEND_USD=26
python -m scripts.customer_intelligence test-openai            # one tiny paid request, asks first
python -m scripts.customer_intelligence observe --sample 20    # cost estimate only; add --approve-usd 0.10 to run
```

Next step for the team: run and review the 20-comment sample (DISCOVER → "Run AI extraction" → "Review extraction
quality") before any full extraction.

## LLM provider (Stage 1: OpenAI)

The default provider is **OpenAI** (`services/llm_client.py::OpenAILLMClient`, Responses API, strict JSON-schema
structured outputs, `store=False`, SDK retries off so every attempt is logged and costed). The Anthropic client is
kept and used **only** with `PRODUCTMIND_LLM_PROVIDER=anthropic`; `make_llm_client` never falls back to the other
provider. All features use the same `complete_json` interface, so custom AI agents, requirement generation,
the decision assistant and debate can adopt it later without changes to the client.

| Setting | Default |
|---|---|
| `PRODUCTMIND_LLM_PROVIDER` | `openai` |
| `PRODUCTMIND_LLM_MODEL` | `gpt-5.6-luna` |
| `PRODUCTMIND_LLM_EFFORT` (OpenAI reasoning effort) | `low` |
| `PRODUCTMIND_MAX_API_SPEND_USD` (application budget, estimate-based) | `26` |
| `PRODUCTMIND_ALLOW_EXTERNAL_LLM` | off |
| `OPENAI_API_KEY` | shell only - never committed, never displayed |

Model choice (checked 2026-10-11 on developers.openai.com): `gpt-5.6-luna` - Structured Outputs supported,
Responses + Chat Completions, $0.20 input / $0.02 cached input / $1.20 output per 1M tokens, no deprecation notice,
and OpenAI's listed replacement for the retiring `gpt-5-nano` / `gpt-4.1-nano`. `gpt-5-mini` was rejected because
its snapshot shuts down on 2026-12-11. Reasoning tokens are billed as output and are included in the cost estimate.
Costs shown by the app are list-price estimates from token usage, not provider billing.

Requirement generation (Part 4) and the decision-assistant rephrasing (Part 6) still have their own disabled,
Anthropic-gated LLM modes; they were not changed in this stage.

```bash
export OPENAI_API_KEY=...  PRODUCTMIND_LLM_PROVIDER=openai  PRODUCTMIND_ALLOW_EXTERNAL_LLM=1  PRODUCTMIND_MAX_API_SPEND_USD=26
python -m scripts.customer_intelligence test-openai      # config + model check, then ONE paid request after typing YES
```

## ProductMind AI 2.0 - Evidence-First Agentic Prioritization

**AI identifies evidence. Python calculates scores. Humans make decisions.**

| Layer | Module | What it does |
|---|---|---|
| Extraction (`observe-v1`) | `services/ai_feedback_analysis.py` | One pass per G60 comment → typed observations (COMPLAINT / IMPROVEMENT_REQUEST / POSITIVE / NEUTRAL / UNCLEAR) with customer need, system, confidence, taxonomy category and an evidence excerpt. Excerpts must be found verbatim (case-sensitive) or the observation is rejected and logged. Observation IDs are generated in Python. Cache key = feedback ID + text hash + model + model version + prompt version + taxonomy version; validated entries are never overwritten (new versions sit next to old ones in `observation_versions`). Shares batching, retries, checkpoint/resume and usage logging with Part 2. |
| Cost control | `services/llm_budget.py`, `services/evidence_extraction.py` | $26 application budget (`PRODUCTMIND_LLM_BUDGET_USD`), spend = sum of logged calls incl. retries, pre-request estimate (measured from the sample once it ran), explicit `approved_max_usd`, per-batch ceiling guard (job stops resumably), 20-comment representative sample required before a confirmed full run. Fallback model off. |
| Selection (`select-v1`) | `services/ai_evidence_selector.py` | Per requirement version: taxonomy matching on its direct comments → complaint / request / positive / counter-evidence / uncertain ID lists (distinct feedback IDs; combined demand = union). Optional paid LLM semantic matching re-judges only uncertain matches and creates a new immutable selection. Human reviews (accept / reject / uncertain + justification, append-only) override every AI run. |
| Scoring | `services/agent_scoring.py`, `services/custom_metrics.py` | `COMPLAINT_FREQUENCY`, `CUSTOMER_REQUEST_FREQUENCY`, `COMBINED_CUSTOMER_DEMAND` (AI_ANALYSIS): score = 100·ln(1+n)/ln(1+N), N = 4000 fixed (`log-feedback-v1`) - a normalised frequency index, not a % of customers. UNSCORED without a selection or with incomplete extraction coverage. `EVIDENCE_COVERAGE` v1.1 also covers AI observations and uploads. `DOCUMENTED_CUSTOM_METRIC` runs only code-implemented, admin-approved metrics. All versions in `scoring_method_versions`. |
| Agents | `services/agent_configuration.py`, `services/agent_documents.py` | Builder fields: Name, Objective, Custom Instructions, Resources (+ PDF/XLSX/CSV/TXT/MD uploads, parsed to text, scoped to the owning agent), Context, Weight, Execution Method + approved scoring rule. Instructions are never executed and never change a formula. |
| Snapshots | `services/dynamic_prioritization.py` | Each run freezes evidence (selection, observation version with model/prompt/taxonomy, validated IDs, counts, normalisation). `verify_reproducibility` recomputes every score from stored inputs; `reanalyze_with_new_evidence` makes a new run plus a comparison - old runs never change. |
| Decide | `decision_assistant`, `agent_chat`, `agent_debate` | Explanations quote frozen counts, counter-evidence, missing evidence and the Python calculation; agents abstain when evidence is insufficient. |

UI: DISCOVER (extraction status, budget, sample run + quality review, AI counts per need), DEFINE (evidence selection, positive/negative observations, match review, history), PRIORITIZE (score calculation per agent, evidence readiness, custom agent builder), DECIDE (calculations, evidence-aware debate, history & reanalysis, reproducibility check, human approval, export).

```bash
python -m scripts.customer_intelligence observe --sample 20                      # estimate only - nothing is sent
python -m scripts.customer_intelligence observe --sample 20 --approve-usd 0.50   # paid sample (needs PRODUCTMIND_ALLOW_EXTERNAL_LLM=1 + key)
python -m scripts.customer_intelligence observe --all --confirm-full-run --approve-usd 20   # only after a completed sample
python -m scripts.customer_intelligence budget
python -m scripts.customer_intelligence select-evidence --user YOUR_ID
python -m scripts.customer_intelligence approve-metric DOCUMENT_PASSAGE_SHARE --admin ID --note "..."
```

Security: comments and uploads are untrusted data (sent as JSON values, prompts say to ignore embedded instructions); no user text is executed as Python/SQL; agents cannot change requirement status; API keys are never shown or stored; API-call logs hold IDs and token counts, not comment text.

## Simplified Workflow UI (Upgrade A)

The default Streamlit experience is now organized for nontechnical product managers:

| Stage | Purpose |
|---|---|
| `DISCOVER` | Understand BMW G60 customer needs, inspect representative evidence on demand, and choose a need to work on. |
| `DEFINE` | Generate an evidence-based requirement draft, edit it, save it, and submit it for review. |
| `PRIORITIZE` | Review active analysis agents, validate weights, collect human assessments when needed, and run prioritization. |
| `DECIDE` | Explain rankings, compare requirements, run debates and what-if simulations, and export reports. |

Selections persist across stages using Streamlit session state: selected need → requirement → prioritization run. Advanced diagnostic and legacy technical pages are still registered in the app navigation and have not been deleted.

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

## Decision Intelligence (Part 6)

The Decision Intelligence page has its own navigation section and five screens: A Overview, B Ask the assistant, C Agent debate, D What-if and scenarios, E Management reports. Every answer is bound to one saved prioritisation run. Nothing on the page changes official scores, agents, requirements or review decisions.

| Module | Purpose |
|---|---|
| `services/decision_assistant.py` | Controller. Routes 8 intents (explain priority, explain agent, evidence, compare, missing, scenario, challenge, reliability) to deterministic handlers. Unclear free text gets suggestions instead of a guessed answer. Every question and structured answer is persisted (evidence IDs only, no verbatim feedback). |
| `services/decision_grounding.py` | Typed statements (FACT, ASSESSMENT, ASSUMPTION, LIMITATION, SIMULATION), run binding, evidence-reference validation, and historical-context notices when a newer run, a newer agent version or an edited requirement exists. |
| `services/agent_chat.py` | Discuss one agent, using its configuration as stored in the run (method, version, resources). Covers custom agents too, shows the human rationale exactly as recorded, and explains what is missing when an agent is UNSCORED. |
| `services/agent_debate.py` | Four rounds: positions from stored scores (agents may abstain with "Unable to conclude"), disagreements, drivers of the difference, and synthesis. The synthesis includes exact sensitivity: the weight shift to one agent that would tie the two requirements. No winner is declared when either requirement is incomplete. |
| `services/what_if_simulator.py` | Re-ranks the whole population of a run under alternative weights, dropped agents, added agents or another profile. It reuses the Part 5 scoring and ranking code. Results are SIMULATION only; "Save as Scenario" never touches profiles, runs or assessments. |
| `services/scenario_comparison.py` | Compares any two of run, scenario and simulation. It discloses input differences (scores and assessments, agent set and versions, requirement version, review status) and attributes changes to weights only when those inputs are held constant. |
| `services/priority_challenges.py` | Challenge workflow (OPEN, UNDER_REVIEW, RESOLVED or DISMISSED, reopen) with an evidence snapshot, a list of what needs review and an audit trail. Challenges never change scores; a reassessment goes through Part 5 and can be linked to the challenge. |
| `services/management_reports.py` | Excel (organised sheets, filters, frozen headers), CSV (long format with source references) and PDF (ReportLab). Reports include run and configuration versions, the prototype disclaimer, incomplete requirements listed unranked, flags, assessments, assumptions, history, and optional scenarios, challenges and debates. Verbatim feedback is excluded by default; secrets are never included; each export is logged. |
| `services/decision_llm.py` | Optional rephrasing only. It is disabled unless `PRODUCTMIND_ALLOW_EXTERNAL_LLM=true`, credentials are present, a cost estimate has been shown and the user confirms. There is no fallback model. Text containing new numbers or IDs is rejected. `register_local_model()` is the hook for a future on-premise model. |

**Tables** (additive): `decision_conversations`, `decision_messages`, `agent_debate_sessions`, `priority_challenges`, `priority_challenge_events`, `prioritization_scenarios`, `scenario_comparisons`, `management_report_exports`. `requirement_priority_scores` also gains a nullable `requirement_version` column, set for runs made from Part 6 onwards.

**Security note:** the prototype has **no authentication**. The user and reviewer IDs you type are labels, not identities. Run the dashboard on localhost (`streamlit run app.py --server.address localhost`).

## Agent-based prioritisation (Part 5)

> Agents are configurable evaluation components. They are not necessarily autonomous LLM agents.

**Flow:** Streamlit → agent configuration → resource registry → execution engine → deterministic scoring → weighted prioritisation → SQLite → explainable results. There is no LangChain, CrewAI or AutoGen, no microservice and no user-supplied SQL or Python.

| Module | Purpose |
|---|---|
| `services/agent_configuration.py` | Create, edit, duplicate, deactivate and remove agents. Each change writes a new configuration version, and removal is a soft delete. An agent's behaviour comes only from its scoring method and resources, never from its name. `seed_starter_agents` loads an optional illustrative starter set. |
| `services/agent_resources.py` | Registry of 8 approved resources over existing tables, with live record counts, fields, limitations and supported methods. |
| `services/agent_scoring.py` | Scoring methods and compatibility checks. Every result has a score from 0–100 or is UNSCORED, and records its inputs, source references, rationale, limitations and method version. |
| `services/agent_weights.py` | Integer basis points (10000 = 100%), Decimal parsing, validation, remaining weight, and a "distribute remaining" proposal that is never applied automatically. |
| `services/prioritization_profiles.py` | Create, edit (as new versions), duplicate, activate (only when weights total 100%) and archive profiles. Each profile version pins every agent's configuration version. |
| `services/dynamic_prioritization.py` | Human assessments and runs. Final score = Σ score × weight / 100. A requirement is COMPLETE only if every agent is SCORED and has provenance; otherwise it is INCOMPLETE with a NULL score and no weight redistribution. Ranking uses competition order (1, 2, 2, 4). |
| `services/agent_explanations.py` | Structured explanations with FACT, ASSESSMENT, ASSUMPTION and LIMITATION items, stored for Part 6. |
| `services/critical_issues.py` | Safety, regulatory and severe-reliability flags, raised and decided only by people. They are never inferred from VFC labels and never change a score. |
| `services/decision_context.py` | Read-only Part 6 interface: `get_decision_context`, `what_if_weights` (recomputes from stored scores and saves nothing) and `compare_requirements`. |

**Execution methods:**
- `DATA_ANALYSIS`: deterministic.
- `HUMAN_ASSESSMENT`: a score from 0–100 with rationale, assessor and timestamp. It is never auto-filled.
- `AI_ANALYSIS`: disabled. It always returns UNSCORED with "AI analysis unavailable. Use Data Analysis or Human Assessment."

**Scoring methods (v1.0):**
- `LOG_NORMALIZED_FEEDBACK_COUNT`: 100 × ln(1+n) / ln(1+N). Here n is the number of distinct G60 feedback IDs linked as direct evidence and N is all distinct G60 feedback IDs (4,000). Unknown, F70 or G70 IDs make the result UNSCORED.
- `HUMAN_SCORE`: the latest assessment for the requirement and agent.
- `EVIDENCE_COVERAGE`: 100 × covered evidence types / assigned evidence types. It is labelled as coverage, not impact.
- `VALIDATED_MARKET_RELEVANCE`: UNSCORED until market mappings are validated. G60 and G68 sales are combined, and country-to-region mappings are unvalidated.

**Reproducibility.** Every run stores the profile version, agent configuration versions, weights, method versions, each agent result (score, weighted contribution, inputs, references, explanation), the final scores, ranks, missing agents and the critical flags at run time. Editing or removing agents later does not change past runs.

**Page.** "Agent Prioritization" has six sections: A Agent Builder, B Weights, C Execution with human assessment forms, D Ranking (incomplete requirements are listed separately), E Profiles and F Explain my score (Plotly charts for contributions, scores and ranking). You can reach it from Vehicle Requirements, Customer Insights and the Evidence Explorer, and the requirement ID is preserved. A priority score never changes approval status.

**Limitations.**
- The log normalisation, equal-weight coverage and starter weights are prototype methodology.
- Demand reflects frequency in the supplied export (about 90% US), not severity or prevalence.
- Supporting evidence is linked per need category.
- Market relevance is always UNSCORED for now.
- The starter profile is illustrative, not an official BMW profile.
- Reviewer, assessor and user IDs are typed in by hand; there is no authentication.

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
