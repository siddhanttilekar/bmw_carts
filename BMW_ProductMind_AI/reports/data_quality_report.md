# BMW ProductMind AI - Data Quality Report (Part 1)

Generated: 2026-10-10T19:28:54

All figures below are computed from the supplied files and the SQLite database. No values were imputed, generated or merged.

## 1. Source documents

| file | dataset_type | original_file_names | origin | type | sheets_or_pages | source_of_truth | status | notes |
|---|---|---|---|---|---|---|---|---|
| 261010_BMW_Product_Management_Key_Note.pptx | briefing | ["261010_BMW_Product_Management_Key_Note.pptx"] | briefing | pptx | 4 | True | imported |  |
| BMW_Hackathon_Briefing_Short.pdf | briefing | ["BMW_Hackathon_Briefing_Short.pdf"] | briefing | pdf | 2 | True | no_text_layer | Pages without a text layer: [1, 2]. Image-only content; OCR not performed, no text invented. |
| BMW_G60_Complete_Prototype_Data.xlsx | consolidated_workbook | ["BMW_G60_Complete_Prototype_Data.xlsx"] | internal_bmw | xlsx | 9 | False | registered_only | Consolidated workbook. Overlapping sheets compared against originals (see dataset_overlaps); only its START HERE notes are stored. Originals are the single source of truth. |
| F70_G60_G68_G70_customer_studies.xlsx | customer_studies | ["F70_G60_G68_G70_customer_studies.xlsx"] | internal_bmw | xlsx | 2 | True | imported |  |
| F70_OptionList.PDF | option_list | ["F70_OptionList.PDF"] | internal_bmw | pdf | 23 | True | imported |  |
| F70_feedback_hackathon.xlsx | bmw_feedback | ["F70_feedback_hackathon.xlsx"] | internal_bmw | xlsx | 1 | True | imported |  |
| G60_OptionList.PDF | option_list | ["G60_OptionList.PDF"] | internal_bmw | pdf | 28 | True | imported |  |
| G60_feedback_hackathon.xlsx | bmw_feedback | ["G60_feedback_hackathon.xlsx"] | internal_bmw | xlsx | 1 | True | imported |  |
| G70_OptionList.PDF | option_list | ["G70_OptionList.PDF"] | internal_bmw | pdf | 27 | True | imported |  |
| G70_feedback_hackathon.xlsx | bmw_feedback | ["G70_feedback_hackathon.xlsx"] | internal_bmw | xlsx | 1 | True | imported |  |
| sales_volumes.xlsx | sales_volumes | ["sales_volumes.xlsx"] | internal_bmw | xlsx | 3 | True | imported |  |
| BMW_5_Series_Public_Owner_Feedback.xlsx | public_owner_feedback | ["BMW_5_Series_Public_Owner_Feedback.xlsx"] | public_external | xlsx | 2 | True | imported |  |
| BMW_G60_Public_Competitor_Evidence.xlsx | public_competitor_evidence | ["BMW_G60_Public_Competitor_Evidence.xlsx"] | public_external | xlsx | 2 | True | imported |  |

## 2. Database tables and record counts

| table | records |
|---|---|
| source_documents | 13 |
| source_document_pages | 84 |
| dataset_notes | 29 |
| feedback | 10128 |
| feedback_classifications | 13771 |
| public_feedback | 15 |
| competitor_evidence | 18 |
| customer_studies | 1833 |
| sales | 36 |
| equipment | 301 |
| equipment_availability | 1095 |
| dataset_overlaps | 9 |
| possible_duplicates | 13 |
| ingestion_audit | 22 |

## Integration checks

| check | passed | detail |
|---|---|---|
| 1. Selected datasets imported | PASS | all 5 imported |
| 2. Feedback counts match source files | PASS | F70_feedback_hackathon.xlsx feedback_classifications 4076/4076; F70_feedback_hackathon.xlsx feedback 2128/2128; G60_feedback_hackathon.xlsx feedback_classifications 5005/5005; G60_feedback_hackathon.xlsx feedback 4000/4000; G70_feedback_hackathon.xlsx feedback_classifications 4690/4690; G70_feedback_hackathon.xlsx feedback 4000/4000 |
| 3. No duplicate imports | PASS | duplicate key groups: {'feedback (file, feedback ID)': 0, 'feedback_classifications (file, row)': 0, 'public_feedback (public ID)': 0, 'competitor_evidence (row)': 0, 'customer_studies (sheet, row, column)': 0, 'sales (sheet, row, column)': 0}; rows imported from consolidated workbook: 0 |
| 4. Original customer feedback preserved verbatim | PASS | 0 feedback texts differ from source (compared every ID in all feedback files) |
| 5. Public and internal feedback distinguishable | PASS | feedback.data_origin=['internal_bmw']; public tables data_origin=['public_external']; separate tables (feedback vs public_feedback / competitor_evidence) |
| 6. Sales associated with correct models | PASS | sheet -> vehicles: F70=F70, G60_G68=G60+G68, G70=G70 |
| 7. G60/G68 combined sales not presented as G60-only | PASS | 12 G60_G68 rows, all is_combined_models=True, scope_label='G60 + G68 combined (not G60-only)'; G60-only sales rows: 0 |

## 3. Count validation (raw-file recount vs. database)

22 of 22 checks match.

| file | sheet | table | basis | expected | imported | status |
|---|---|---|---|---|---|---|
| F70_feedback_hackathon.xlsx | Feedback_Explorer | feedback_classifications | worksheet data rows | 4076 | 4076 | ok |
| F70_feedback_hackathon.xlsx | Feedback_Explorer | feedback | distinct values in ID column | 2128 | 2128 | ok |
| G60_feedback_hackathon.xlsx | Feedback_Explorer | feedback_classifications | worksheet data rows | 5005 | 5005 | ok |
| G60_feedback_hackathon.xlsx | Feedback_Explorer | feedback | distinct values in ID column | 4000 | 4000 | ok |
| G70_feedback_hackathon.xlsx | Feedback_Explorer | feedback_classifications | worksheet data rows | 4690 | 4690 | ok |
| G70_feedback_hackathon.xlsx | Feedback_Explorer | feedback | distinct values in ID column | 4000 | 4000 | ok |
| F70_G60_G68_G70_customer_studies.xlsx | CN_EU_2025 | customer_studies | numeric cells below header rows 1-3, columns B+ | 185 | 185 | ok |
| F70_G60_G68_G70_customer_studies.xlsx | US_2025 | customer_studies | numeric cells in columns C+ | 1648 | 1648 | ok |
| sales_volumes.xlsx | F70 | sales | data rows x volume_YYYY columns | 12 | 12 | ok |
| sales_volumes.xlsx | G60_G68 | sales | data rows x volume_YYYY columns | 12 | 12 | ok |
| sales_volumes.xlsx | G70 | sales | data rows x volume_YYYY columns | 12 | 12 | ok |
| F70_OptionList.PDF | None | source_document_pages | PDF page count | 23 | 23 | ok |
| G60_OptionList.PDF | None | source_document_pages | PDF page count | 28 | 28 | ok |
| G70_OptionList.PDF | None | source_document_pages | PDF page count | 27 | 27 | ok |
| BMW_Hackathon_Briefing_Short.pdf | None | source_document_pages | PDF page count | 2 | 2 | ok |
| 261010_BMW_Product_Management_Key_Note.pptx | None | source_document_pages | slide count | 4 | 4 | ok |
| BMW_5_Series_Public_Owner_Feedback.xlsx | Public G60 Feedback | public_feedback | worksheet data rows | 15 | 15 | ok |
| BMW_5_Series_Public_Owner_Feedback.xlsx | README | dataset_notes | non-empty rows | 7 | 7 | ok |
| BMW_G60_Public_Competitor_Evidence.xlsx | Public Evidence | competitor_evidence | worksheet data rows | 18 | 18 | ok |
| BMW_G60_Public_Competitor_Evidence.xlsx | Method and caveats | dataset_notes | non-empty rows | 12 | 12 | ok |
| BMW_G60_Complete_Prototype_Data.xlsx | START HERE | dataset_notes | non-empty rows | 10 | 10 | ok |
| BMW_G60_Complete_Prototype_Data.xlsx | None | dataset_overlaps | one comparison per sheet | 9 | 9 | ok |

Records missing mandatory provenance (record_id, source_file, sheet/page, imported_at):

| table | records_with_gaps |
|---|---|
| source_document_pages | 0 |
| dataset_notes | 0 |
| feedback | 0 |
| feedback_classifications | 0 |
| public_feedback | 0 |
| competitor_evidence | 0 |
| customer_studies | 0 |
| sales | 0 |
| equipment | 0 |
| equipment_availability | 0 |

## 4. Worksheet profiles

### F70_feedback_hackathon.xlsx / Feedback_Explorer

4076 rows, header row 1; exact duplicate rows: 0; key `ID`: 2128 distinct, 1009 keys on >1 row (max 11)

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| ID | string | 0 | 0.0 | 2128 |
| Source | string | 0 | 0.0 | 1 |
| Brand | string | 0 | 0.0 | 1 |
| Sales Description | string | 0 | 0.0 | 1 |
| Country | string | 0 | 0.0 | 9 |
| Derivate (e-code) | string | 0 | 0.0 | 1 |
| Engine Type | string | 0 | 0.0 | 1 |
| Feedback Type | string | 1087 | 26.67 | 4 |
| Customer Feedback | string | 0 | 0.0 | 2128 |
| Vfc level2 3 Name | string | 0 | 0.0 | 336 |
| Vfc level2 Name | string | 0 | 0.0 | 120 |
| Vfc level3 Name | string | 1087 | 26.67 | 57 |

### G60_feedback_hackathon.xlsx / Feedback_Explorer

5005 rows, header row 1; exact duplicate rows: 0; key `ID`: 4000 distinct, 365 keys on >1 row (max 18)

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| ID | string | 0 | 0.0 | 4000 |
| Source | string | 0 | 0.0 | 4 |
| Brand | string | 0 | 0.0 | 1 |
| Sales Description | string | 0 | 0.0 | 1 |
| Country | string | 0 | 0.0 | 13 |
| Derivate (e-code) | string | 0 | 0.0 | 1 |
| Engine Type | string | 0 | 0.0 | 3 |
| Feedback Type | string | 731 | 14.61 | 4 |
| Customer Feedback | string | 0 | 0.0 | 4000 |
| Vfc level2 3 Name | string | 200 | 4.0 | 953 |
| Vfc level2 Name | string | 200 | 4.0 | 221 |
| Vfc level3 Name | string | 725 | 14.49 | 82 |

### G70_feedback_hackathon.xlsx / Feedback_Explorer

4690 rows, header row 1; exact duplicate rows: 0; key `ID`: 4000 distinct, 262 keys on >1 row (max 17)

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| ID | string | 0 | 0.0 | 4000 |
| Source | string | 0 | 0.0 | 4 |
| Brand | string | 0 | 0.0 | 1 |
| Sales Description | string | 0 | 0.0 | 1 |
| Country | string | 0 | 0.0 | 13 |
| Derivate (e-code) | string | 0 | 0.0 | 1 |
| Engine Type | string | 0 | 0.0 | 3 |
| Feedback Type | string | 682 | 14.54 | 4 |
| Customer Feedback | string | 0 | 0.0 | 3993 |
| Vfc level2 3 Name | string | 200 | 4.26 | 854 |
| Vfc level2 Name | string | 200 | 4.26 | 211 |
| Vfc level3 Name | string | 670 | 14.29 | 77 |

### F70_G60_G68_G70_customer_studies.xlsx / CN_EU_2025

76 rows, no header row (cross-tab / notes layout); exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| 0 | string | 2 | 2.63 | 38 |
| 1 | mixed | 36 | 47.37 | 40 |
| 2 | mixed | 38 | 50.0 | 38 |
| 3 | mixed | 37 | 48.68 | 39 |
| 4 | mixed | 38 | 50.0 | 38 |
| 5 | mixed | 38 | 50.0 | 38 |

### F70_G60_G68_G70_customer_studies.xlsx / US_2025

929 rows, no header row (cross-tab / notes layout); exact duplicate rows: 134

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| 0 | string | 2 | 0.22 | 111 |
| 1 | string | 105 | 11.3 | 1 |
| 2 | mixed-integer | 104 | 11.19 | 286 |
| 3 | mixed-integer | 104 | 11.19 | 304 |

### sales_volumes.xlsx / F70

4 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| market | string | 0 | 0.0 | 4 |
| market_code | string | 0 | 0.0 | 4 |
| volume_2024 | integer | 0 | 0.0 | 3 |
| volume_2025 | integer | 0 | 0.0 | 3 |
| volume_2030 | integer | 0 | 0.0 | 3 |

### sales_volumes.xlsx / G60_G68

4 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| market | string | 0 | 0.0 | 4 |
| market_code | string | 0 | 0.0 | 4 |
| volume_2024 | integer | 0 | 0.0 | 4 |
| volume_2025 | integer | 0 | 0.0 | 4 |
| volume_2030 | integer | 0 | 0.0 | 4 |

### sales_volumes.xlsx / G70

4 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| market | string | 0 | 0.0 | 4 |
| market_code | string | 0 | 0.0 | 4 |
| volume_2024 | integer | 0 | 0.0 | 4 |
| volume_2025 | integer | 0 | 0.0 | 4 |
| volume_2030 | integer | 0 | 0.0 | 4 |

### BMW_5_Series_Public_Owner_Feedback.xlsx / Public G60 Feedback

15 rows, header row 1; exact duplicate rows: 0; key `ID`: 15 distinct, 0 keys on >1 row (max 1)

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| ID | string | 0 | 0.0 | 15 |
| Model | string | 0 | 0.0 | 5 |
| Model year | string | 0 | 0.0 | 2 |
| Review date | string | 0 | 0.0 | 9 |
| Source | string | 0 | 0.0 | 4 |
| Source type | string | 0 | 0.0 | 2 |
| Market | string | 0 | 0.0 | 2 |
| Sentiment | string | 0 | 0.0 | 2 |
| Theme | string | 0 | 0.0 | 15 |
| Paraphrased owner feedback | string | 0 | 0.0 | 15 |
| Product implication | string | 0 | 0.0 | 15 |
| Source URL | string | 0 | 0.0 | 5 |
| Evidence note | string | 0 | 0.0 | 12 |

### BMW_5_Series_Public_Owner_Feedback.xlsx / README

7 rows, no header row (cross-tab / notes layout); exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| 0 | string | 0 | 0.0 | 7 |
| 1 | string | 0 | 0.0 | 7 |

### BMW_G60_Public_Competitor_Evidence.xlsx / Public Evidence

18 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| Brand | string | 0 | 0.0 | 5 |
| Model | string | 0 | 0.0 | 6 |
| Source type | string | 0 | 0.0 | 6 |
| Date or period | string | 0 | 0.0 | 10 |
| Market | string | 0 | 0.0 | 3 |
| Theme | string | 0 | 0.0 | 17 |
| Sentiment | string | 0 | 0.0 | 3 |
| Extracted insight (paraphrased) | string | 0 | 0.0 | 18 |
| Evidence limitation | string | 0 | 0.0 | 12 |
| Source URL | string | 0 | 0.0 | 13 |

### BMW_G60_Public_Competitor_Evidence.xlsx / Method and caveats

11 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| Field | string | 0 | 0.0 | 11 |
| Explanation | string | 0 | 0.0 | 11 |

### BMW_G60_Complete_Prototype_Data.xlsx / START HERE

10 rows, no header row (cross-tab / notes layout); exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| 0 | string | 0 | 0.0 | 10 |
| 1 | string | 0 | 0.0 | 10 |

### BMW_G60_Complete_Prototype_Data.xlsx / G60 Feedback All

5005 rows, header row 1; exact duplicate rows: 0; key `ID`: 4000 distinct, 365 keys on >1 row (max 18)

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| ID | string | 0 | 0.0 | 4000 |
| Source | string | 0 | 0.0 | 4 |
| Brand | string | 0 | 0.0 | 1 |
| Sales Description | string | 0 | 0.0 | 1 |
| Country | string | 0 | 0.0 | 13 |
| Derivate (e-code) | string | 0 | 0.0 | 1 |
| Engine Type | string | 0 | 0.0 | 3 |
| Feedback Type | string | 731 | 14.61 | 4 |
| Customer Feedback | string | 0 | 0.0 | 4000 |
| Vfc level2 3 Name | string | 200 | 4.0 | 953 |
| Vfc level2 Name | string | 200 | 4.0 | 221 |
| Vfc level3 Name | string | 725 | 14.49 | 82 |

### BMW_G60_Complete_Prototype_Data.xlsx / Feedback Theme Counts

659 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| VFC Level 2 | string | 0 | 0.0 | 222 |
| Feedback type | string | 0 | 0.0 | 5 |
| Records | integer | 0 | 0.0 | 46 |

### BMW_G60_Complete_Prototype_Data.xlsx / Public Competitors

18 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| Brand | string | 0 | 0.0 | 5 |
| Model | string | 0 | 0.0 | 6 |
| Source type | string | 0 | 0.0 | 6 |
| Date or period | string | 0 | 0.0 | 10 |
| Market | string | 0 | 0.0 | 3 |
| Theme | string | 0 | 0.0 | 17 |
| Sentiment | string | 0 | 0.0 | 3 |
| Extracted insight (paraphrased) | string | 0 | 0.0 | 18 |
| Evidence limitation | string | 0 | 0.0 | 12 |
| Source URL | string | 0 | 0.0 | 13 |

### BMW_G60_Complete_Prototype_Data.xlsx / Public Caveats

11 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| Field | string | 0 | 0.0 | 11 |
| Explanation | string | 0 | 0.0 | 11 |

### BMW_G60_Complete_Prototype_Data.xlsx / Sales G60 G68

4 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| market | string | 0 | 0.0 | 4 |
| market_code | string | 0 | 0.0 | 4 |
| volume_2024 | integer | 0 | 0.0 | 4 |
| volume_2025 | integer | 0 | 0.0 | 4 |
| volume_2030 | integer | 0 | 0.0 | 4 |

### BMW_G60_Complete_Prototype_Data.xlsx / Study CN EU 2025

76 rows, no header row (cross-tab / notes layout); exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| 0 | string | 2 | 2.63 | 38 |
| 1 | mixed | 36 | 47.37 | 40 |
| 2 | mixed | 38 | 50.0 | 38 |
| 3 | mixed | 37 | 48.68 | 39 |
| 4 | mixed | 38 | 50.0 | 38 |
| 5 | mixed | 38 | 50.0 | 38 |

### BMW_G60_Complete_Prototype_Data.xlsx / Study US 2025

929 rows, no header row (cross-tab / notes layout); exact duplicate rows: 134

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| 0 | string | 2 | 0.22 | 111 |
| 1 | string | 105 | 11.3 | 1 |
| 2 | mixed-integer | 104 | 11.19 | 286 |
| 3 | mixed-integer | 104 | 11.19 | 304 |

### BMW_G60_Complete_Prototype_Data.xlsx / Evidence Link Template

1 rows, header row 1; exact duplicate rows: 0

| column | inferred_type | missing | missing_pct | distinct |
|---|---|---|---|---|
| Requirement ID | string | 0 | 0.0 | 1 |
| Vehicle | string | 0 | 0.0 | 1 |
| Market | string | 0 | 0.0 | 1 |
| Customer need / VFC | string | 0 | 0.0 | 1 |
| BMW feedback IDs | string | 0 | 0.0 | 1 |
| Public competitor source URLs | string | 0 | 0.0 | 1 |
| Survey reference | string | 0 | 0.0 | 1 |
| Existing feature reference | string | 0 | 0.0 | 1 |
| Proposed requirement | string | 0 | 0.0 | 1 |
| Assumptions | string | 0 | 0.0 | 1 |
| Human decision | string | 0 | 0.0 | 1 |

## 5. Internal feedback quality

| vehicle | classification_rows | feedback_ids | ids_with_multiple_classifications | feedback_type_missing | vfc_level2_missing | vfc_no_class_found | vfc_level3_missing | vfc_classified | vfc_level2_3_separator_variants | distinct_vfc_level2 |
|---|---|---|---|---|---|---|---|---|---|---|
| F70 | 4076 | 2128 | 1009 | 1087 | 0 | 1087 | 1087 | 2989 | 0 | 120 |
| G60 | 5005 | 4000 | 365 | 731 | 200 | 525 | 725 | 4280 | 2 | 221 |
| G70 | 4690 | 4000 | 262 | 682 | 200 | 470 | 670 | 4020 | 3 | 211 |

* One source row = one (feedback ID x VFC classification). IDs with several rows are multi-label, not duplicates; stored once in `feedback` and once per row in `feedback_classifications`.
* `vfc_level2_3_separator_variants`: rows where `Vfc level2 3 Name` is not exactly `<level2> : <level3>` (e.g. `...Intervention: Malfunction`). Original strings preserved.
* Missing `Feedback Type` / VFC values are kept as NULL; `no_class_found` is BMW's own label and is preserved.

## 6. Overlap between consolidated workbook and originals

| consolidated_sheet | original | rows_consolidated | rows_original | result | differing | decision |
|---|---|---|---|---|---|---|
| START HERE | - / - | 10 | None | metadata | None | Workbook description; stored in dataset_notes. |
| G60 Feedback All | G60_feedback_hackathon.xlsx / Feedback_Explorer | 5005 | 5005 | exact_copy | 0 | Exact copy of G60_feedback_hackathon.xlsx/Feedback_Explorer. Original is the source of truth; not re-imported. |
| Feedback Theme Counts | G60_feedback_hackathon.xlsx / Feedback_Explorer | 659 | 659 | derived_aggregate | 0 | Aggregate of G60 feedback rows by VFC level 2 x feedback type. Recomputed from original and matches exactly. Not imported; aggregates are recomputed from feedback_classifications. |
| Public Competitors | BMW_G60_Public_Competitor_Evidence.xlsx / Public Evidence | 18 | 18 | exact_copy | 0 | Exact copy of BMW_G60_Public_Competitor_Evidence.xlsx/Public Evidence. Original is the source of truth; not re-imported. |
| Public Caveats | BMW_G60_Public_Competitor_Evidence.xlsx / Method and caveats | 11 | 11 | exact_copy | 0 | Exact copy of BMW_G60_Public_Competitor_Evidence.xlsx/Method and caveats. Original is the source of truth; not re-imported. |
| Sales G60 G68 | sales_volumes.xlsx / G60_G68 | 4 | 4 | exact_copy | 0 | Exact copy of sales_volumes.xlsx/G60_G68. Original is the source of truth; not re-imported. |
| Study CN EU 2025 | F70_G60_G68_G70_customer_studies.xlsx / CN_EU_2025 | 76 | 76 | exact_copy | 0 | Exact copy of F70_G60_G68_G70_customer_studies.xlsx/CN_EU_2025. Original is the source of truth; not re-imported. |
| Study US 2025 | F70_G60_G68_G70_customer_studies.xlsx / US_2025 | 929 | 929 | exact_copy | 0 | Exact copy of F70_G60_G68_G70_customer_studies.xlsx/US_2025. Original is the source of truth; not re-imported. |
| Evidence Link Template | - / - | 1 | None | template | None | Template with an explicit example row ('Example only'). Not data; not imported. |

## 7. Possible duplicates (flagged for review, never merged)

| table | rule | groups | records_involved |
|---|---|---|---|
| competitor_evidence | same_source_url | 2 | 5 |
| competitor_evidence+public_feedback | same_source_url_across_public_datasets | 1 | 7 |
| feedback | identical_normalized_text_different_ids | 7 | 15 |
| public_feedback | same_source_url | 3 | 10 |

Public feedback also carries explicit relations from its `Evidence note` column (e.g. 'Same review as P001'), stored in `derived_related_public_id` / `derived_relation_type`.

## 8. Customer studies checks

US_2025: 206 attribute x vehicle blocks; blocks whose seven answer proportions do not sum to 1 (+/-0.02): 0

## 9. Equipment (option-list PDFs)

| vehicle | option_entries | distinct_option_codes | pages_with_codes | availability_cells | entries_without_name | entries_with_parse_note | variants |
|---|---|---|---|---|---|---|---|
| F70 | 89 | 68 | 10 | 248 | 0 | 14 | ['116', '118d', '120', '120d', '123 xDrive', 'M135 xDrive'] |
| G60 | 102 | 79 | 10 | 541 | 0 | 14 | ['520d', '520d xDrive', '520i', '530e', '530e xDrive', '540d xDrive', '550e xDrive', 'i5 M60 xDrive', 'i5 eDrive40', 'i5 xDrive40'] |
| G70 | 110 | 92 | 12 | 306 | 2 | 19 | ['740 xDrive', 'i7 50xDrive', 'i7 60xDrive', 'i7 M70 xDrive'] |

## 10. Text extraction (PDF / PPTX)

| file | pages | pages_without_text | pages_with_images | total_chars |
|---|---|---|---|---|
| F70_OptionList.PDF | 23 | [] | 6 | 31364 |
| G60_OptionList.PDF | 28 | [] | 6 | 43720 |
| G70_OptionList.PDF | 27 | [] | 6 | 38173 |
| BMW_Hackathon_Briefing_Short.pdf | 2 | [1, 2] | 2 | 0 |
| 261010_BMW_Product_Management_Key_Note.pptx | 4 | [] | 4 | 1914 |

## 11. Relationships between datasets

Vehicle coverage by dataset:

| vehicle | feedback | customer_studies | sales | equipment_option_list |
|---|---|---|---|---|
| F70 | True | True | True | True |
| G60 | True | True | True | True |
| G68 | False | True | True | False |
| G70 | True | True | True | True |

Vehicles per customer-study market:

| study / market | vehicles |
|---|---|
| CN_EU_2025 / China | BMW 5 Series G68, BMW 7 Series G70 |
| CN_EU_2025 / EU | BMW 1 Series F70, BMW 5 Series G60, BMW 7 Series G70 |
| US_2025 / US | BMW 5 Series G60, BMW 7 Series G70  |

VFC level-2 taxonomy: distinct classified level-2 names per vehicle {'F70': 119, 'G60': 220, 'G70': 210}; 118 shared by F70, G60 and G70; 15 appear only for G60.

Feedback country distribution (feedback IDs):

| vehicle | countries |
|---|---|
| F70 | GB=1293, JP=672, SE=71, FR=53, AU=16, DK=8, FI=8, KR=4, ZA=3 |
| G60 | US=3610, KR=205, JP=53, GB=36, SE=28, DK=20, CA=18, FI=9, FR=8, NO=8, AU=3, ES=1, LV=1 |
| G70 | US=3792, KR=137, JP=40, CA=8, GB=8, FR=3, AU=2, DK=2, FI=2, SE=2, ZA=2, ES=1, NO=1 |

Join keys identified (for later parts; nothing below is materialised as data):

| from | to | type | caveat |
|---|---|---|---|
| feedback.vehicle_code | customer_studies.vehicle_code / sales.vehicle_codes / equipment.vehicle_code | exact key | Sales for G60 are combined with G68 (sheet G60_G68); G68 has no feedback or option list. |
| feedback.country (ISO-2) | sales.market_code (EU/CN/US/RoW) | candidate mapping | Countries present: AU, CA, DK, ES, FI, FR, GB, JP, KR, LV, NO, SE, US, ZA. No CN feedback. Assigning GB/NO and non-EU countries to EU vs RoW is an assumption to confirm. |
| feedback.country | customer_studies.survey_market (China/EU/US) | candidate mapping | Survey markets are regions; same caveat as above. |
| feedback_classifications.vfc_level2_name | public_feedback.theme / competitor_evidence.theme / customer_studies.attribute | thematic only | Different vocabularies; no shared key. Requires curated mapping (Part 2). |
| equipment (German option names, SA codes) | VFC feature names (English) | thematic only | No shared code; e.g. 'Driving Assistant Professional' vs VFC level-2 names. |
| public_feedback.source_url | competitor_evidence.source_url | exact (URL) | Same public source cited in both external files; see possible_duplicates. |
| BMW_G60_Complete_Prototype_Data.xlsx sheets | original files | exact copy | See section 6; originals are the source of truth. |

## 12. Zero sales volumes (stored as 0, not missing)

F70 CN 2024, F70 CN 2025, F70 CN 2030, F70 US 2024, F70 US 2025, F70 US 2030

## 13. Files present but not registered

_none_

## 14. Failed or missing files

_none_
