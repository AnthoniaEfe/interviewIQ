# Worked-rule brief (round 3 retraining)

Judge **one job requirement against the whole CV**. The JD only defines what the requirement means. It is not evidence.

## Labels

| Code | Name | Meaning |
| --- | --- | --- |
| 2 | Met | CV shows this requirement itself at about the needed level (role, project, or clear skill line that names it or an obvious synonym). |
| 1 | Partial | Related or adjacent experience only; lower level; vague mention; or only part of a compound requirement. |
| 0 | Not evidenced | After reading the CV, no support and no clear related experience. |
| C | cannot_judge | Requirement too vague to interpret (e.g. “positive attitude”, “fast-paced”), or CV unreadable. |

## Hard rules

1. **Met requires the requirement itself**, not a neighbour skill (React ≠ Vue; Lean/Six Sigma ≠ generic “improved efficiency” unless process-improvement work clearly matches).
2. **Related stays at partial.** Never upgrade adjacency to met.
3. **Compound requirements:** 2 only if all parts are evidenced; 1 if only some; 0 if none.
4. Prefer **cannot_judge** when you cannot tell what would count as evidence.
5. Do not use the JD to invent CV content.
6. Do not reward keyword stuffing without use.

## Calibration key (training only — discussed labels)

| Item | Requirement | Label | One-line lesson |
| --- | --- | ---: | --- |
| c001 | Amazon DynamoDB | 1 | Related DB/NoSQL-style work without DynamoDB → partial. |
| c002 | Docker | 1 | Related containers/dev work without Docker named → partial. |
| c003 | Financial Edge | 2 | CV shows the requirement (or clear equivalent named use). |
| c004 | Microsoft PowerPoint | 2 | Explicit or clear office-presentation evidence at met level. |
| c005 | Unit testing | 0 | No unit-testing evidence → not evidenced. |
| c006 | Contract compliance | 0 | No contract-compliance evidence → not evidenced. |
| c007 | OLAP | 1 | Related analytics/warehouse work without OLAP → partial. |
| c008 | Ruby | 0 | No Ruby → not evidenced. |
| c009 | Paperless environment proficiency | 2 | Clear paperless/digital-workflow evidence can be met. |
| c010 | ETL data pipeline design and development | 0 | No ETL pipeline evidence → not evidenced. |

Full discussion: `experiments/gold_set/round2/calibration_discussion.md`.
