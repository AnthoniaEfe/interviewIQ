# Judge validation pre-registration (proposed, not approved)

**approval: pending**

Do not run the judge on holdout-30, and do not load human ratings into the
judge, until this line is changed by the user to `approval: approved`. The
runner refuses while the status is pending.

## Prompt

`prompts/evaluation/judge_v1.md`. Hash it at run time with
`app.services.human_eval.judge.prompt_sha256` and store that hash on the output.
The prompt is frozen as of the commit that adds this file. Later edits require
a new prompt file, not a silent change.

## Model

`LLMTask.SIMPLE_TRANSFORMATION`, which the existing router sends to Luna
(`gpt-5.6-luna`), not Terra. The judge sees only the blinded job summary,
candidate summary, and the two question sets. Both presentation orders (X then
Y, and Y then X) are run. Position bias is a preference that follows whichever
set is shown first.

## Proposed agreement thresholds (for the user to approve or replace)

These were not fit on holdout ratings. No human ratings exist yet.

| Check | Proposed bar |
| --- | --- |
| Weighted kappa, judge vs human mean, each criterion | ≥ 0.40 |
| Spearman, judge vs human mean, each criterion | ≥ 0.40 |
| Preference agreement, ties excluded | ≥ 0.60 |
| Position-bias rate | ≤ 0.20 of pairs |
| Sign of the T−B1 effect | Same sign as the human effect on at least 2 of 3 criteria |

The judge does not replace the human confirmatory claim. Scoring all 150 pairs
with the judge is allowed only after every bar above holds, and that extension
is secondary.

## Spend

Same gates as the 150-pair run: SCALE verdict, freeze hashes, and a dry-run
under the cap. Those gates currently fail, so the judge is not run.
