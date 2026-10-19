# Human evaluation protocol

Committed before any rating is collected. Pilot ratings, if they are later
collected, stay in the `pilot` store and are excluded from the main analysis.

## Items

Holdout-30 pairs, two conditions each: **T** (treatment) and **B1**
(profile-aware baseline). That is 60 sets, shown as Set X and Set Y for the
same pair. Automatic metrics, when the 150-pair run exists, cover all 150
pairs. This human study does not.

The rubric pilot uses about five **primary-30** pairs, not holdout-30.

## Raters

At least three raters, ideally at least one with HR or interviewing experience.
Raters are separate from the gold-set annotators. The store keeps a rater id
only. Optional background is yes/no HR or interviewing experience and years.
No names are stored.

## Rubric

Three ordinal 1–5 ratings per set: relevance, coverage, non-redundancy, plus
one preference per pair: Set X, Set Y, or no preference. Anchors are in
`docs/human_eval_rubric.md`. That file is a **draft** until the pilot export
exists and the wording is revised at most once. No other criteria are added.

## Presentation

Side of T versus B1 is balanced (15 pairs each way). Item order is randomised
per rater. Question order inside a set is shuffled once when the blinded
payload is built. The condition key is not in the API, the client bundle, or
`frontend/public/`.

## Analysis

Main analysis reads the `main` export and the condition key only.

1. Refuse demo and pilot stores.
2. Refuse if any rater is below 90% completion.
3. Report Krippendorff's alpha (ordinal) per criterion before any condition test.
4. Average raters within a pair. Wilcoxon signed-rank on T minus B1 for each
   criterion. Holm across the three criteria.
5. Preference: exact binomial sign test, ties excluded.
6. Leave-one-rater-out is reported as a sensitivity when at least three raters
   have completed the export. The current script reports the primary tests; the
   leave-one-rater-out rerun uses the same function on the reduced export.

Development choices are not changed after holdout ratings are opened. The
rating window uses holdout items only, so there is no second holdout.

## Workload

Time per pair is **not estimated**. It will be the median `elapsed_ms` from a
real pilot export. Until that export exists, the platform shows that the time
estimate is pending.
