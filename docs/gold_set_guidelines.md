# Gold-set annotation guidelines

Judge **one job requirement against the whole CV**. Do not try to guess a system
score. You will not be shown any model labels, similarity scores, split names, or
dataset fit labels.

Time guide: about 90 seconds per item.

## Labels

| Code | Name | Meaning |
| --- | --- | --- |
| 2 | Clearly met | Explicit evidence of the skill, experience, or qualification at about the required level. |
| 1 | Partially met | Related or adjacent experience, a lower level, a vague mention, or only part of a compound requirement. |
| 0 | Not evidenced | After reading the CV, you cannot find support. Absence of evidence, not a claim the person lacks the skill in real life. |
| C | cannot_judge | Requirement is too vague to interpret, or the CV text is unreadable/garbled. |

Optional: paste an `evidence_snippet` from the CV that supports 2 or 1. Optional `notes`.

## Examples (illustrative, not from the live pack)

**Requirement:** “Production Kubernetes administration”

- 2: CV bullet “Administered EKS clusters (upgrades, RBAC, cluster autoscaler) for 18 months.”
- 1: “Deployed a demo app to Kubernetes in a bootcamp project.”
- 0: Mentions Docker only, or no container platform at all.

**Requirement:** “5+ years of Java”

- 2: Dates/roles that add to ≥5 years of Java (or an explicit “6 years Java”).
- 1: “Java” listed with ~2–4 years, or Java plus similar JVM work that might count.
- 0: No Java (or only a one-line mention with no duration when the JD is explicit about years).

**Requirement:** “Bachelor’s degree in Computer Science or related field”

- 2: BSc/BA in CS, SE, EE, or a clearly related quantitative field.
- 1: Degree in an adjacent field, incomplete degree, or “equivalent experience” described.
- 0: No education section and no degree mentioned.

**Soft / leadership requirement:** “Excellent stakeholder communication”

- 2: Concrete evidence (led client workshops, wrote exec updates, managed vendors).
- 1: Generic “strong communicator” with some people-facing role.
- 0: Purely individual contributor technical list with no people/comms evidence.

**Vague requirement:** “Fast-paced environment”

- Prefer **cannot_judge** if you cannot tell what would count as evidence.
- If the CV clearly describes shipping under tight deadlines, 1 is acceptable; 2 is rare.

## Compound requirements

If the requirement names two skills (“Python and Spark”), 2 only if both are evidenced
at the needed level; 1 if only one is; 0 if neither.

## What not to do

- Do not reward keyword stuffing with a 2 unless the CV shows actual use.
- Do not use the job description to *infer* CV content; the JD is only context for
  what the requirement means.
- Do not discuss items with other annotators until the calibration round is done
  and, for the main set, until both people assigned to an item have finished
  independently.

## Calibration then main set

Complete items marked CALIBRATION first (`c001`…). All four annotators (A–D)
discuss those ten. Then annotate **only your assigned** main items independently
in the order given in the app for your id.
