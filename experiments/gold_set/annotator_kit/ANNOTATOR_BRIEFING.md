# Gold-set annotator briefing

## What you are doing

You judge **one job requirement against the whole CV**. You are not scoring a
model, and you will not be shown model labels, similarity scores, split names,
or dataset fit labels.

There are 160 items: 10 calibration items (`c001`…) and 150 main items
(`g001`…). The guideline time is about 90 seconds per item, so the expected
effort is about **4 hours** for each annotator.

## Labelling rubric

The rubric below is copied from `docs/gold_set_guidelines.md`. Use these labels
only.

Judge **one job requirement against the whole CV**. Do not try to guess a system
score. You will not be shown any model labels, similarity scores, split names, or
dataset fit labels.

Time guide: about 90 seconds per item.

### Labels

| Code | Name | Meaning |
| --- | --- | --- |
| 2 | Clearly met | Explicit evidence of the skill, experience, or qualification at about the required level. |
| 1 | Partially met | Related or adjacent experience, a lower level, a vague mention, or only part of a compound requirement. |
| 0 | Not evidenced | After reading the CV, you cannot find support. Absence of evidence, not a claim the person lacks the skill in real life. |
| C | cannot_judge | Requirement is too vague to interpret, or the CV text is unreadable/garbled. |

Optional: paste an `evidence_snippet` from the CV that supports 2 or 1. Optional `notes`.

### Examples (illustrative, not from the live pack)

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

### Compound requirements

If the requirement names two skills (“Python and Spark”), 2 only if both are evidenced
at the needed level; 1 if only one is; 0 if neither.

### What not to do

- Do not reward keyword stuffing with a 2 unless the CV shows actual use.
- Do not use the job description to *infer* CV content; the JD is only context for
  what the requirement means.
- Do not discuss items with the other annotator until the calibration round is done
  and, for the main set, until both have finished independently.

### Calibration then main set

Complete items marked CALIBRATION first (`c001`…). Discuss those ten. Then annotate
the main `g001`… items independently in the order given in the app for your id (A or B).

## How to use the app offline

1. Open `experiments/gold_set/annotator_app.html` in a browser (double-click, or
   File → Open). The pack is embedded in the file. Do not use a network connection
   and do not upload the CV text anywhere.
2. Set **Annotator** to A or B. That choice is your id. The app then shows items
   in your order. Calibration items are flagged.
3. For each item, read the requirement and the whole CV. Keys: `0` not evidenced,
   `1` partial, `2` clearly met, `C` cannot judge. You can also use the buttons.
4. Optional evidence snippet: paste text from the CV into the snippet box when
   your label is 2 or 1. Leave it empty for 0 or cannot_judge unless you want a
   note about why the item cannot be judged. Optional notes go in the notes box.
5. `cannot_judge` (key `C`) is for a requirement that is too vague to interpret,
   or a CV that is unreadable or garbled. It is not a substitute for “I did not
   find the skill”.
6. Click **Export CSV**. The browser saves `annotations.csv`.
7. Rename that file to exactly one of:
   - `annotator_A.csv`
   - `annotator_B.csv`
   matching the id you selected. Columns must stay
   `item_id,label,cannot_judge,snippet,notes`.

## Independence and assistance

Work independently. Do not discuss items with the other annotator until both of
you have finished the calibration ten, and do not discuss main-set items until
both of you have finished the main set. **Do not use AI assistance** (chatbots,
coding assistants, or automated labelling) on any item.

## How to return the file

Send only your renamed CSV back to the contact who recruited you. Do not send
the HTML pack, the CV text, or the other annotator’s file.
