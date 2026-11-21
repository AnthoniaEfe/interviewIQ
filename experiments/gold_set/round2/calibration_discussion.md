# Round 2 calibration discussion — 10 shared items

This sheet is the calibration meeting for the second round. It is not a new gold label and it does not replace round 1.

Round 1 used the five files in `annotator_key/` with no adjudication. Those labels stay as they are.

## Worked rule

A requirement is **met (2)** only when the CV shows that requirement: the skill, qualification, or experience itself, at about the level the requirement names.

Related or adjacent experience counts as **partial (1)** at most. It does not count as met.

If the CV does not show the requirement and does not show related experience, the label is **not evidenced (0)**.

The job description explains what the requirement means. It is not evidence that the CV contains it.

Worked example, not from this pack: the requirement is React.

- 2: the CV describes React use (a role, a project, or a skill line that names React).
- 1: the CV shows a related interface skill, such as another JavaScript UI library, and does not show React.
- 0: the CV shows backend Java only, with no interface work that bears on React.

The discussion outcome for these ten items is recorded in `experiments/gold_set/annotations/adjudicated_calibration.csv`. The independent round-1 labels below are unchanged. The 30 main disagreements are still open in `adjudication_worksheet.csv`.

## c001 — split

Requirement: Amazon DynamoDB

Literal requirement phrase in the CV: false.

Adjudicated label: 1.

Round-1 labels: A=0, B=0, C=1, D=1, E=1.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c002 — split

Requirement: Docker

Literal requirement phrase in the CV: false.

Adjudicated label: 1.

Round-1 labels: A=1, B=0, C=0, D=0, E=0.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

Notes from round 1:

- annotator_D: The only met criteria is the programming in C++ and HTML, this role require extensive software engineering experience, a staff level requirement and the CV is by my analysis junior to mid level

## c003 — split

Requirement: Financial Edge

Literal requirement phrase in the CV: false.

Adjudicated label: 2.

Round-1 labels: A=2, B=2, C=2, D=1, E=1.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

Notes from round 1:

- annotator_D: i believe the applicant CV should include their years of relevant experience in profile summary

## c004 — split

Requirement: Microsoft PowerPoint

Literal requirement phrase in the CV: false.

Adjudicated label: 2.

Round-1 labels: A=0, B=1, C=0, D=2, E=0.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c005 — split

Requirement: Unit testing

Literal requirement phrase in the CV: false.

Adjudicated label: 0.

Round-1 labels: A=1, B=0, C=1, D=1, E=2.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c006 — split

Requirement: Contract compliance

Literal requirement phrase in the CV: false.

Adjudicated label: 0.

Round-1 labels: A=0, B=1, C=1, D=0, E=0.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c007 — split

Requirement: OLAP

Literal requirement phrase in the CV: false.

Adjudicated label: 1.

Round-1 labels: A=1, B=2, C=2, D=2, E=0.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c008 — split

Requirement: Ruby

Literal requirement phrase in the CV: false.

Adjudicated label: 0.

Round-1 labels: A=1, B=1, C=0, D=0, E=1.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c009 — split

Requirement: Paperless environment proficiency

Literal requirement phrase in the CV: false.

Adjudicated label: 2.

Round-1 labels: A=C, B=0, C=2, D=2, E=2.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.

## c010 — unanimous

Requirement: ETL data pipeline design and development

Literal requirement phrase in the CV: false.

Adjudicated label: 0.

Round-1 labels: A=C, B=0, C=0, D=0, E=0.

The exact requirement string is not a substring of the CV. That check is a prompt, not the label. Under the worked rule, 2 still requires the CV to show this requirement, including an obvious name for it. Related experience stays at 1. If the CV shows neither, the label is 0.

No round-1 notes on this item.
