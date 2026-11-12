# Rater briefing

You are rating interview questions for a thesis. You are not one of the
gold-set annotators, and you should not use that pack.

## Task

Each item shows one job and one candidate, then **Set X** and **Set Y**. Score
each set from 1 to 5 on relevance, coverage, and non-redundancy, using the
anchors in the study rubric. Then choose Set X, Set Y, or no preference.

The two sets are different methods. The page does not say which is which. Do
not try to guess.

## Effort

Time per pair has **not been measured**. A short pilot on a different set of
pairs will measure it before the main study. Until that number exists, plan on
a multi-hour task only after the study owner sends the measured estimate. Do
not treat any number in this file as a measured duration.

## How to work

There is no hosted web page. You receive a personal token and use the FastAPI
human-eval session and ratings endpoints (the study owner will send the exact
requests). There is no account and no password.

1. Load your session with the token you were sent (`GET /api/v1/human-eval/session`).
2. Read the job, candidate, Set X, and Set Y for the current pair.
3. Submit scores with `POST /api/v1/human-eval/ratings`. You can change a score
   by posting again.
4. Reload the same session later to resume at the first unfinished pair.
5. When every pair is complete, stop. Do not guess which set is which.

## Independence

Work alone. Do not discuss items with other raters. **Do not use AI assistance**
to choose scores.

## Problems

If the token fails, a score does not stick after reload, or you see a label other
than Set X and Set Y, stop and email the contact who sent the token.
