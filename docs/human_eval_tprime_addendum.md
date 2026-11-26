# Human-eval addendum: treatment arm is T′

The committed protocol in `docs/human_eval_protocol.md` names conditions **T** and **B1**. That file is not rewritten.

After `docs/selector_tprime_adoption.md`, the treatment arm **T** in any new blinded study is the **T′** selector (dimension quotas + reserved non-gap slots + weighted MMR remainder), not historical `competency_weighted_mmr`.

## Rubric pilot (ready to rate)

Five primary-30 pairs sampled with seed `20261126`:

```bash
PYTHONPATH=backend python experiments/scripts/prepare_human_eval_pilot.py
```

Inputs: `experiments/human_eval/inputs/pilot5_Tprime_vs_B1.json`.  
Manifest: `experiments/human_eval/manifest_pilot.json`.  
Rater UI: `experiments/human_eval/rater_app.html`.  
Launch checklist: `docs/human_eval_launch.md`.

Blinded keys remain `T` / `B1`; here **T means T′**.

## Full primary-30 exploratory inputs (after rubric freeze)

```bash
PYTHONPATH=backend python experiments/human_eval/build_study.py \
  --inputs experiments/human_eval/inputs/primary30_Tprime_vs_B1.json \
  --raters R1 R2 R3 \
  --mode pilot
```

Label any primary-30 human contrast **exploratory**. Confirmatory main study stays holdout-30.

## Holdout main study

When holdout T pools are generated, use `--selection tprime`. Do not build a main study from historical weighted-MMR treatment files.
