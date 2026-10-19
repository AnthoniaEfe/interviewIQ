# Classifier reframing: computed results

This file records the full local `classifier_eval` run. The protocol in
`docs/classifier_reframing_report.md` is unchanged. Nothing here is a thesis claim.

- Experiment: `exp_20261019T123223Z`
- Git commit at run time: `bb76b89d96cdf3589c56727601c22961f6178db5`
- Stage: `classifier_eval`; smoke: `False`; status: `completed`
- Embedding: `all-MiniLM-L6-v2` on `cuda`
- LLM calls: none. Spend: $0.00.
- Feature caches reused for the original MiniLM pair features and for chunk features (`X3_*`) and gap-lite train (`Xg_train.npy`). Gap-lite test (`Xg_test.npy`) was computed in this run.

## Frozen decision rule

A variant is **improved over V0** only if it was selected on group-aware
cross-validation on the train split and the paired-bootstrap 95% CI
(2000 resamples, seed 42) of its test macro-F1 gain over V0 excludes 0.
Holm adjustment is reported on the McNemar accuracy p-values. The pipeline's
`improved_over_V0` flag follows the validation-plus-CI rule above.

## V0 regression and majority baseline

V0 was retrained on the cached MiniLM pair features. Regression against the
stored expected test metrics passed within tolerance 0.002.

| quantity | value |
| --- | --- |
| V0 accuracy | 0.463900 |
| V0 macro-F1 | 0.423951 |
| V0 weighted-F1 | 0.461521 |
| V0 regression passed | True |
| majority class id | 0 |
| majority accuracy | 0.487209 |
| majority balanced accuracy | 0.333333 |
| majority macro-F1 | 0.218400 |
| majority weighted-F1 | 0.319219 |
| majority QWK | 0.000000 |

## Test metrics

Three-class variants:

| variant | accuracy | balanced accuracy | macro-F1 | QWK |
| --- | --- | --- | --- | --- |
| V0 | 0.463900 | 0.423298 | 0.423951 | 0.187807 |
| V1 | 0.459920 | 0.417182 | 0.418097 | 0.153878 |
| V3 | 0.501421 | 0.465819 | 0.461546 | 0.291178 |
| V4_LR | 0.435475 | 0.399733 | 0.399977 | 0.105506 |
| V4_HGB | 0.480387 | 0.396514 | 0.388886 | 0.129269 |
| V5_LR | 0.494031 | 0.462726 | 0.459150 | 0.287916 |
| V5_HGB | 0.504264 | 0.400835 | 0.378301 | 0.145046 |
| V6 | 0.471859 | 0.443027 | 0.442482 | 0.280427 |

Binary No Fit vs fit-positive:

| variant | accuracy | balanced accuracy | macro-F1 | AUC |
| --- | --- | --- | --- | --- |
| V2 | 0.602615 | 0.605688 | 0.598219 | 0.669464 |
| V0_as_binary | 0.587834 | 0.588132 | 0.587829 | 0.644407 |
| V1_as_binary | 0.573053 | 0.573777 | 0.572973 | 0.626101 |

## Literal verdict

| variant | selected on validation | macro-F1 gain vs V0 | 95% CI | CI entirely above 0 | improved over V0 | McNemar p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | True | -0.005855 | [-0.018861, 0.007097] | False | False | 0.623194 | 1.000000 |
| V3 | True | 0.037594 | [0.006586, 0.067825] | True | True | 0.011977 | 0.071860 |
| V4_LR | True | -0.023975 | [-0.057615, 0.009453] | False | False | 0.083907 | 0.335628 |
| V4_HGB | True | -0.035066 | [-0.068055, -0.001589] | False | False | 0.304314 | 0.912941 |
| V5_LR | True | 0.035198 | [0.005178, 0.066026] | True | True | 0.045261 | 0.226303 |
| V5_HGB | True | -0.045650 | [-0.077320, -0.013369] | False | False | 0.004929 | 0.034506 |
| V6 | True | 0.018531 | [-0.013027, 0.049029] | False | False | 0.622710 | 1.000000 |

`improved over V0` is the pipeline flag: selected on validation and the gain CI lies entirely above 0. V4_HGB and V5_HGB have CIs entirely below 0, so they differ from V0 in the wrong direction and are not improvements. Holm p-values are on the McNemar accuracy tests, not a second gate on the macro-F1 CI.

V2 is a binary model. The three-class improved-over-V0 rule does not apply to it.

Under that rule, improved over V0: V3, V5_LR.
Not improved over V0: V1, V4_LR, V4_HGB, V5_HGB, V6.

Train group-CV macro-F1 stored by the run (this is the validation score the selector used):

| variant | group-CV macro-F1 |
| --- | --- |
| V1 | 0.911050 |
| V3 | 0.999036 |
| V4_LR | 0.427194 |
| V4_HGB | 0.573448 |
| V5_LR | 0.999036 |
| V5_HGB | 0.851146 |
| V6 | 0.999036 |

These validation scores are much higher than the corresponding test macro-F1 values for V3, V5_LR, and V6. This file reports them as computed. It does not treat the classifier as a thesis result.

## Figures

- `docs/figures/fig30_classifier_headline.png`
- `docs/figures/fig31_per_class_f1.png`
- `docs/figures/fig32_confusion.png`
- `docs/figures/fig33_binary_roc.png`
- `docs/figures/fig34_gap_lite_importance.png`

Tables: `experiments/results/classifier_reframing_combined/`.

