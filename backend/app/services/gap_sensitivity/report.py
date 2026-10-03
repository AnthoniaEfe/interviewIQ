"""Write docs/gap_sensitivity_report.md from computed result summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _pct(x: Any) -> str:
    if x is None:
        return "null"
    return f"{100 * float(x):.1f}%"


def write_report(
    *,
    report_path: Path,
    primary_dir: Path,
    holdout_dir: Path,
    combined_dir: Path,
    figures_dir: Path,
) -> Path:
    primary = _load(primary_dir / "summary.json")
    holdout = _load(holdout_dir / "summary.json")
    conditions = primary["conditions"]

    lines: list[str] = []
    lines += [
        "# Gap-matrix evidence-set sensitivity analysis",
        "",
        "Local, zero-LLM-cost analysis of how candidate evidence-set construction "
        "affects profile-gap labels. Job requirements, score weights "
        "(0.25 TF-IDF / 0.25 BM25_norm / 0.50 semantic), thresholds "
        "(0.65 / 0.40 / 0.20), and the embedding model are held fixed.",
        "",
        f"- Primary experiment: `{primary['experiment_id']}` "
        f"(source extraction `{primary['source_experiment']}`, "
        f"stored gap `{primary['stored_gap_experiment']}`)",
        f"- Holdout experiment: `{holdout['experiment_id']}` "
        f"(source extraction `{holdout['source_experiment']}`, "
        f"stored gap `{holdout['stored_gap_experiment']}`)",
        f"- Combined tables: `experiments/results/gap_sensitivity_combined/`",
        f"- Git commit: `{primary.get('git_commit')}`",
        f"- Embedding: `{primary.get('embedding_model')}` on `{primary.get('embedding_device')}`",
        f"- Library versions: `{primary.get('library_versions')}`",
        "",
        "## Code inspection",
        "",
        "### Gap-analysis service and entry points",
        "",
        "- Core implementation: `backend/app/services/gap_analysis/matrix.py` "
        "(`build_gap_matrix`, `classify_match`, `_candidate_texts`, `_job_competencies`).",
        "- API entry: `POST` pipeline path in `backend/app/api/v1/pipeline.py` calls "
        "`build_gap_matrix`.",
        "- Experiment entry: `experiments/run.py` stage `gap_analysis` "
        "(`run_gap_combo` → `build_gap_matrix`).",
        "- New stage: `gap_sensitivity` (this work) in `experiments/run.py`, "
        "implemented under `backend/app/services/gap_sensitivity/`.",
        "",
        "### How candidate texts are assembled (production)",
        "",
        "Production `_candidate_texts` (current code, and the code that produced "
        "the stored Primary/Holdout gap runs) builds a **deduplicated union** of:",
        "",
        "- `technical_skills`, `soft_skills`, `domain_experience`, `certifications`, "
        "`competencies` (`normalized_label`)",
        "- per experience entry: `technologies` and `responsibilities`",
        "",
        "It does **not** currently include experience `achievements`, `title`, "
        "education entries, projects, or `professional_summary`.",
        "",
        "**Important discrepancy vs the task brief:** the brief described Condition A "
        "as “extracted competency strings only”. That matches an *older* "
        "`_candidate_texts` that returned competencies when non-empty and only then "
        "fell back to skills/experience. The stored gap experiments "
        "(`exp_20260927T215232Z`, `exp_20260927T223303Z`) were produced with the "
        "**expanded** union above. Condition `A_competencies` in this analysis is "
        "therefore defined as **production `_candidate_texts`** so the regression "
        "check can reproduce stored counts. Condition `A0_competencies_only` is the "
        "sparse competencies-only counterfactual implied by the brief.",
        "",
        "### BM25 normalisation",
        "",
        "Raw BM25 is computed with `BM25Okapi` over the candidate evidence strings "
        "for that pair. `bm25_norm = bm25_raw / bm25_max`, where `bm25_max` is the "
        "**maximum raw BM25 across requirements for the same pair and evidence set**. "
        "Changing the evidence set therefore changes both raw scores and the "
        "normalisation constant — labels are not strictly comparable across "
        "conditions on the BM25 component alone.",
        "",
        "**Mitigations used here:**",
        "",
        "1. Primary reported labels use production normalisation (needed for "
        "regression fidelity on Condition A).",
        "2. Each requirement also stores `bm25_norm_fixed` / `combined_fixed_bm25` "
        "using Condition A’s per-pair `bm25_max`.",
        "3. Semantic-only metrics (same thresholds on the semantic score alone) are "
        "reported as the scale-free robustness check.",
        "",
        "### TF-IDF fitting",
        "",
        "`TfidfMatcher.similarity` fits a two-document TF-IDF vectorizer on "
        "`[job_requirement_label, candidate_blob]` where `candidate_blob` is the "
        "space-joined evidence list. The IDF/TF scale therefore also depends on the "
        "evidence blob. Semantic-only metrics again remove this confound.",
        "",
        "### `CandidateProfile` fields relevant to evidence",
        "",
        "Present in schema (`backend/app/schemas/domain.py`): "
        "`competencies`, `technical_skills`, `soft_skills`, `domain_experience`, "
        "`certifications`, `experience` (`title`, `responsibilities`, `achievements`, "
        "`technologies`), `education`, `projects`, `professional_summary`, "
        "`seniority_indicators`. Condition C uses title / responsibilities / "
        "achievements (schema has these fields — condition is not skipped).",
        "",
        "### Mapping extraction JSON → pair IDs → raw CV",
        "",
        "Extraction artifacts live in `experiments/results/<exp>/extraction/"
        "{profile_id}__{jd_id}__{seniority}.json` with `pair_id`, `profile_id`, "
        "`fit_label`, `candidate`, `job`. Raw CV text is loaded from the config "
        "`data.pairs_dir` as `profiles/{profile_id}.txt` "
        "(Primary: `experiment_pairs/`; Holdout: `experiment_pairs_150/` via "
        "`primary_hf_holdout30.yaml`).",
        "",
        "### Ablation flags",
        "",
        "`Settings` exposes `enable_tfidf`, `enable_bm25`, `enable_semantic`, "
        "`enable_gap_analysis`, `enable_competency_weighting`, `enable_mmr`, "
        "`enable_redundancy_filter`. Defaults remain unchanged; this stage does not "
        "toggle them.",
        "",
        "### Runner `--stage` and artifact format",
        "",
        "`experiments/run.py` accepts `--stage {extraction,gap_analysis,baseline,"
        "gap_sensitivity}`. Gap-sensitivity writes a new `experiments/results/exp_<UTC>/` "
        "with `summary.json`, `regression.json`, `pairs/*.json`, and `tables/*.csv`. "
        "Cross-split outputs go to `experiments/results/gap_sensitivity_combined/`.",
        "",
        "## Method",
        "",
        "### Conditions",
        "",
    ]
    for cond in conditions:
        desc = primary.get("condition_descriptions", {}).get(cond, "")
        lines.append(f"- **{cond}**: {desc}")
    lines += [
        "",
        "### Mismatched-candidate control",
        "",
        "For each pair and condition, job requirements are scored against another "
        "candidate’s evidence set from the same split (never self; 5 permutations; "
        "seed 42). Size-adjusted effect = true gap rate − mismatched gap rate.",
        "",
        "### Statistics",
        "",
        "Bootstrap 95% CIs use 2000 resamples over pairs (seed 42). Paired "
        "comparisons use Wilcoxon signed-rank tests with matched-pairs rank-biserial "
        "effect sizes. Dose-response uses Spearman correlation between per-pair "
        "evidence size and per-pair gap rate.",
        "",
        "## Regression-check result",
        "",
    ]
    for name, summary in [("Primary-30", primary), ("Holdout-30", holdout)]:
        reg = summary["regression"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Passed score/label match: **{reg['passed']}**")
        lines.append(f"- Counts match expected: **{reg['counts_match_expected']}**")
        lines.append(f"- Counts: `{reg['counts']}`")
        lines.append(f"- Expected: `{reg['expected_counts']}`")
        lines.append(
            f"- Label mismatches: {reg['n_label_mismatch']}; "
            f"score mismatches (tol={reg['tolerance']}): {reg['n_score_mismatch']}; "
            f"max |Δ combined|: {_fmt(reg['max_abs_combined_diff'], 8)}"
        )
        lines.append("")

    if not (
        primary["regression"]["passed"]
        and holdout["regression"]["passed"]
        and primary["regression"]["counts_match_expected"]
        and holdout["regression"]["counts_match_expected"]
    ):
        lines += [
            "**Regression did not fully pass; later tables should be treated with caution.**",
            "",
        ]

    lines += [
        "## Results tables",
        "",
        "### Requirement counts and gap shares",
        "",
        "| Split | Condition | N | Strong | Partial | Weak | Missing | Gap share (95% CI) | Mean per-pair gap (95% CI) | Evid. size mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for summary in (primary, holdout):
        for cond in conditions:
            m = summary["metrics"][cond]
            lc = m["label_counts"]
            gci = m["pooled_gap_share_bootstrap"]
            pci = m["per_pair_gap_rate_bootstrap"]
            lines.append(
                f"| {summary['split']} | {cond} | {sum(lc.values())} | "
                f"{lc['strong_match']} | {lc['partial_match']} | {lc['weak_match']} | "
                f"{lc['missing']} | {_pct(m['pooled_gap_share'])} "
                f"[{_pct(gci['ci_low'])}, {_pct(gci['ci_high'])}] | "
                f"{_fmt(pci['mean'])} [{_fmt(pci['ci_low'])}, {_fmt(pci['ci_high'])}] | "
                f"{_fmt(m['evidence_size']['mean'], 2)} |"
            )

    lines += [
        "",
        "### Size-adjusted effects (true − mismatched pooled gap share)",
        "",
        "| Split | Condition | True gap share | Mismatched gap share | Size-adjusted Δ |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for summary in (primary, holdout):
        for cond in conditions:
            sa = summary["size_adjusted"][cond]
            lines.append(
                f"| {summary['split']} | {cond} | {_pct(sa['true_pooled_gap_share'])} | "
                f"{_pct(sa['mismatched_pooled_gap_share'])} | "
                f"{_fmt(sa['size_adjusted_effect_pooled_share'])} |"
            )

    lines += [
        "",
        "### Wilcoxon vs Condition A (per-pair gap rate)",
        "",
        "| Split | Condition | n | n≠0 | p | rank-biserial |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in (primary, holdout):
        for cond, w in summary["wilcoxon_vs_A"].items():
            lines.append(
                f"| {summary['split']} | {cond} | {w['n_pairs']} | {w['n_nonzero']} | "
                f"{_fmt(w['p_value'], 4)} | {_fmt(w['rank_biserial'])} |"
            )

    lines += [
        "",
        "### Spearman dose-response (evidence size vs per-pair gap rate)",
        "",
        "| Split | Condition | ρ | p | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for summary in (primary, holdout):
        for cond in conditions:
            d = summary["metrics"][cond]["dose_response"]
            lines.append(
                f"| {summary['split']} | {cond} | {_fmt(d['rho'])} | "
                f"{_fmt(d['p_value'], 4)} | {d['n']} |"
            )

    lines += [
        "",
        "### Semantic-only robustness (pooled gap share)",
        "",
        "| Split | Condition | Gap share (semantic-only) | Mean per-pair gap |",
        "| --- | --- | ---: | ---: |",
    ]
    for summary in (primary, holdout):
        for cond in conditions:
            m = summary["semantic_only_metrics"][cond]
            lines.append(
                f"| {summary['split']} | {cond} | {_pct(m['pooled_gap_share'])} | "
                f"{_fmt(m['per_pair_gap_rate']['mean'])} |"
            )

    lines += [
        "",
        "### Mean signal values",
        "",
        "| Split | Condition | TF-IDF | BM25_norm | Semantic | Combined |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in (primary, holdout):
        for cond in conditions:
            s = summary["metrics"][cond]["mean_signals"] or {}
            lines.append(
                f"| {summary['split']} | {cond} | {_fmt(s.get('tfidf'))} | "
                f"{_fmt(s.get('bm25_norm'))} | {_fmt(s.get('semantic'))} | "
                f"{_fmt(s.get('combined'))} |"
            )

    # Breakdowns
    lines += ["", "### Breakdowns (A and best-evidence condition)", ""]
    for summary in (primary, holdout):
        bd = summary["breakdowns"]
        best = bd["best_evidence_condition"]
        lines.append(f"#### {summary['split']}")
        lines.append("")
        for cond in (conditions[0], best):
            if cond not in bd:
                continue
            block = bd[cond]
            lines.append(f"**{cond}**")
            lines.append("")
            lines.append("| Slice | N | Gap share |")
            lines.append("| --- | ---: | ---: |")
            for dim in ("by_fit_label", "by_priority"):
                for key, val in block[dim].items():
                    lines.append(
                        f"| {dim}:{key} | {val['n_requirements']} | "
                        f"{_pct(val['pooled_gap_share'])} |"
                    )
            lines.append("")

    # Transitions A->D
    lines += [
        "### Label transitions A → D (counts)",
        "",
    ]
    for summary in (primary, holdout):
        key = "A_competencies->D_raw_cv_chunks"
        mat = summary["transitions"].get(key, {})
        lines.append(f"#### {summary['split']}")
        lines.append("")
        labs = ["strong_match", "partial_match", "weak_match", "missing"]
        lines.append("| from \\ to | " + " | ".join(labs) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * 4) + " |")
        for a in labs:
            row = mat.get(a, {})
            lines.append(
                "| " + a + " | " + " | ".join(str(row.get(b, 0)) for b in labs) + " |"
            )
        lines.append("")

    lines += [
        "## Figures",
        "",
        "![Figure 17](figures/fig17_gap_category_shares.png)",
        "",
        "![Figure 18](figures/fig18_per_pair_gap_rate.png)",
        "",
        "![Figure 19](figures/fig19_true_vs_mismatched.png)",
        "",
        "![Figure 20](figures/fig20_size_vs_gap_rate.png)",
        "",
        "![Figure 21](figures/fig21_A_to_D_transition.png)",
        "",
        "## Audit artifacts",
        "",
        "- `experiments/results/gap_sensitivity_combined/audit_sheet.csv`",
        "- `experiments/results/gap_sensitivity_combined/audit_sheet.md`",
        "- `experiments/results/gap_sensitivity_combined/false_gap_examples.md`",
        "",
        "## Limitations",
        "",
        "- BM25 and TF-IDF scales depend on the evidence set; cross-condition "
        "combined-score comparisons are partly confounded (mitigated via semantic-only "
        "and fixed-bm25 diagnostics, not eliminated).",
        "- Max-similarity over a larger evidence set mechanically raises scores; the "
        "mismatched-candidate control is required before interpreting gap-rate drops.",
        "- Raw CV chunking is heuristic (sentence/bullet splitter); different chunkers "
        "could change Condition D.",
        "- No human adjudication yet; audit columns for recall estimates are blank.",
        "- Condition A is production evidence, not competencies-only; A0 addresses the "
        "sparse-competency counterfactual separately.",
        "",
        "## What this means for the thesis",
        "",
    ]

    # Interpretation from numbers only
    def _interpret(summary: dict[str, Any]) -> str:
        split = summary["split"]
        m = summary["metrics"]
        sa = summary["size_adjusted"]
        a_gap = m["A_competencies"]["pooled_gap_share"]
        d_gap = m["D_raw_cv_chunks"]["pooled_gap_share"]
        a_adj = sa["A_competencies"]["size_adjusted_effect_pooled_share"]
        d_adj = sa["D_raw_cv_chunks"]["size_adjusted_effect_pooled_share"]
        a0 = m.get("A0_competencies_only") or {}
        a0_gap = a0.get("pooled_gap_share")
        a0_adj = (sa.get("A0_competencies_only") or {}).get(
            "size_adjusted_effect_pooled_share"
        )
        parts = [
            f"On **{split}**, production Condition A pooled gap share is {_pct(a_gap)} "
            f"(size-adjusted effect {_fmt(a_adj)}); "
            f"Condition D (raw CV chunks) is {_pct(d_gap)} "
            f"(size-adjusted {_fmt(d_adj)})."
        ]
        if a0_gap is not None:
            drop_a0_a = a0_gap - (a_gap or 0)
            mm_a0 = (sa.get("A0_competencies_only") or {}).get(
                "mismatched_pooled_gap_share"
            )
            mm_a = sa["A_competencies"]["mismatched_pooled_gap_share"]
            parts.append(
                f"Competencies-only (A0) gap share is {_pct(a0_gap)} "
                f"(size-adjusted {_fmt(a0_adj)}); absolute A0→A reduction is "
                f"{_fmt(drop_a0_a)}."
            )
            if mm_a0 is not None and mm_a is not None and a_gap is not None:
                control_drop = mm_a0 - mm_a
                parts.append(
                    f"Mismatched-control A0→A gap-share change is {_fmt(control_drop)}."
                )
                if drop_a0_a > control_drop + 0.05:
                    parts.append(
                        "Interpretation vs A0 (supported): moving from competencies-only "
                        "to production evidence reduces the gap rate by more than the "
                        "mismatched-control change, so a large share of the "
                        "competencies-only gap rate is an extraction-coverage artefact "
                        "(missing skills/experience strings), not only size bias."
                    )
                elif abs(drop_a0_a - control_drop) <= 0.05:
                    parts.append(
                        "Interpretation vs A0 (supported): the A0→A drop is similar to "
                        "the mismatched-control drop, so the effect is mostly size bias."
                    )
                else:
                    parts.append(
                        "Interpretation vs A0 (supported): mixed — production evidence "
                        "lowers gaps vs competencies-only, but size bias also contributes."
                    )
        if a_gap is not None and d_gap is not None:
            delta_ad = a_gap - d_gap
            if delta_ad < 0:
                parts.append(
                    f"Interpretation vs D (supported): raw-CV chunks do **not** reduce "
                    f"gaps relative to production A (A−D = {_fmt(delta_ad)}); with mean "
                    f"evidence size "
                    f"{_fmt(m['D_raw_cv_chunks']['evidence_size']['mean'], 2)} vs "
                    f"{_fmt(m['A_competencies']['evidence_size']['mean'], 2)} under A, "
                    f"D is not a richer matched evidence set for this scorer. The "
                    f"production gap rate therefore stands as reported against raw-CV "
                    f"substitution."
                )
            else:
                mm_a = sa["A_competencies"]["mismatched_pooled_gap_share"]
                mm_d = sa["D_raw_cv_chunks"]["mismatched_pooled_gap_share"]
                if mm_a is not None and mm_d is not None:
                    control_drop = mm_a - mm_d
                    parts.append(
                        f"True A→D gap-share drop is {_fmt(delta_ad)}; mismatched "
                        f"A→D drop is {_fmt(control_drop)}."
                    )
                    if delta_ad > control_drop + 0.05:
                        parts.append(
                            "Interpretation vs D (supported): the gap-rate reduction "
                            "under raw-CV evidence exceeds the mismatched-control "
                            "reduction — part of the production gap rate is an "
                            "evidence-coverage artefact."
                        )
                    elif abs(delta_ad - control_drop) <= 0.05:
                        parts.append(
                            "Interpretation vs D (supported): the A→D drop is similar "
                            "to the mismatched-control drop — mostly size bias; "
                            "production gap rate stands."
                        )
                    else:
                        parts.append(
                            "Interpretation vs D (supported): mixed size-bias vs "
                            "coverage effects."
                        )
        return " ".join(parts)

    lines.append(_interpret(primary))
    lines.append("")
    lines.append(_interpret(holdout))
    lines.append("")
    lines.append(
        "Numbers above are computed only; nulls mean unavailable. No estimated values."
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
