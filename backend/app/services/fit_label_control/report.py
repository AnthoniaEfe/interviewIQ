"""Markdown report writer. Never rewrites the frozen hypotheses block."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MARKER = "<!-- GENERATED_BELOW -->"


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def _desc_rows(desc: list[dict]) -> list[list[Any]]:
    out = []
    for r in desc:
        ci = f"[{_fmt(r.get('gap_rate_ci_low'))}, {_fmt(r.get('gap_rate_ci_high'))}]"
        out.append(
            [
                r.get("fit_label"),
                r.get("n_pairs"),
                r.get("n_requirements"),
                r.get("gap_share"),
                r.get("mean_per_pair_gap_rate"),
                ci,
                r.get("mean_combined"),
                r.get("strength_share"),
            ]
        )
    return out


def write_generated_report(
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    text = report_path.read_text(encoding="utf-8")
    if MARKER in text:
        prefix = text.split(MARKER, 1)[0].rstrip() + "\n\n" + MARKER + "\n"
    else:
        prefix = text.rstrip() + "\n\n" + MARKER + "\n"

    p = payload
    decision = p.get("decision") or {}
    lines = [
        "",
        "## Method",
        "",
        "Train-split pairs from `cnamuangtoun/resume-job-description-fit`. "
        "NF-primary-30: 30 No Fit pairs, `sample_experiment_pairs` with seed 42, "
        "excluding all pair ids in primary-30 / holdout-30 / primary-150. "
        "NF-holdout-30: 30 No Fit pairs, seed 2026, additionally excluding NF-primary-30. "
        "Fit-positive comparison sets are the existing Primary-30 (seed 42) and "
        "Holdout-30 (seed 2026) Good/Potential samples.",
        "",
        "Scoring is Condition A (production `_candidate_texts`, weights "
        "0.25 TF-IDF + 0.25 BM25_norm + 0.50 semantic, thresholds 0.65 / 0.40 / 0.20). "
        "Robustness: Condition D (raw CV chunks) and semantic-only labels with the same thresholds. "
        "Mismatched-candidate control: 5 permutations, same fit-label group, never self, seed 42.",
        "",
        f"- Git commit: `{p.get('git_commit')}`",
        f"- Extraction model: `{p.get('extraction_model')}`",
        f"- Prompt versions: `{json.dumps(p.get('prompt_versions'))}`",
        f"- Embedding: `{p.get('embedding_model')}` on `{p.get('embedding_device')}`",
        f"- Library versions: `{p.get('library_versions')}`",
        "",
        "## Regression check",
        "",
        _fmt_regression(p.get("regression")),
        "",
        "## Cost and extraction failures",
        "",
        f"- Actual paid extraction cost (USD): **{_fmt(p.get('actual_cost_usd'), 6)}**",
        f"- Cached/reused pairs: {_fmt(p.get('n_reused'))}",
        f"- Paid pair calls: {_fmt(p.get('n_paid'))}",
        f"- Failed pairs: {_fmt(p.get('n_failed'))}",
        "",
    ]
    failures = p.get("failures") or []
    if failures:
        lines.append("Failures:")
        for err in failures:
            lines.append(
                f"- `{err.get('pair_id')}` {err.get('error_type')}: {err.get('error')}"
            )
        lines.append("")
    else:
        lines.append("No extraction failures were recorded.")
        lines.append("")

    lines += [
        "## Descriptives (Condition A)",
        "",
        "### Primary",
        "",
        _md_table(
            [
                "label",
                "n pairs",
                "n req",
                "gap share",
                "mean gap rate",
                "95% CI",
                "mean combined",
                "strength share",
            ],
            _desc_rows((p.get("primary") or {}).get("A", {}).get("descriptives") or []),
        ),
        "",
        "### Holdout",
        "",
        _md_table(
            [
                "label",
                "n pairs",
                "n req",
                "gap share",
                "mean gap rate",
                "95% CI",
                "mean combined",
                "strength share",
            ],
            _desc_rows((p.get("holdout") or {}).get("A", {}).get("descriptives") or []),
        ),
        "",
        "### Pooled",
        "",
        _md_table(
            [
                "label",
                "n pairs",
                "n req",
                "gap share",
                "mean gap rate",
                "95% CI",
                "mean combined",
                "strength share",
            ],
            _desc_rows((p.get("pooled") or {}).get("A", {}).get("descriptives") or []),
        ),
        "",
        "## Gradient tests (per-pair gap rate, Condition A)",
        "",
        _fmt_gradient("Primary", (p.get("primary") or {}).get("A", {}).get("gradient_gap_rate")),
        "",
        _fmt_gradient("Holdout", (p.get("holdout") or {}).get("A", {}).get("gradient_gap_rate")),
        "",
        _fmt_gradient("Pooled", (p.get("pooled") or {}).get("A", {}).get("gradient_gap_rate")),
        "",
        "## Discrimination (AUC)",
        "",
        _fmt_auc_table(p),
        "",
        "## Robustness (Condition D and semantic-only)",
        "",
        "See combined CSVs under `experiments/results/fit_label_control_combined/`. "
        "Headline AUC values are in the table above.",
        "",
        "## Confound audit",
        "",
        _fmt_confound((p.get("pooled") or {}).get("A", {}).get("confound")),
        "",
        "## Matched-job analysis",
        "",
        p.get("matched_job_text") or "null",
        "",
        "## Figures",
        "",
        "![Figure 22](figures/fig22_gap_rate_by_fit_label.png)",
        "",
        "![Figure 23](figures/fig23_roc_fit_labels.png)",
        "",
        "![Figure 24](figures/fig24_stacked_shares_by_fit_label.png)",
        "",
        "![Figure 25](figures/fig25_confound_panel.png)",
        "",
        "## Pre-registered decision-rule outcome",
        "",
        f"**{decision.get('outcome', 'null')}**",
        "",
        f"- Primary No Fit vs Good Fit: Holm p={_fmt((decision.get('primary') or {}).get('holm_p'))}, "
        f"AUC={_fmt((decision.get('primary') or {}).get('auc'))}, "
        f"significant_coarse={_fmt((decision.get('primary') or {}).get('significant_coarse'))}",
        f"- Holdout No Fit vs Good Fit: Holm p={_fmt((decision.get('holdout') or {}).get('holm_p'))}, "
        f"AUC={_fmt((decision.get('holdout') or {}).get('auc'))}, "
        f"significant_coarse={_fmt((decision.get('holdout') or {}).get('significant_coarse'))}",
        f"- H1 monotonic claim allowed on both splits: "
        f"{_fmt(decision.get('h1_monotonic_allowed_both_splits'))} "
        f"(must not claim H1 unless this is true).",
        "",
        "## Limitations",
        "",
        "- Small n (30 No Fit / 15 Potential / 15 Good per split).",
        "- Human fit labels in this dataset are noisy; adjacent classes overlap.",
        "- Job-family and document-length confounds are possible; see the confound audit.",
        "- Single dataset (`cnamuangtoun/resume-job-description-fit`, train split).",
        "- Semantic similarity is still in the *score*; labels are independent of our scorer, "
        "which is the point of this control, but extraction quality still affects scores.",
        "",
        "## Deviations from the spec",
        "",
    ]
    for d in p.get("deviations") or []:
        lines.append(f"- {d}")
    if not p.get("deviations"):
        lines.append("- None recorded.")
    lines += [
        "",
        "## What this means for the thesis",
        "",
        p.get("thesis_paragraph") or "null",
        "",
        "## Files",
        "",
    ]
    for f in p.get("files") or []:
        lines.append(f"- `{f}`")
    report_path.write_text(prefix + "\n".join(lines), encoding="utf-8")


def _fmt_regression(reg: dict | None) -> str:
    if not reg:
        return "null"
    bits = []
    for split, item in reg.items():
        bits.append(
            f"- **{split}**: passed={_fmt(item.get('passed'))}, "
            f"counts_match_expected={_fmt(item.get('counts_match_expected'))}, "
            f"counts=`{item.get('counts')}`, expected=`{item.get('expected_counts')}`, "
            f"label mismatches={_fmt(item.get('n_label_mismatch'))}, "
            f"score mismatches={_fmt(item.get('n_score_mismatch'))}"
        )
    return "\n".join(bits)


def _fmt_gradient(title: str, grad: dict | None) -> str:
    if not grad:
        return f"### {title}\n\nnull"
    kw = grad.get("kruskal_wallis") or {}
    jt = grad.get("jonckheere_terpstra") or {}
    sp = grad.get("spearman_ordinal") or {}
    lines = [
        f"### {title}",
        "",
        f"- Kruskal-Wallis: H={_fmt(kw.get('statistic'))}, p={_fmt(kw.get('p_value'))}",
        f"- Jonckheere-Terpstra (Good < Potential < No Fit, greater, "
        f"{jt.get('n_permutations')} perms, seed {jt.get('seed')}): "
        f"S={_fmt(jt.get('statistic'))}, p={_fmt(jt.get('p_value'))}",
        f"- Spearman (ordinal 0/1/2 vs gap rate): ρ={_fmt(sp.get('rho'))}, p={_fmt(sp.get('p_value'))}",
        f"- H1 monotonic claim allowed: {_fmt(grad.get('h1_monotonic_claim_allowed'))}",
        "",
        _md_table(
            ["comparison", "U", "p", "Holm p", "Cliff δ", "δ 95% CI", "direction_ok"],
            [
                [
                    r.get("comparison"),
                    (r.get("mann_whitney") or {}).get("statistic"),
                    (r.get("mann_whitney") or {}).get("p_value"),
                    r.get("holm_p"),
                    (r.get("cliffs_delta") or {}).get("delta"),
                    f"[{_fmt((r.get('cliffs_delta') or {}).get('ci_low'))}, "
                    f"{_fmt((r.get('cliffs_delta') or {}).get('ci_high'))}]",
                    r.get("direction_ok_for_h1"),
                ]
                for r in grad.get("pairwise") or []
            ],
        ),
    ]
    return "\n".join(lines)


def _fmt_auc_table(payload: dict) -> str:
    rows = []
    for split in ("primary", "holdout", "pooled"):
        block = payload.get(split) or {}
        for cond in ("A", "D", "semantic"):
            disc = (block.get(cond) or {}).get("discrimination") or {}
            gap = disc.get("gap_rate") or {}
            for contrast in ("no_fit_vs_good", "no_fit_vs_fit_positive"):
                item = gap.get(contrast) or {}
                rows.append(
                    [
                        split,
                        cond,
                        contrast,
                        item.get("auc"),
                        item.get("ci_low"),
                        item.get("ci_high"),
                    ]
                )
    return _md_table(["split", "condition", "contrast", "AUC", "CI low", "CI high"], rows)


def _fmt_confound(conf: dict | None) -> str:
    if not conf:
        return "null"
    lines = []
    for key, item in (conf.get("kruskal") or {}).items():
        kw = item.get("kruskal_wallis") or {}
        lines.append(
            f"- **{item.get('title', key)}**: KW p={_fmt(kw.get('p_value'))}, "
            f"means=`{item.get('means')}`"
        )
        if item.get("note"):
            lines.append(f"  - {item['note']}")
    ols = conf.get("ols_hc3_secondary") or {}
    lines.append("")
    lines.append("Secondary OLS (HC3), low-powered:")
    if not ols.get("available"):
        lines.append(f"- unavailable: {ols.get('error')}")
    elif ols.get("error"):
        lines.append(f"- error: {ols.get('error')}")
    else:
        lines.append(f"- n={ols.get('n')}, R²={_fmt(ols.get('rsquared'))}")
        for c in ols.get("coefficients") or []:
            lines.append(
                f"- {c['name']}: coef={_fmt(c['coef'])} "
                f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}] p={_fmt(c['p_value'])}"
            )
    mm = (conf.get("kruskal") or {}).get("mismatch_gap_rate") or {}
    kw = mm.get("kruskal_wallis") or {}
    p = kw.get("p_value")
    if p is not None and p < 0.05:
        lines.append("")
        lines.append(
            "**Job-side confound: random-candidate gap rate differs by fit label "
            f"(Kruskal-Wallis p={_fmt(p)}). The JD side may drive part of the result.**"
        )
    return "\n".join(lines)
