"""Markdown reports for Part A / Part B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.gap_sensitivity.io_utils import project_root


def write_baseline_report(summary: dict[str, Any], *, freeze_path: Path) -> Path:
    root = project_root()
    path = root / "docs" / "metrics_baseline_report.md"
    dist = summary.get("distributions") or {}
    v = summary.get("validity") or {}
    counts = summary.get("counts") or {}
    lines = [
        "# Metrics baseline report (Part A)",
        "",
        f"**Experiment:** `{summary.get('experiment_id')}`  ",
        f"**Freeze file:** `{freeze_path}`  ",
        f"**Spec SHA-256:** `{summary.get('spec_sha256')}`  ",
        f"**Git commit:** `{summary.get('git_commit')}`  ",
        "",
        "## Limitation",
        "",
        "These metrics measure alignment with the gap matrix and surface diversity, "
        "**not** question quality. The fit-label control (`exp_20261003T163033Z`) was null "
        "(AUC 0.62 primary / 0.50 holdout), so gap labels are noisy.",
        "",
        "## Counts",
        "",
        f"- M1/M4 computed on **{counts.get('baseline_sets_M1_M4')}** of 150 B0 sets",
        f"- M2 computed on **{counts.get('baseline_sets_M2')}** pairs with gap matrices",
        f"- M3 computed on **{counts.get('baseline_sets_M3')}** pairs",
        f"- Gated extraction of missing 90: `{summary.get('extraction_missing')}`",
        "",
        "## Baseline distributions (mean, 95% CI)",
        "",
        "| Metric | Overall | Good Fit | Potential Fit |",
        "|--------|---------|----------|---------------|",
    ]
    for m in ("M1.P", "M4.P", "M2.P", "M3.P"):
        d = dist.get(m) or {}
        o = d.get("overall") or {}
        g = (d.get("by_fit_label") or {}).get("Good Fit") or {}
        p = (d.get("by_fit_label") or {}).get("Potential Fit") or {}
        lines.append(
            f"| {m} | {_fmt(o)} | {_fmt(g)} | {_fmt(p)} |"
        )
    lines.extend(["", "## Validity", ""])
    v1 = v.get("V1") or {}
    v2 = v.get("V2") or {}
    v3 = v.get("V3") or {}
    v4 = v.get("V4") or {}
    lines.append(f"- **V1 known-answer passed:** {v1.get('passed')}")
    lines.append(
        f"- **V2 null control:** true M2.P={v2.get('true_M2.P_mean')}, "
        f"null M2.P={v2.get('null_M2.P_mean')}, gap={v2.get('gap_M2.P')}; "
        f"true M3.P={v2.get('true_M3.P_mean')}, null M3.P={v2.get('null_M3.P_mean')}, "
        f"gap={v2.get('gap_M3.P')}"
    )
    lines.append(
        f"- **V3 duplicate injection:** fraction M1.P rises={v3.get('fraction_M1P_rises')}, "
        f"fraction M4.P falls={v3.get('fraction_M4P_falls')}"
    )
    lines.append("- **V4 noise floor (primary-30 replicates):**")
    for m, row in (v4 or {}).items():
        if isinstance(row, dict):
            lines.append(
                f"  - {m}: mean |Δ_replicate|={row.get('mean_abs_replicate_diff')}, "
                f"mean per-pair SD={row.get('mean_per_pair_sd')}"
            )
    lines.extend(
        [
            "",
            "## Freeze integrity",
            "",
            "- Pilot question files were **not** opened before writing the freeze file.",
            f"- Pilot stage/status dirs present at freeze: "
            f"`{summary.get('pilot_summaries_stage_status_only')}`",
            "",
            "If the researcher already inspected pilot outputs in a prior task, the "
            "subsequent scale decision is **advisory** (freeze not blind to that history).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_scale_decision_report(summary: dict[str, Any]) -> Path:
    root = project_root()
    path = root / "docs" / "scale_decision_report.md"
    decision = summary.get("decision") or {}
    comps = summary.get("comparisons") or []
    outcomes = summary.get("win_tie_loss") or {}
    diag = summary.get("diagnostics") or {}
    proj = summary.get("cost_projection") or {}
    lines = [
        "# Scale decision report (Part B)",
        "",
        f"**Experiment:** `{summary.get('experiment_id')}`  ",
        f"**Verdict:** **{decision.get('verdict')}** (advisory)  ",
        f"**Spec SHA-256:** `{summary.get('spec_sha256')}`  ",
        "",
        "## Frozen decision rule — literal clause outcomes",
        "",
    ]
    for c in decision.get("clauses") or []:
        lines.append(
            f"- Clause {c.get('id')}: {c.get('text')} → "
            f"**{'HOLD' if c.get('holds') else 'FAIL'}** "
            f"(detail: `{ {k:v for k,v in c.items() if k not in ('id','text')} }`)"
        )
    lines.extend(
        [
            "",
            f"**Rule outcome:** `{decision.get('verdict')}`",
            "",
            "## Validity caveat",
            "",
            summary.get("validity_caveat") or "",
            "",
            "## Twelve pre-registered comparisons (T vs baselines)",
            "",
            "| Baseline | Metric | Favourable Δ | 95% CI | p | p_Holm | r_rb | outcome |",
            "|----------|--------|--------------|--------|---|--------|------|---------|",
        ]
    )
    for c in comps:
        lines.append(
            f"| {c.get('baseline')} | {c.get('metric')} | "
            f"{_n(c.get('mean_favourable_diff'))} | "
            f"[{_n(c.get('ci_low'))}, {_n(c.get('ci_high'))}] | "
            f"{_n(c.get('wilcoxon_p'))} | {_n(c.get('wilcoxon_p_holm'))} | "
            f"{_n(c.get('rank_biserial'))} | {c.get('outcome')} |"
        )
    lines.extend(["", "## Win / tie / loss matrix", ""])
    lines.append("| | M1.P | M2.P | M3.P | M4.P |")
    lines.append("|--|------|------|------|------|")
    for b in ("B0", "B1", "B2"):
        row = outcomes.get(b) or {}
        lines.append(
            f"| T vs {b} | {row.get('M1.P')} | {row.get('M2.P')} | "
            f"{row.get('M3.P')} | {row.get('M4.P')} |"
        )

    nf = summary.get("noise_floor_V4") or {}
    lines.extend(["", "## V4 noise floor (quote next to comparisons)", ""])
    for m, row in nf.items():
        if isinstance(row, dict):
            lines.append(
                f"- {m}: smallest distinguishable paired mean diff ≈ "
                f"{row.get('smallest_distinguishable_paired_mean_diff')}"
            )

    d1 = diag.get("D1") or {}
    d8 = diag.get("D8") or {}
    lines.extend(
        [
            "",
            "## Diagnostics (not decision evidence)",
            "",
            f"- **D1 pool vs selected vs oracle:** {d1}",
            f"- **D5 failures:** {diag.get('D5')}",
            f"- **D6 robustness flips:** {(diag.get('D6') or {}).get('flips')}",
            f"- **D8 attribution:** {d8.get('likely_components')}",
            "",
            "## Scale-cost projection (linear extrapolation)",
            "",
            f"- Total: `{proj.get('total_usd')}` USD / `{proj.get('total_cny')}` CNY",
            f"- Breakdown: `{proj}`",
            f"- Budget: `{summary.get('scale_budget_cny')}` CNY",
            "",
            "## What this means for the thesis",
            "",
        ]
    )
    verdict = decision.get("verdict")
    if verdict == "SCALE":
        lines.append(
            "Under the pre-registered rule, scaling to 150 pairs is justified on the "
            "frozen alignment/diversity proxies. This does **not** establish question quality."
        )
    elif verdict == "SCALE_BLOCKED_BY_BUDGET":
        lines.append(
            "Scientific SCALE criteria held, but projected cost exceeds the budget cap "
            "(SCALE_BLOCKED_BY_BUDGET). The researcher decides whether to raise the budget."
        )
    else:
        lines.append(
            "Under the pre-registered rule the outcome is **DIAGNOSE**: do not spend on "
            "the 150-pair run until components are investigated. A DIAGNOSE / null result "
            "is a legitimate finding. Metrics are alignment and diversity proxies, not quality."
        )
    if summary.get("freeze_not_blind_to_prior_pilot"):
        lines.extend(
            [
                "",
                "> Note: pilot outputs existed and may have been inspected before this freeze; "
                "the decision is advisory.",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(d: dict) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    return f"{d['mean']:.4f} [{d.get('ci_low'):.4f}, {d.get('ci_high'):.4f}] (n={d.get('n')})"


def _n(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)
