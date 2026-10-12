"""Figures 50–56 for frozen metrics and scale decision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.services.gap_sensitivity.figures import DPI, _savefig

ACCENT = {
    "B0": "#2A6F97",
    "B1": "#1B4332",
    "T": "#E76F51",
    "B2": "#E9C46A",
    "S0": "#6C757D",
    "S1": "#495057",
    "S2": "#ADB5BD",
}
FIT_COLORS = {"Good Fit": "#2A6F97", "Potential Fit": "#E76F51"}


def figure_50_baseline_distributions(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = ["M1.P", "M4.P", "M2.P", "M3.P"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, m in zip(axes.ravel(), metrics, strict=True):
        for label, color in FIT_COLORS.items():
            vals = [
                r[m]
                for r in rows
                if r.get("fit_label") == label and r.get(m) is not None
            ]
            if vals:
                ax.hist(vals, bins=15, alpha=0.55, label=label, color=color)
        ax.set_title(m)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 50. Baseline-150 primary metrics by fit label")
    _savefig(path)


def figure_51_validity(
    path: Path,
    v2: dict[str, Any],
    v3: dict[str, Any],
    v4: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    # Null control
    axes[0].bar(
        ["true M2", "null M2", "true M3", "null M3"],
        [
            v2.get("true_M2.P_mean") or 0,
            v2.get("null_M2.P_mean") or 0,
            v2.get("true_M3.P_mean") or 0,
            v2.get("null_M3.P_mean") or 0,
        ],
        color=["#2A6F97", "#ADB5BD", "#1B4332", "#CED4DA"],
    )
    axes[0].set_title("V2 null control")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(
        ["M1.P rises", "M4.P falls"],
        [
            v3.get("fraction_M1P_rises") or 0,
            v3.get("fraction_M4P_falls") or 0,
        ],
        color=["#E76F51", "#2A6F97"],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("V3 duplicate injection")

    metrics = ["M1.P", "M2.P", "M3.P", "M4.P"]
    axes[2].bar(
        metrics,
        [
            (v4.get(m) or {}).get("mean_abs_replicate_diff") or 0
            for m in metrics
        ],
        color="#E9C46A",
    )
    axes[2].set_title("V4 noise floor (|Δ|)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 51. Validity checks")
    _savefig(path)


def figure_52_pilot_by_condition(
    path: Path,
    condition_summaries: dict[str, dict[str, dict[str, float | None]]],
) -> None:
    """condition -> metric -> {mean, ci_low, ci_high}."""
    metrics = ["M1.P", "M2.P", "M3.P", "M4.P"]
    conditions = [c for c in ("B0", "B1", "T", "B2", "S0", "S1", "S2") if c in condition_summaries]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, m in zip(axes.ravel(), metrics, strict=True):
        means, lows, highs, colors = [], [], [], []
        for c in conditions:
            s = condition_summaries[c].get(m) or {}
            means.append(s.get("mean") or 0)
            lows.append(s.get("ci_low") or 0)
            highs.append(s.get("ci_high") or 0)
            colors.append(ACCENT.get(c, "#333"))
        x = np.arange(len(conditions))
        yerr = [
            np.maximum(0, np.asarray(means) - np.asarray(lows)),
            np.maximum(0, np.asarray(highs) - np.asarray(means)),
        ]
        ax.bar(x, means, color=colors, yerr=yerr, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_title(m)
    fig.suptitle("Figure 52. Pilot primary metrics by condition (95% CI)")
    _savefig(path)


def figure_53_win_matrix(
    path: Path,
    matrix: dict[str, dict[str, str]],
    holm_p: dict[str, dict[str, float | None]] | None = None,
) -> None:
    baselines = ["B0", "B1", "B2"]
    metrics = ["M1.P", "M2.P", "M3.P", "M4.P"]
    color_map = {"win": "#1B4332", "tie": "#ADB5BD", "lose": "#E76F51"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        for j, b in enumerate(baselines):
            outcome = (matrix.get(b) or {}).get(m, "tie")
            ax.add_patch(
                plt.Rectangle(
                    (j, len(metrics) - 1 - i),
                    1,
                    1,
                    facecolor=color_map.get(outcome, "#ccc"),
                    edgecolor="white",
                )
            )
            label = outcome
            if holm_p and holm_p.get(b) and holm_p[b].get(m) is not None:
                label = f"{outcome}\np={holm_p[b][m]:.3g}"
            ax.text(j + 0.5, len(metrics) - 0.5 - i, label, ha="center", va="center", fontsize=8, color="white")
    ax.set_xlim(0, len(baselines))
    ax.set_ylim(0, len(metrics))
    ax.set_xticks([x + 0.5 for x in range(len(baselines))])
    ax.set_xticklabels([f"T vs {b}" for b in baselines])
    ax.set_yticks([y + 0.5 for y in range(len(metrics))])
    ax.set_yticklabels(list(reversed(metrics)))
    ax.set_title("Figure 53. Win / tie / loss (Holm-adjusted p)")
    ax.set_aspect("equal")
    _savefig(path)


def figure_54_pool_oracle(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [r["label"] for r in rows]
    m3 = [r.get("M3.P") or 0 for r in rows]
    m2 = [r.get("M2.P") or 0 for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.15, m3, width=0.3, label="M3.P", color="#2A6F97")
    ax.bar(x + 0.15, m2, width=0.3, label="M2.P", color="#E76F51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(frameon=False)
    ax.set_title("Figure 54. Pool vs selected vs oracle (diagnostic)")
    _savefig(path)


def figure_55_lambda_sensitivity(path: Path, series: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals in series.items():
        ax.plot(list(range(len(vals))), vals, marker="o", label=name)
    ax.set_xticks(list(range(len(next(iter(series.values()), [])))))
    if series:
        # assume keys aligned to lambdas stored separately in labels
        pass
    ax.set_xlabel("lambda index (see report)")
    ax.set_title("Figure 55. Lambda sensitivity (diagnostic, not evidence)")
    ax.legend(frameon=False, fontsize=8)
    _savefig(path)


def figure_55_lambda_sensitivity_labeled(
    path: Path,
    lambdas: list[float],
    by_condition: dict[str, dict[str, list[float | None]]],
    metric: str = "M3.P",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond, metrics in by_condition.items():
        vals = metrics.get(metric) or []
        ax.plot(lambdas, vals, marker="o", label=f"{cond} {metric}", color=ACCENT.get(cond))
    ax.set_xlabel("λ")
    ax.set_title("Figure 55. Lambda sensitivity (diagnostic, not evidence)")
    ax.legend(frameon=False, fontsize=8)
    _savefig(path)


def figure_56_scale_cost(path: Path, breakdown: dict[str, float]) -> None:
    labels = list(breakdown.keys())
    vals = [breakdown[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals, color="#2A6F97")
    ax.set_ylabel("CNY")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Figure 56. Scale-cost projection (CNY, linear)")
    _savefig(path)
