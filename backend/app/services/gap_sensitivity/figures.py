"""PNG figures for gap-sensitivity analysis (matplotlib, 200 dpi)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.services.gap_sensitivity.evidence import CONDITION_A, CONDITION_D

DPI = 200
COLORS = {
    "strength": "#2A6F97",
    "partial": "#E9C46A",
    "gap": "#E76F51",
}


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def figure_17_gap_shares(
    path: Path,
    primary: dict[str, Any],
    holdout: dict[str, Any],
    conditions: list[str],
) -> None:
    """Stacked bars of strength/partial/gap shares by condition, primary vs holdout."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, summary) in zip(
        axes, [("Primary-30", primary), ("Holdout-30", holdout)], strict=True
    ):
        metrics = summary["metrics"]
        strength, partial, gap = [], [], []
        for cond in conditions:
            tc = metrics[cond]["three_category_counts"]
            total = sum(tc.values()) or 1
            strength.append(tc["strength"] / total)
            partial.append(tc["partial"] / total)
            gap.append(tc["gap"] / total)
        x = np.arange(len(conditions))
        ax.bar(x, strength, label="strength", color=COLORS["strength"])
        ax.bar(
            x,
            partial,
            bottom=strength,
            label="partial",
            color=COLORS["partial"],
        )
        bottom2 = np.asarray(strength) + np.asarray(partial)
        ax.bar(x, gap, bottom=bottom2, label="gap", color=COLORS["gap"])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of requirements")
        ax.set_title(title)
        ax.axhline(0, color="#333", lw=0.5)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Figure 17. Gap-category shares by evidence condition")
    _savefig(path)


def figure_18_gap_rate_distributions(
    path: Path,
    primary_rates: dict[str, np.ndarray],
    holdout_rates: dict[str, np.ndarray],
    primary_ci: dict[str, tuple[float, float, float]],
    holdout_ci: dict[str, tuple[float, float, float]],
    conditions: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, rates, cis) in zip(
        axes,
        [
            ("Primary-30", primary_rates, primary_ci),
            ("Holdout-30", holdout_rates, holdout_ci),
        ],
        strict=True,
    ):
        data = [rates[c] for c in conditions]
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#90BE6D")
            body.set_alpha(0.35)
        ax.boxplot(
            data,
            widths=0.25,
            showfliers=False,
            medianprops={"color": "#1B4332"},
        )
        means = [cis[c][0] for c in conditions]
        lows = [cis[c][1] for c in conditions]
        highs = [cis[c][2] for c in conditions]
        x = np.arange(1, len(conditions) + 1)
        ax.errorbar(
            x,
            means,
            yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
            fmt="o",
            color="#1B4332",
            capsize=3,
            label="mean ± 95% CI",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Per-pair gap rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Figure 18. Per-pair gap-rate distributions by condition")
    _savefig(path)


def figure_19_true_vs_mismatched(
    path: Path,
    primary: dict[str, Any],
    holdout: dict[str, Any],
    conditions: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = 0.35
    for ax, (title, summary) in zip(
        axes, [("Primary-30", primary), ("Holdout-30", holdout)], strict=True
    ):
        true_vals = [
            summary["size_adjusted"][c]["true_pooled_gap_share"] or 0 for c in conditions
        ]
        mm_vals = [
            summary["size_adjusted"][c]["mismatched_pooled_gap_share"] or 0
            for c in conditions
        ]
        x = np.arange(len(conditions))
        ax.bar(x - width / 2, true_vals, width, label="true", color="#2A6F97")
        ax.bar(x + width / 2, mm_vals, width, label="mismatched", color="#ADB5BD")
        effects = [
            summary["size_adjusted"][c]["size_adjusted_effect_pooled_share"]
            for c in conditions
        ]
        for i, eff in enumerate(effects):
            if eff is None:
                continue
            ax.text(i, max(true_vals[i], mm_vals[i]) + 0.02, f"Δ={eff:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Pooled gap share")
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
    axes[1].legend(frameon=False)
    fig.suptitle("Figure 19. True vs mismatched-control gap rate (size-adjusted Δ)")
    _savefig(path)


def figure_20_dose_response(
    path: Path,
    primary_sizes: dict[str, np.ndarray],
    primary_rates: dict[str, np.ndarray],
    holdout_sizes: dict[str, np.ndarray],
    holdout_rates: dict[str, np.ndarray],
    primary_rho: dict[str, float | None],
    holdout_rho: dict[str, float | None],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, sizes, rates, rhos) in zip(
        axes,
        [
            ("Primary-30", primary_sizes, primary_rates, primary_rho),
            ("Holdout-30", holdout_sizes, holdout_rates, holdout_rho),
        ],
        strict=True,
    ):
        for cond, color in [(CONDITION_A, "#2A6F97"), (CONDITION_D, "#E76F51")]:
            ax.scatter(
                sizes[cond],
                rates[cond],
                alpha=0.75,
                label=f"{cond} (ρ={rhos.get(cond)})",
                color=color,
            )
        ax.set_xlabel("Evidence-set size")
        ax.set_ylabel("Per-pair gap rate")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 20. Evidence-set size vs per-pair gap rate (A and D)")
    _savefig(path)


def figure_21_transition_heatmap(
    path: Path,
    primary_mat: dict[str, dict[str, int]],
    holdout_mat: dict[str, dict[str, int]],
) -> None:
    labels = [
        "strong_match",
        "partial_match",
        "weak_match",
        "missing",
    ]
    short = ["strong", "partial", "weak", "missing"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, mat) in zip(
        axes, [("Primary-30", primary_mat), ("Holdout-30", holdout_mat)], strict=True
    ):
        arr = np.array([[mat[a][b] for b in labels] for a in labels], dtype=float)
        im = ax.imshow(arr, cmap="YlOrRd")
        ax.set_xticks(range(4), short)
        ax.set_yticks(range(4), short)
        ax.set_xlabel("D label")
        ax.set_ylabel("A label")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Figure 21. A→D label-transition heatmap")
    _savefig(path)
