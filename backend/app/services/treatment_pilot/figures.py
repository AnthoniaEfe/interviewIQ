"""PNG figures 40–43 for the treatment pilot (200 dpi, CNY money axes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.services.gap_sensitivity.figures import COLORS, DPI, _savefig

ACCENT = {
    "B0": "#2A6F97",
    "B1": "#1B4332",
    "T": "#E76F51",
    "B2": "#E9C46A",
    "S0": "#6C757D",
    "S1": "#495057",
    "S2": "#ADB5BD",
}


def figure_40_cost_latency(
    path: Path,
    stage_rows: list[dict[str, Any]],
    pipeline_by_condition: dict[str, dict[str, float]],
) -> None:
    """Stacked full-pipeline cost (CNY) and latency (s) per pair by condition."""
    conditions = [c for c in ("B0", "B1", "T", "B2") if c in pipeline_by_condition]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Cost stacked by attributed stages
    stage_keys = ["extraction", "gap_analysis", "generation", "selection"]
    bottoms = np.zeros(len(conditions))
    stage_colors = {
        "extraction": COLORS["strength"],
        "gap_analysis": COLORS["partial"],
        "generation": COLORS["gap"],
        "selection": "#6C757D",
    }
    for sk in stage_keys:
        vals = []
        for c in conditions:
            vals.append(float(pipeline_by_condition[c].get(f"{sk}_cny", 0.0) or 0.0))
        axes[0].bar(conditions, vals, bottom=bottoms, label=sk, color=stage_colors[sk])
        bottoms = bottoms + np.asarray(vals)
    axes[0].set_ylabel("CNY per pair")
    axes[0].set_title("Full-pipeline cost")
    axes[0].legend(frameon=False, fontsize=8)

    lats = [float(pipeline_by_condition[c].get("latency_s", 0.0) or 0.0) for c in conditions]
    axes[1].bar(conditions, lats, color=[ACCENT.get(c, "#333") for c in conditions])
    axes[1].set_ylabel("Seconds per pair")
    axes[1].set_title("Full-pipeline latency")

    fig.suptitle("Figure 40. Full-pipeline cost and latency per pair by condition")
    _savefig(path)


def figure_41_redundancy(path: Path, diagnostics: dict[str, dict]) -> None:
    conditions = [c for c in ("B0", "B1", "T", "B2", "S0", "S1", "S2") if c in diagnostics]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = []
    positions = []
    means = []
    lows = []
    highs = []
    for i, c in enumerate(conditions):
        per = diagnostics[c]["per_pair"]
        vals = [
            r["redundancy"]["mean_pairwise_sim"]
            for r in per
            if r["redundancy"]["mean_pairwise_sim"] is not None
        ]
        data.append(vals)
        positions.append(i)
        ci = diagnostics[c]["redundancy_mean_pairwise"]
        means.append(ci.get("mean"))
        lows.append(ci.get("ci_low"))
        highs.append(ci.get("ci_high"))
    ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
               boxprops=dict(facecolor="#2A6F97", alpha=0.35))
    for i, c in enumerate(conditions):
        m, lo, hi = means[i], lows[i], highs[i]
        if m is None:
            continue
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color="#E76F51", capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Figure 41. Redundancy diagnostics by condition")
    _savefig(path)


def figure_42_coverage(path: Path, diagnostics: dict[str, dict]) -> None:
    conditions = [c for c in ("B0", "B1", "T", "B2", "S0", "S1", "S2") if c in diagnostics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cov_means = []
    cov_err = []
    for c in conditions:
        ci = diagnostics[c]["coverage_0.45"]
        m = ci.get("mean")
        cov_means.append(m if m is not None else 0.0)
        if m is not None and ci.get("ci_low") is not None:
            cov_err.append([m - ci["ci_low"], ci["ci_high"] - m])
        else:
            cov_err.append([0.0, 0.0])
    err = np.asarray(cov_err).T
    axes[0].bar(
        conditions,
        cov_means,
        yerr=err,
        color=[ACCENT.get(c, "#333") for c in conditions],
        capsize=3,
    )
    axes[0].set_ylabel("Coverage rate (thr=0.45)")
    axes[0].set_title("Requirement-coverage proxy")

    # gap/partial/strength shares among covered (pooled micro from per-pair means)
    cats = ["gap", "partial", "strength"]
    shares = {k: [] for k in cats}
    for c in conditions:
        g = p = s = n = 0.0
        for row in diagnostics[c]["per_pair"]:
            thr = row["coverage"]["by_threshold"].get("0.45") or {}
            nc = thr.get("n_covered") or 0
            if not nc:
                continue
            g += (thr.get("share_gap") or 0) * nc
            p += (thr.get("share_partial") or 0) * nc
            s += (thr.get("share_strength") or 0) * nc
            n += nc
        shares["gap"].append(g / n if n else 0.0)
        shares["partial"].append(p / n if n else 0.0)
        shares["strength"].append(s / n if n else 0.0)
    bottom = np.zeros(len(conditions))
    for cat, color in (
        ("gap", COLORS["gap"]),
        ("partial", COLORS["partial"]),
        ("strength", COLORS["strength"]),
    ):
        vals = np.asarray(shares[cat])
        axes[1].bar(conditions, vals, bottom=bottom, label=cat, color=color)
        bottom = bottom + vals
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Share of covered requirements")
    axes[1].set_title("Gap / partial / strength among covered")
    fig.suptitle(
        "Figure 42. Requirement-coverage proxy (embedding assignment, not human judgement)"
    )
    _savefig(path)


def figure_43_selection(path: Path, diagnostics: dict[str, dict], overlaps: dict) -> None:
    conditions = [c for c in ("S0", "S1", "S2", "B2", "T") if c in diagnostics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    red = [
        (diagnostics[c]["redundancy_mean_pairwise"] or {}).get("mean") or 0.0
        for c in conditions
    ]
    cov = [
        (diagnostics[c]["coverage_0.45"] or {}).get("mean") or 0.0 for c in conditions
    ]
    axes[0].bar(conditions, red, color=[ACCENT.get(c, "#333") for c in conditions])
    axes[0].set_ylabel("Mean pairwise similarity")
    axes[0].set_title("Redundancy")
    axes[1].bar(conditions, cov, color=[ACCENT.get(c, "#333") for c in conditions])
    axes[1].set_ylabel("Coverage rate (0.45)")
    axes[1].set_title("Coverage proxy")
    fig.suptitle("Figure 43. Selection-variant comparison on the shared pool")
    _savefig(path)
