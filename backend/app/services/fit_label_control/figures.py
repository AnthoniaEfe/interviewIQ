"""Figures 22–25 for the fit-label control (200 dpi, existing colour style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.services.fit_label_control.constants import LABEL_ORDER
from app.services.gap_sensitivity.figures import COLORS, DPI, _savefig


def _jitter(n: int, rng: np.random.Generator, scale: float = 0.08) -> np.ndarray:
    return rng.uniform(-scale, scale, size=n)


def figure_22_gap_rate_by_label(
    path: Path,
    primary_rows: list[dict],
    holdout_rows: list[dict],
    pooled_rows: list[dict],
) -> None:
    rng = np.random.default_rng(42)
    panels = [
        ("Primary", primary_rows),
        ("Holdout", holdout_rows),
        ("Pooled", pooled_rows),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (title, rows) in zip(axes, panels, strict=True):
        data = []
        for i, lab in enumerate(LABEL_ORDER, start=1):
            vals = np.asarray(
                [r["gap_rate"] for r in rows if r.get("fit_label") == lab],
                dtype=np.float64,
            )
            data.append(vals)
            if vals.size:
                ax.scatter(
                    np.full(vals.size, i) + _jitter(vals.size, rng),
                    vals,
                    s=18,
                    alpha=0.7,
                    color="#1B4332",
                    zorder=3,
                )
        ax.boxplot(
            data,
            widths=0.45,
            showfliers=False,
            medianprops={"color": "#E76F51"},
            boxprops={"color": "#2A6F97"},
        )
        ax.set_xticks(range(1, 4), ["Good", "Potential", "No Fit"], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Per-pair gap rate")
    fig.suptitle("Figure 22. Per-pair gap rate by fit label (Condition A)")
    _savefig(path)


def figure_23_roc(
    path: Path,
    curves: dict[str, Any],
) -> None:
    """curves keys like primary_A_nf_good with fpr/tpr/auc/ci."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    specs = [
        ("No Fit vs Good Fit", "nf_good"),
        ("No Fit vs fit-positive", "nf_pos"),
    ]
    style = {
        ("primary", "A"): ("#2A6F97", "-"),
        ("holdout", "A"): ("#2A6F97", "--"),
        ("primary", "D"): ("#E76F51", "-"),
        ("holdout", "D"): ("#E76F51", "--"),
    }
    for ax, (title, suffix) in zip(axes, specs, strict=True):
        ax.plot([0, 1], [0, 1], color="#ADB5BD", lw=1, ls=":")
        for split in ("primary", "holdout"):
            for cond in ("A", "D"):
                key = f"{split}_{cond}_{suffix}"
                item = curves.get(key) or {}
                fpr = item.get("fpr") or []
                tpr = item.get("tpr") or []
                if not fpr:
                    continue
                color, ls = style[(split, cond)]
                auc = item.get("auc")
                lo = item.get("ci_low")
                hi = item.get("ci_high")
                if auc is None:
                    lab = f"{split} {cond}"
                elif lo is None:
                    lab = f"{split} {cond} AUC={auc:.2f}"
                else:
                    lab = f"{split} {cond} AUC={auc:.2f} [{lo:.2f},{hi:.2f}]"
                ax.plot(fpr, tpr, color=color, ls=ls, label=lab)
        ax.set_title(title)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("Figure 23. ROC curves (higher gap → No Fit)")
    _savefig(path)


def figure_24_stacked_shares(
    path: Path,
    primary_desc: list[dict],
    holdout_desc: list[dict],
    pooled_desc: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (title, rows) in zip(
        axes,
        [("Primary", primary_desc), ("Holdout", holdout_desc), ("Pooled", pooled_desc)],
        strict=True,
    ):
        lookup = {r["fit_label"]: r for r in rows}
        strength, partial, gap = [], [], []
        for lab in LABEL_ORDER:
            r = lookup.get(lab, {})
            s = r.get("strength_share") or 0.0
            p = r.get("partial_share") or 0.0
            g = r.get("gap_share") or 0.0
            strength.append(s)
            partial.append(p)
            gap.append(g)
        x = np.arange(len(LABEL_ORDER))
        ax.bar(x, strength, color=COLORS["strength"], label="strength")
        ax.bar(x, partial, bottom=strength, color=COLORS["partial"], label="partial")
        bottom = np.asarray(strength) + np.asarray(partial)
        ax.bar(x, gap, bottom=bottom, color=COLORS["gap"], label="gap")
        ax.set_xticks(x, ["Good", "Potential", "No Fit"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Share of requirements")
    axes[2].legend(frameon=False, loc="upper right")
    fig.suptitle("Figure 24. Strength / partial / gap shares by fit label (Condition A)")
    _savefig(path)


def figure_25_confound(
    path: Path,
    rows: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [
        (axes[0, 0], "n_requirements", "Requirements per pair"),
        (axes[0, 1], "evidence_strings", "Evidence strings per pair"),
        (axes[1, 0], "jd_length", "JD length (chars)"),
        (axes[1, 1], "mismatch_gap_rate", "Random-candidate gap rate"),
    ]
    rng = np.random.default_rng(42)
    for ax, field, title in specs:
        data = []
        for i, lab in enumerate(LABEL_ORDER, start=1):
            vals = np.asarray(
                [r[field] for r in rows if r.get("fit_label") == lab and r.get(field) is not None],
                dtype=np.float64,
            )
            data.append(vals)
            if vals.size:
                ax.scatter(
                    np.full(vals.size, i) + _jitter(vals.size, rng),
                    vals,
                    s=14,
                    alpha=0.65,
                    color="#1B4332",
                    zorder=3,
                )
        if any(d.size for d in data):
            ax.boxplot(data, widths=0.4, showfliers=False, medianprops={"color": "#E76F51"})
        ax.set_xticks(range(1, 4), ["Good", "Potential", "No Fit"], fontsize=8)
        ax.set_title(title)
    fig.suptitle("Figure 25. Confound panel (pooled Condition A)")
    _savefig(path)
