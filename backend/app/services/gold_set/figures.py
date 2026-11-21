"""Figures 26–29 for the gold-set phase-2 analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 200
LABELS = ["Not evidenced", "Partial", "Met"]


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def figure_26_confusion(path: Path, confusion: list[list[int]]) -> None:
    matrix = np.asarray(confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("System three-category label")
    ax.set_ylabel("Human consensus")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="#1D3557")
    ax.set_title("Figure 26. System vs human consensus")
    _savefig(path)


def figure_27_roc(path: Path, roc: dict[str, dict[str, list[float]]], auc: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colors = {
        "combined": "#1D3557",
        "semantic": "#2A9D8F",
        "bm25_norm": "#E9C46A",
        "tfidf": "#E76F51",
    }
    for name in ("combined", "semantic", "bm25_norm", "tfidf"):
        curve = roc.get(name) or {}
        if not curve.get("fpr"):
            continue
        value = (auc.get(name) or {}).get("value")
        label = name if value is None else f"{name} (AUC {value:.2f})"
        ax.plot(curve["fpr"], curve["tpr"], color=colors[name], lw=2, label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="#6C757D", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 27. Met/partial vs not evidenced")
    ax.legend(frameon=False, loc="lower right")
    _savefig(path)


def figure_28_false_gaps(path: Path, g3: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    names = ["Evidence\nnot extracted", "Matched but\nscored low", "Missed gap\n(human 0, system ≥ 1)"]
    vals = [
        g3.get("cause_evidence_not_extracted") or 0,
        g3.get("cause_matched_but_scored_low") or 0,
        g3.get("missed_gap_n") or 0,
    ]
    colors = ["#E76F51", "#F4A261", "#2A6F97"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Consensus items")
    ax.set_title("Figure 28. False gaps and the complementary miss")
    for bar, val in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val), ha="center", va="bottom")
    _savefig(path)


def figure_29_recalibration(path: Path, g4: dict[str, Any]) -> None:
    adoption = g4.get("adoption_test") or {}
    improvement = adoption.get("improvement") or {}
    selected = adoption.get("selected") or {}
    labels = ["Production\non holdout", "Candidate\non holdout", "Candidate\non primary (tune)"]
    values = [
        improvement.get("qwk_production") if improvement else None,
        improvement.get("qwk_candidate") if improvement else None,
        selected.get("qwk_tune") if selected else None,
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    if all(v is None for v in values):
        ax.axis("off")
        ax.set_title("Figure 29. G4 not evaluated")
        ax.text(
            0.5,
            0.5,
            "Tune split has fewer than 8 consensus items.\nProduction weights and thresholds are kept.",
            ha="center",
            va="center",
            fontsize=12,
        )
        _savefig(path)
        return
    plot_vals = [0.0 if v is None else v for v in values]
    colors = ["#1D3557", "#2A9D8F", "#ADB5BD"]
    bars = ax.bar(labels, plot_vals, color=colors)
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_ylabel("Quadratic-weighted kappa")
    ax.set_title("Figure 29. G4 candidate vs production")
    for bar, val in zip(bars, values, strict=True):
        text = "n/a" if val is None else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom")
    _savefig(path)


def write_gold_figures(fig_dir: Path, payload: dict[str, Any]) -> None:
    confusion = payload["system_vs_human"]["pooled"]["confusion"]
    figure_26_confusion(fig_dir / "fig26_gold_confusion.png", confusion)
    pooled = payload["g2"]["pooled"]
    figure_27_roc(fig_dir / "fig27_gold_roc.png", pooled["roc"], pooled["auc"])
    figure_28_false_gaps(fig_dir / "fig28_gold_false_gaps.png", payload["g3"]["pooled"])
    figure_29_recalibration(fig_dir / "fig29_gold_recalibration.png", payload["g4"])
