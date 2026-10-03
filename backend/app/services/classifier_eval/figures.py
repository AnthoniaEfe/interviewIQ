"""Classifier reframing figures (PNG, 200 dpi; numbers 30–34)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import RocCurveDisplay  # noqa: E402

DPI = 200
COLORS = {
    "V0": "#2A6F97",
    "majority": "#6c757d",
    "gap": "#E76F51",
    "partial": "#E9C46A",
    "strength": "#2A6F97",
}


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def figure_30_headline(path: Path, rows: list[dict[str, Any]], majority_acc: float | None) -> None:
    names = [r["variant"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, key in enumerate(["macro_f1", "balanced_accuracy", "accuracy"]):
        vals = [r.get(key) for r in rows]
        lo = [r.get(f"{key}_ci_low") for r in rows]
        hi = [r.get(f"{key}_ci_high") for r in rows]
        offset = (i - 1) * 0.22
        yerr = None
        if all(a is not None and b is not None and v is not None for a, b, v in zip(lo, hi, vals)):
            yerr = np.vstack(
                [
                    [v - a for v, a in zip(vals, lo)],
                    [b - v for v, b in zip(vals, hi)],
                ]
            )
        ax.bar(x + offset, [v if v is not None else 0 for v in vals], width=0.2, yerr=yerr, label=key, capsize=3)
    if majority_acc is not None:
        ax.axhline(majority_acc, color=COLORS["majority"], linestyle="--", label="majority accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.set_title("Fig 30. Headline metrics by variant (bootstrap 95% CI)")
    _savefig(path)


def figure_31_per_class_f1(path: Path, per_class: dict[str, dict[str, float]]) -> None:
    variants = list(per_class.keys())
    classes = ["No Fit", "Potential Fit", "Good Fit"]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cls in enumerate(classes):
        vals = [per_class[v].get(cls, 0.0) for v in variants]
        ax.bar(x + (i - 1) * 0.22, vals, width=0.2, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Fig 31. Per-class F1 by variant")
    _savefig(path)


def figure_32_confusion(path: Path, cm_a: np.ndarray, cm_b: np.ndarray, name_a: str, name_b: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title in zip(axes, [cm_a, cm_b], [name_a, name_b], strict=True):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks([0, 1, 2], ["No", "Pot", "Good"])
        ax.set_yticks([0, 1, 2], ["No", "Pot", "Good"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig 32. Confusion matrices")
    _savefig(path)


def figure_33_binary_roc(path: Path, curves: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, y, scores in curves:
        if scores is None or len(np.unique(y)) < 2:
            continue
        RocCurveDisplay.from_predictions(y, scores, name=name, ax=ax)
    ax.set_title("Fig 33. Binary ROC (fit-positive vs No Fit)")
    _savefig(path)


def figure_34_importances(path: Path, items: list[dict[str, Any]]) -> None:
    items = sorted(items, key=lambda d: d.get("mean_abs_coef") or 0, reverse=True)
    names = [d["name"] for d in items]
    vals = [d.get("mean_abs_coef") or 0 for d in items]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], vals[::-1], color=COLORS["V0"])
    ax.set_xlabel("mean |coefficient|")
    ax.set_title("Fig 34. Gap-lite feature magnitudes (V4_LR)")
    _savefig(path)
