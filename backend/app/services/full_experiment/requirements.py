"""Requirement-count summary for stored gap records. Local only."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

STORED_GAP_DIRS = (
    "experiments/results/exp_20260927T215232Z/gap_analysis",
    "experiments/results/exp_20260927T223303Z/gap_analysis",
)


def summarize_gap_dir(path: Path) -> dict:
    counts: list[int] = []
    empty: list[str] = []
    for file in sorted(path.glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        gaps = ((payload.get("gap_matrix") or {}).get("gaps")) or []
        if not isinstance(gaps, list):
            raise ValueError(f"{file.name} gap_matrix.gaps is not a list")
        if len(gaps) == 0:
            empty.append(file.name)
        counts.append(len(gaps))
    if not counts:
        return {
            "n_pairs": 0,
            "min": None,
            "median": None,
            "max": None,
            "n_below_10": 0,
            "n_empty": 0,
        }
    return {
        "n_pairs": len(counts),
        "min": min(counts),
        "median": float(statistics.median(counts)),
        "max": max(counts),
        "n_below_10": sum(1 for count in counts if count < 10),
        "n_empty": len(empty),
    }


def summarize_stored_gaps(root: Path) -> dict:
    parts = []
    all_counts_note = []
    for relative in STORED_GAP_DIRS:
        summary = summarize_gap_dir(root / relative)
        summary["dir"] = relative
        parts.append(summary)
        all_counts_note.append(summary)
    # Recompute the pooled summary from the files so the total is not a guess.
    pooled_counts: list[int] = []
    for relative in STORED_GAP_DIRS:
        for file in sorted((root / relative).glob("*.json")):
            payload = json.loads(file.read_text(encoding="utf-8"))
            gaps = ((payload.get("gap_matrix") or {}).get("gaps")) or []
            pooled_counts.append(len(gaps))
    pooled = {
        "n_pairs": len(pooled_counts),
        "min": min(pooled_counts) if pooled_counts else None,
        "median": float(statistics.median(pooled_counts)) if pooled_counts else None,
        "max": max(pooled_counts) if pooled_counts else None,
        "n_below_10": sum(1 for count in pooled_counts if count < 10),
    }
    return {"by_dir": parts, "pooled_existing_60": pooled}
