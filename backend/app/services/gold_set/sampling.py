"""Load stored gap requirements and sample the gold-set items."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.services.gold_set.constants import (
    CALIBRATION_N,
    HOLDOUT,
    MAX_PER_PAIR,
    MIN_PAIRS_PER_SPLIT,
    PER_SPLIT,
    PRIMARY,
    SEED,
    STRATUM_TARGETS,
)
from app.services.gap_sensitivity.io_utils import project_root


def _combined(tfidf: float, bm25: float, bm25_max: float, semantic: float) -> float:
    bm25_norm = (bm25 / bm25_max) if bm25_max > 0 else 0.0
    return 0.25 * tfidf + 0.25 * bm25_norm + 0.50 * semantic


def load_split_requirements(root: Path, spec: dict) -> list[dict[str, Any]]:
    gap_dir = root / "experiments/results" / spec["gap"] / "gap_analysis"
    pairs_dir = root / spec["pairs_dir"]
    items: list[dict[str, Any]] = []
    for path in sorted(gap_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        pair_id = rec.get("pair_id") or rec.get("profile_id")
        gaps = rec.get("gap_matrix", {}).get("gaps") or []
        bm25_max = max(
            (float(g["evidence"].get("bm25_score") or 0.0) for g in gaps),
            default=0.0,
        )
        cv_path = pairs_dir / "profiles" / f"{pair_id}.txt"
        jd_path = pairs_dir / "jobs" / f"{pair_id}.txt"
        cv_text = cv_path.read_text(encoding="utf-8") if cv_path.is_file() else ""
        jd_text = jd_path.read_text(encoding="utf-8") if jd_path.is_file() else ""
        for i, g in enumerate(gaps):
            ev = g.get("evidence") or {}
            tfidf = float(ev.get("tfidf_score") or 0.0)
            bm25 = float(ev.get("bm25_score") or 0.0)
            sem = float(ev.get("semantic_score") or 0.0)
            items.append(
                {
                    "pair_id": pair_id,
                    "split": spec["name"],
                    "fit_label": rec.get("fit_label"),
                    "requirement_id": f"{pair_id}::{i:03d}",
                    "requirement_index": i,
                    "requirement_text": g.get("job_competency"),
                    "priority": g.get("priority"),
                    "match_label": g.get("match_label"),
                    "tfidf": tfidf,
                    "bm25_raw": bm25,
                    "bm25_norm": (bm25 / bm25_max) if bm25_max > 0 else 0.0,
                    "semantic": sem,
                    "combined": _combined(tfidf, bm25, bm25_max, sem),
                    "evidence_string": ev.get("matched_candidate_competency"),
                    "cv_text": cv_text,
                    "jd_text": jd_text,
                }
            )
    return items


def _sample_split(
    items: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_label[str(it["match_label"])].append(it)
    stratum_n = {k: len(v) for k, v in by_label.items()}

    # Shuffle within strata
    for lab, rows in by_label.items():
        order = rng.permutation(len(rows))
        by_label[lab] = [rows[i] for i in order]

    selected: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    stratum_got: Counter[str] = Counter()

    def try_add(it: dict) -> bool:
        lab = str(it["match_label"])
        if stratum_got[lab] >= STRATUM_TARGETS.get(lab, 0):
            return False
        if pair_counts[it["pair_id"]] >= MAX_PER_PAIR:
            return False
        if it["requirement_id"] in {s["requirement_id"] for s in selected}:
            return False
        selected.append(it)
        pair_counts[it["pair_id"]] += 1
        stratum_got[lab] += 1
        return True

    # Pass 1: prefer new pairs to reach MIN_PAIRS_PER_SPLIT
    for lab in STRATUM_TARGETS:
        for it in by_label.get(lab, []):
            if len(pair_counts) < MIN_PAIRS_PER_SPLIT and pair_counts[it["pair_id"]] > 0:
                continue
            try_add(it)

    # Pass 2: fill remaining stratum slots
    for lab in STRATUM_TARGETS:
        for it in by_label.get(lab, []):
            try_add(it)

    # Pass 3: if still short on pairs, pull from unused pairs even if a stratum is full
    # (do not exceed PER_SPLIT)
    if len(pair_counts) < MIN_PAIRS_PER_SPLIT:
        unused = [it for it in items if pair_counts[it["pair_id"]] == 0]
        order = rng.permutation(len(unused))
        unused = [unused[i] for i in order]
        for it in unused:
            if len(selected) >= PER_SPLIT:
                break
            if len(pair_counts) >= MIN_PAIRS_PER_SPLIT:
                break
            if pair_counts[it["pair_id"]] >= MAX_PER_PAIR:
                continue
            if it["requirement_id"] in {s["requirement_id"] for s in selected}:
                continue
            selected.append(it)
            pair_counts[it["pair_id"]] += 1
            stratum_got[str(it["match_label"])] += 1

    # Trim to PER_SPLIT if over (should not happen)
    if len(selected) > PER_SPLIT:
        selected = selected[:PER_SPLIT]

    sampled_n = Counter(str(s["match_label"]) for s in selected)
    for s in selected:
        lab = str(s["match_label"])
        n_s = sampled_n[lab]
        n_pop = stratum_n.get(lab, 0)
        pi = (n_s / n_pop) if n_pop else None
        s["inclusion_probability"] = pi
        s["design_weight"] = (n_pop / n_s) if n_s else None
        s["stratum_size"] = n_pop
        s["stratum_sample_size"] = n_s

    meta = {
        "n_selected": len(selected),
        "n_pairs": len(pair_counts),
        "stratum_population": stratum_n,
        "stratum_sampled": dict(sampled_n),
        "fit_label_pairs": dict(
            Counter(items_pair_fit(selected))
        ),
        "targets": dict(STRATUM_TARGETS),
    }
    return selected, meta


def items_pair_fit(selected: list[dict]) -> list[str]:
    seen = {}
    for s in selected:
        seen[s["pair_id"]] = s.get("fit_label") or "unknown"
    return list(seen.values())


def sample_gold_items(
    root: Path | None = None,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    root = root or project_root()
    rng = np.random.default_rng(seed)
    primary_items = load_split_requirements(root, PRIMARY)
    holdout_items = load_split_requirements(root, HOLDOUT)
    p_sel, p_meta = _sample_split(primary_items, rng)
    h_sel, h_meta = _sample_split(holdout_items, rng)
    main = p_sel + h_sel
    main_ids = {m["requirement_id"] for m in main}
    main_pairs = {m["pair_id"] for m in main}

    # Calibration: unused requirements from the same pairs
    cal_pool = [
        it
        for it in primary_items + holdout_items
        if it["pair_id"] in main_pairs and it["requirement_id"] not in main_ids
    ]
    order = rng.permutation(len(cal_pool)) if cal_pool else np.asarray([], dtype=int)
    cal_pool = [cal_pool[i] for i in order]
    calibration = cal_pool[:CALIBRATION_N]
    for i, it in enumerate(calibration):
        it = dict(it)
        it["inclusion_probability"] = None
        it["design_weight"] = None
        it["stratum_size"] = None
        it["stratum_sample_size"] = None
        it["is_calibration"] = True
        calibration[i] = it
    for it in main:
        it["is_calibration"] = False

    return {
        "seed": seed,
        "main": main,
        "calibration": calibration,
        "primary_meta": p_meta,
        "holdout_meta": h_meta,
    }
