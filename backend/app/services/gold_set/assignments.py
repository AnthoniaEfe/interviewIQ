"""Four-annotator dual-coding assignment for the frozen gold-set items.

Does not resample items. Each main item is labelled by exactly two of A–D.
Calibration items are labelled by all four.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.services.gold_set.constants import ANNOTATORS, ASSIGNMENT_SEED, PAIR_BLOCKS


def assign_dual_coding(
    main_items: list[dict[str, Any]],
    calibration_ids: list[str],
    *,
    seed: int = ASSIGNMENT_SEED,
) -> dict[str, Any]:
    """Balanced incomplete-block assignment: 6 pairs × 25 main items.

    Each annotator codes every calibration item plus 75 main items.
    """
    if len(PAIR_BLOCKS) != 6:
        raise ValueError("PAIR_BLOCKS must list the six unique annotator pairs")
    ids = [str(it["item_id"]) for it in main_items]
    if len(ids) != 150:
        raise ValueError(f"expected 150 main items, got {len(ids)}")
    if len(set(ids)) != 150:
        raise ValueError("main item_id values are not unique")

    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for it in main_items:
        key = (str(it.get("split") or ""), str(it.get("match_label") or ""))
        strata[key].append(str(it["item_id"]))

    pair_loads: list[list[str]] = [[] for _ in PAIR_BLOCKS]
    cursor = 0
    for stratum_key in sorted(strata):
        block = list(strata[stratum_key])
        rng.shuffle(block)
        for iid in block:
            pair_loads[cursor % len(PAIR_BLOCKS)].append(iid)
            cursor += 1

    annotators: dict[str, dict[str, Any]] = {
        name: {"main_ids": [], "calibration_ids": list(calibration_ids), "item_ids": []}
        for name in ANNOTATORS
    }
    item_to_annotators: dict[str, list[str]] = {}
    pair_records = []
    for block, (left, right) in zip(pair_loads, PAIR_BLOCKS):
        for iid in block:
            annotators[left]["main_ids"].append(iid)
            annotators[right]["main_ids"].append(iid)
            item_to_annotators[iid] = [left, right]
        pair_records.append(
            {
                "annotators": [left, right],
                "n_main": len(block),
                "main_ids": list(block),
            }
        )

    for name, rec in annotators.items():
        rec["main_ids"] = sorted(rec["main_ids"])
        rec["item_ids"] = list(calibration_ids) + rec["main_ids"]
        rec["n_main"] = len(rec["main_ids"])
        rec["n_calibration"] = len(calibration_ids)
        rec["n_total"] = len(rec["item_ids"])

    counts = [rec["n_main"] for rec in annotators.values()]
    if set(counts) != {75}:
        raise ValueError(f"unbalanced main loads: {counts}")
    if len(item_to_annotators) != 150:
        raise ValueError("every main item must be assigned to a pair")
    if any(len(v) != 2 for v in item_to_annotators.values()):
        raise ValueError("every main item must have exactly two annotators")

    return {
        "seed": seed,
        "design": "balanced_incomplete_block_dual_coding",
        "n_annotators": len(ANNOTATORS),
        "n_main": 150,
        "n_calibration": len(calibration_ids),
        "items_per_annotator_main": 75,
        "items_per_annotator_total": 75 + len(calibration_ids),
        "annotators": annotators,
        "item_to_annotators": item_to_annotators,
        "pairs": pair_records,
        "note": (
            "Each main item is dual-coded by one of the six A–D pairs. "
            "All four annotators complete the ten calibration items."
        ),
    }


def orders_for_annotators(
    assignment: dict[str, Any],
    *,
    seed: int,
) -> dict[str, list[str]]:
    """Calibration first (shuffled per annotator), then that annotator's main items."""
    orders: dict[str, list[str]] = {}
    for i, name in enumerate(ANNOTATORS):
        rec = assignment["annotators"][name]
        cal_rng = np.random.default_rng(seed + 101 + i)
        main_rng = np.random.default_rng(seed + 201 + i)
        cal = np.asarray(rec["calibration_ids"])
        main = np.asarray(rec["main_ids"])
        cal_order = cal[cal_rng.permutation(len(cal))].tolist() if len(cal) else []
        main_order = main[main_rng.permutation(len(main))].tolist() if len(main) else []
        orders[name] = cal_order + main_order
    return orders
