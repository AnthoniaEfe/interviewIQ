"""Five-annotator dual-coding assignment for a 30-pair review cap.

Does not resample the frozen pool. Each reviewer labels the shared calibration
items plus 20 main items (30 CV/JD pairs total). Each assigned main item is
labelled by exactly two of A–E.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.services.gold_set.constants import (
    ANNOTATORS,
    ASSIGNED_MAIN,
    ASSIGNED_STRATUM_PER_SPLIT,
    ASSIGNMENT_SEED,
    CALIBRATION_N,
    ITEMS_PER_ANNOTATOR,
    MAIN_PER_ANNOTATOR,
    PAIR_BLOCKS,
    PER_SPLIT,
)


def _select_main_ids(main_items: list[dict[str, Any]], rng: np.random.Generator) -> list[str]:
    """Take ASSIGNED_MAIN items, balanced across split and match label."""
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for it in main_items:
        key = (str(it.get("split") or ""), str(it.get("match_label") or ""))
        by_key[key].append(str(it["item_id"]))
    splits = sorted({key[0] for key in by_key})
    if len(splits) != 2:
        raise ValueError(f"expected two splits, got {splits}")
    selected: list[str] = []
    for split in splits:
        for label, n_take in ASSIGNED_STRATUM_PER_SPLIT.items():
            block = list(by_key.get((split, label), []))
            if len(block) < n_take:
                raise ValueError(
                    f"{split}/{label} has {len(block)} items; need {n_take} for the review cap"
                )
            rng.shuffle(block)
            selected.extend(block[:n_take])
    if len(selected) != ASSIGNED_MAIN:
        raise ValueError(f"expected {ASSIGNED_MAIN} assigned main items, got {len(selected)}")
    if len(set(selected)) != len(selected):
        raise ValueError("assigned main item_id values are not unique")
    return selected


def assign_dual_coding(
    main_items: list[dict[str, Any]],
    calibration_ids: list[str],
    *,
    seed: int = ASSIGNMENT_SEED,
) -> dict[str, Any]:
    """Assign 50 of 150 main items across the ten A–E pairs, 5 items each.

    Every annotator then has exactly 10 calibration + 20 main = 30 pairs.
    """
    n_blocks = len(ANNOTATORS) * (len(ANNOTATORS) - 1) // 2
    if len(PAIR_BLOCKS) != n_blocks:
        raise ValueError(f"PAIR_BLOCKS must list the {n_blocks} unique annotator pairs")
    if len(calibration_ids) != CALIBRATION_N:
        raise ValueError(
            f"expected {CALIBRATION_N} calibration items, got {len(calibration_ids)}"
        )
    ids = [str(it["item_id"]) for it in main_items]
    if len(ids) != PER_SPLIT * 2:
        raise ValueError(f"expected {PER_SPLIT * 2} main items, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("main item_id values are not unique")

    rng = np.random.default_rng(seed)
    chosen = set(_select_main_ids(main_items, rng))
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for it in main_items:
        iid = str(it["item_id"])
        if iid not in chosen:
            continue
        key = (str(it.get("split") or ""), str(it.get("match_label") or ""))
        strata[key].append(iid)

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
    if set(counts) != {MAIN_PER_ANNOTATOR}:
        raise ValueError(f"unbalanced main loads: {counts}")
    totals = [rec["n_total"] for rec in annotators.values()]
    if set(totals) != {ITEMS_PER_ANNOTATOR}:
        raise ValueError(f"review cap is {ITEMS_PER_ANNOTATOR}, loads were {totals}")
    if len(annotators) != len(ANNOTATORS):
        raise ValueError(f"expected {len(ANNOTATORS)} annotators, got {len(annotators)}")
    if len(item_to_annotators) != ASSIGNED_MAIN:
        raise ValueError("every selected main item must be assigned to a pair")
    if any(len(v) != 2 for v in item_to_annotators.values()):
        raise ValueError("every assigned main item must have exactly two annotators")
    block_sizes = [len(block) for block in pair_loads]
    if len(set(block_sizes)) != 1:
        raise ValueError(f"unbalanced pair blocks: {block_sizes}")

    return {
        "seed": seed,
        "design": "five_annotator_dual_coding_cap_30",
        "n_annotators": len(ANNOTATORS),
        "n_main_pool": len(ids),
        "n_main": ASSIGNED_MAIN,
        "n_calibration": len(calibration_ids),
        "items_per_annotator_main": MAIN_PER_ANNOTATOR,
        "items_per_annotator_total": ITEMS_PER_ANNOTATOR,
        "max_annotators": len(ANNOTATORS),
        "max_items_per_annotator": ITEMS_PER_ANNOTATOR,
        "annotators": annotators,
        "item_to_annotators": item_to_annotators,
        "pairs": pair_records,
        "note": (
            "Review cap: annotators A–E, 30 CV/JD pairs each "
            "(10 calibration shared by all five, plus 20 main). "
            "50 of the frozen 150 main items are dual-coded; "
            "every pair of annotators shares 5 main items. "
            "Unassigned main items are not reviewed."
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
