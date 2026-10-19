"""Blinded study builder. The condition key is returned separately and must not be served."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

STUDY_SEED = 20261019
CONDITIONS = ("T", "B1")
FORBIDDEN_PAYLOAD_KEYS = {
    "condition",
    "condition_id",
    "arm",
    "treatment",
    "baseline",
    "pair_id",
    "profile_id",
    "jd_id",
}


def _questions(raw: Any) -> list[str]:
    texts: list[str] = []
    for item in raw or []:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("question") or item.get("text") or "").strip()
        else:
            text = str(getattr(item, "question", "") or "").strip()
        if text:
            texts.append(text)
    return texts


def _summary(text: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    compact = " ".join(text.split())
    if len(compact) <= 280:
        return compact
    return compact[:277] + "..."


def _assert_blind(payload: dict) -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN_PAYLOAD_KEYS:
                    raise ValueError(f"blinded payload contains forbidden key {key!r}")
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)


def build_blinded_study(
    pairs: list[dict[str, Any]],
    rater_ids: list[str],
    *,
    seed: int = STUDY_SEED,
) -> dict[str, Any]:
    """Return blinded items, per-rater orders, and a separate condition key.

    Side assignment is balanced: half the pairs (floor) place T on Set X.
    Question order inside a set is shuffled once per item. Item order is shuffled
    separately for each rater.
    """
    if len(pairs) < 1:
        raise ValueError("study needs at least one pair")
    if len(rater_ids) < 1:
        raise ValueError("study needs at least one rater id")
    if len(set(rater_ids)) != len(rater_ids):
        raise ValueError("rater ids must be unique")
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_t_on_x = len(pairs) // 2
    x_is_t = [True] * n_t_on_x + [False] * (len(pairs) - n_t_on_x)
    rng.shuffle(x_is_t)

    items: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    for slot, pair_index in enumerate(order):
        pair = pairs[pair_index]
        t_questions = _questions(pair.get("T") or pair.get("set_t"))
        b_questions = _questions(pair.get("B1") or pair.get("set_b1"))
        if not t_questions or not b_questions:
            raise ValueError(f"pair index {pair_index} is missing T or B1 questions")
        rng.shuffle(t_questions)
        rng.shuffle(b_questions)
        item_id = f"item-{slot + 1:03d}"
        if x_is_t[slot]:
            set_x, set_y = t_questions, b_questions
            sides = {"X": "T", "Y": "B1"}
        else:
            set_x, set_y = b_questions, t_questions
            sides = {"X": "B1", "Y": "T"}
        job_text = str(pair.get("job_text") or "")
        candidate_text = str(pair.get("candidate_text") or "")
        items.append(
            {
                "item_id": item_id,
                "job_summary": _summary(job_text, pair.get("job_summary")),
                "candidate_summary": _summary(candidate_text, pair.get("candidate_summary")),
                "job_text": job_text,
                "candidate_text": candidate_text,
                "set_x": set_x,
                "set_y": set_y,
            }
        )
        key[item_id] = sides

    blinded_items = items
    _assert_blind({"items": blinded_items})
    assignments = []
    item_ids = [item["item_id"] for item in blinded_items]
    for rater_id in rater_ids:
        rater_rng = random.Random(f"{seed}:{rater_id}")
        item_order = list(item_ids)
        rater_rng.shuffle(item_order)
        assignments.append({"rater_id": rater_id, "item_order": item_order})

    blinded = {"seed": seed, "items": blinded_items, "assignments": assignments}
    manifest = {
        "seed": seed,
        "n_items": len(blinded_items),
        "n_raters": len(assignments),
        "n_t_on_x": sum(1 for sides in key.values() if sides["X"] == "T"),
        "condition_key_sha256": hashlib.sha256(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return {"blinded": blinded, "condition_key": key, "manifest": manifest}
