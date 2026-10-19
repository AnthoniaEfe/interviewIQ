"""Pair-set roles for the 150-pair experiment. Holdout-30 is confirmation only."""

from __future__ import annotations

import json
from pathlib import Path

PRIMARY_150_MANIFEST = (
    "experiments/datasets/resume_job_fit/experiment_pairs_150/manifest.json"
)
HOLDOUT_30_MANIFEST = (
    "experiments/datasets/resume_job_fit/experiment_pairs_holdout30/manifest.json"
)
PRIMARY_30_MANIFEST = (
    "experiments/datasets/resume_job_fit/experiment_pairs/manifest.json"
)


def load_pair_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a pair manifest list")
    ids: list[str] = []
    for row in payload:
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id:
            raise ValueError(f"{path} has a row without pair_id")
        ids.append(pair_id)
    return ids


def split_roles(root: Path) -> dict[str, list[str]]:
    """Development is primary-150 minus holdout-30. Confirmation is holdout-30.

    The returned development list never contains a holdout id.
    """
    primary_150 = load_pair_ids(root / PRIMARY_150_MANIFEST)
    holdout = load_pair_ids(root / HOLDOUT_30_MANIFEST)
    primary_30 = load_pair_ids(root / PRIMARY_30_MANIFEST)
    holdout_set = set(holdout)
    if len(holdout_set) != len(holdout):
        raise ValueError("holdout-30 manifest has duplicate pair ids")
    development = [pair_id for pair_id in primary_150 if pair_id not in holdout_set]
    leaked = [pair_id for pair_id in development if pair_id in holdout_set]
    if leaked:
        raise ValueError(f"holdout ids leaked into development: {leaked}")
    missing_holdout = [pair_id for pair_id in holdout if pair_id not in set(primary_150)]
    if missing_holdout:
        raise ValueError(
            "holdout-30 ids missing from primary-150: " + ", ".join(missing_holdout[:5])
        )
    return {
        "primary_150": primary_150,
        "holdout_30": holdout,
        "primary_30": primary_30,
        "development_120": development,
        "confirmatory_30": list(holdout),
    }
