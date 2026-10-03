"""Deterministic No Fit subset sampling (train split, disjoint from used pairs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.datasets.resume_fit import (
    export_pair_files,
    load_split,
    sample_experiment_pairs,
)
from app.services.fit_label_control.constants import (
    FITPOS_ASSETS,
    HF_DATASET,
    HOLDOUT_SEED,
    N_NO_FIT,
    NF_SUBSET_DIRS,
    SEED,
    SOURCE_SPLIT,
    USED_MANIFESTS,
)
from app.services.gap_sensitivity.io_utils import project_root, write_json


def load_used_pair_ids(root: Path | None = None) -> set[str]:
    root = root or project_root()
    used: set[str] = set()
    for rel in USED_MANIFESTS:
        path = root / rel
        if not path.is_file():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            pid = item.get("pair_id")
            if pid:
                used.add(str(pid))
    return used


def sample_no_fit_pairs(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    exclude_ids: set[str],
) -> pd.DataFrame:
    """Sample ``n`` No Fit rows with ``df.sample(random_state=seed)``, excluding ids."""
    pool = df[(df["label"] == "No Fit") & ~df["pair_id"].isin(exclude_ids)]
    if len(pool) < n:
        raise ValueError(
            f"Need {n} No Fit pairs after exclusions; only {len(pool)} remain."
        )
    return sample_experiment_pairs(
        pool, n_per_label=n, seed=seed, labels=("No Fit",)
    )


def persist_nf_subsets(
    *,
    root: Path | None = None,
    n: int = N_NO_FIT,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Write NF-primary-30 and NF-holdout-30 subset files. Existing manifests untouched."""
    root = root or project_root()
    cache_dir = cache_dir or (root / "experiments/datasets/resume_job_fit/cache")
    train = load_split(SOURCE_SPLIT, cache_dir=cache_dir)
    used = load_used_pair_ids(root)

    primary = sample_no_fit_pairs(train, n=n, seed=SEED, exclude_ids=used)
    used_after_primary = used | set(primary["pair_id"].astype(str))
    holdout = sample_no_fit_pairs(
        train, n=n, seed=HOLDOUT_SEED, exclude_ids=used_after_primary
    )

    primary_dir = root / NF_SUBSET_DIRS["primary"]
    holdout_dir = root / NF_SUBSET_DIRS["holdout"]
    p_manifest = export_pair_files(primary, primary_dir)
    h_manifest = export_pair_files(holdout, holdout_dir)

    p_ids = [row["pair_id"] for row in p_manifest]
    h_ids = [row["pair_id"] for row in h_manifest]
    overlap = set(p_ids) & set(h_ids)
    if overlap:
        raise RuntimeError(f"NF primary/holdout overlap: {sorted(overlap)[:5]}")
    if set(p_ids) & used or set(h_ids) & used:
        raise RuntimeError("NF subsets intersect previously used pair ids")

    control_manifest = {
        "dataset": HF_DATASET,
        "source_split": SOURCE_SPLIT,
        "n_per_split": n,
        "primary": {
            "seed": SEED,
            "pair_ids": p_ids,
            "labels": ["No Fit"] * len(p_ids),
            "pairs_dir": NF_SUBSET_DIRS["primary"],
        },
        "holdout": {
            "seed": HOLDOUT_SEED,
            "pair_ids": h_ids,
            "labels": ["No Fit"] * len(h_ids),
            "pairs_dir": NF_SUBSET_DIRS["holdout"],
        },
        "excluded_used_pair_count": len(used),
        "fit_positive_assets": FITPOS_ASSETS,
    }
    combined_dir = root / "experiments/datasets/resume_job_fit/experiment_pairs_nf_control"
    combined_dir.mkdir(parents=True, exist_ok=True)
    write_json(combined_dir / "manifest.json", control_manifest)
    meta = {
        "source_split": SOURCE_SPLIT,
        "n_per_label": n,
        "labels": {"No Fit": n},
        "pairs": n,
    }
    (primary_dir / "sample_meta.json").write_text(
        json.dumps({**meta, "seed": SEED, "split": "nf-primary-30"}, indent=2),
        encoding="utf-8",
    )
    (holdout_dir / "sample_meta.json").write_text(
        json.dumps({**meta, "seed": HOLDOUT_SEED, "split": "nf-holdout-30"}, indent=2),
        encoding="utf-8",
    )
    return control_manifest
