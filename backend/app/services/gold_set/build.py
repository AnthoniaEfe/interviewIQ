"""Write gold-set annotation pack (blind items, answer key, HTML, CSV)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from app.services.gold_set.constants import SEED
from app.services.gold_set.html_app import render_annotator_html
from app.services.gold_set.sampling import sample_gold_items
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    project_root,
    write_json,
)


BLIND_FIELDS = (
    "item_id",
    "requirement_text",
    "cv_text",
    "jd_text",
    "is_calibration",
)


def _assign_ids(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, it in enumerate(items, start=1):
        row = dict(it)
        row["item_id"] = f"{prefix}{i:03d}"
        out.append(row)
    return out


def _blind(items: list[dict]) -> list[dict]:
    return [{k: it[k] for k in BLIND_FIELDS} for it in items]


def _key_record(it: dict) -> dict:
    return {
        "item_id": it["item_id"],
        "pair_id": it["pair_id"],
        "split": it["split"],
        "fit_label": it["fit_label"],
        "requirement_id": it["requirement_id"],
        "requirement_index": it["requirement_index"],
        "priority": it["priority"],
        "system_label": it["match_label"],
        "tfidf": it["tfidf"],
        "bm25_raw": it["bm25_raw"],
        "bm25_norm": it["bm25_norm"],
        "semantic": it["semantic"],
        "combined": it["combined"],
        "evidence_string": it["evidence_string"],
        "inclusion_probability": it.get("inclusion_probability"),
        "design_weight": it.get("design_weight"),
        "stratum_size": it.get("stratum_size"),
        "stratum_sample_size": it.get("stratum_sample_size"),
        "is_calibration": it.get("is_calibration", False),
    }


def _write_csv_template(path: Path, item_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["item_id", "label", "cannot_judge", "snippet", "notes"],
        )
        w.writeheader()
        for iid in item_ids:
            w.writerow(
                {
                    "item_id": iid,
                    "label": "",
                    "cannot_judge": "false",
                    "snippet": "",
                    "notes": "",
                }
            )


def _orders(item_ids: list[str], seed: int) -> dict[str, list[str]]:
    a = np.random.default_rng(seed + 1)
    b = np.random.default_rng(seed + 2)
    arr = np.asarray(item_ids)
    return {
        "annotator_A": arr[a.permutation(len(arr))].tolist(),
        "annotator_B": arr[b.permutation(len(arr))].tolist(),
    }


def build_gold_set(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    sampled = sample_gold_items(root, seed=SEED)
    main = _assign_ids(sampled["main"], "g")
    cal = _assign_ids(sampled["calibration"], "c")
    all_items = cal + main

    out = root / "experiments/gold_set"
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(exist_ok=True)

    blind_main = _blind(main)
    blind_cal = _blind(cal)
    write_json(out / "items_blind.json", {"items": blind_main, "calibration": blind_cal})
    write_json(
        out / "answer_key_do_not_open_before_annotating.json",
        {
            "warning": "Do not open until annotation is finished.",
            "items": [_key_record(it) for it in all_items],
        },
    )
    orders = _orders([it["item_id"] for it in all_items], SEED)
    write_json(out / "annotator_orders.json", orders)
    write_json(
        out / "sampling_meta.json",
        {
            "seed": SEED,
            "primary": sampled["primary_meta"],
            "holdout": sampled["holdout_meta"],
            "n_main": len(main),
            "n_calibration": len(cal),
            "git_commit": git_commit_hash(),
            "library_versions": library_versions(),
        },
    )
    _write_csv_template(out / "annotation_template.csv", [it["item_id"] for it in all_items])
    html_path = out / "annotator_app.html"
    html_path.write_text(
        render_annotator_html(blind_cal + blind_main, orders),
        encoding="utf-8",
    )
    (out / "annotations" / "README.txt").write_text(
        "Place completed files here:\n"
        "  annotator_A.csv\n"
        "  annotator_B.csv\n"
        "  adjudicated.csv  (optional)\n"
        "Columns: item_id,label,cannot_judge,snippet,notes\n"
        "label in {0,1,2}; cannot_judge true/false.\n",
        encoding="utf-8",
    )
    return {
        "out_dir": str(out),
        "n_main": len(main),
        "n_calibration": len(cal),
        "primary": sampled["primary_meta"],
        "holdout": sampled["holdout_meta"],
    }
