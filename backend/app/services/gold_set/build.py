"""Write gold-set annotation pack (blind items, answer key, HTML, CSV)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.services.gold_set.assignments import assign_dual_coding, orders_for_annotators
from app.services.gold_set.constants import ANNOTATORS, ASSIGNMENT_SEED, SEED
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


def _write_annotator_templates(
    templates_dir: Path,
    assignment: dict[str, Any],
) -> None:
    templates_dir.mkdir(parents=True, exist_ok=True)
    for name in ANNOTATORS:
        _write_csv_template(
            templates_dir / f"{name}.csv",
            assignment["annotators"][name]["item_ids"],
        )


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
    assignment = assign_dual_coding(
        main,
        [it["item_id"] for it in cal],
        seed=ASSIGNMENT_SEED,
    )
    orders = orders_for_annotators(assignment, seed=ASSIGNMENT_SEED)
    write_json(out / "annotator_assignments.json", assignment)
    write_json(out / "annotator_orders.json", orders)
    write_json(
        out / "sampling_meta.json",
        {
            "seed": SEED,
            "assignment_seed": ASSIGNMENT_SEED,
            "n_annotators": len(ANNOTATORS),
            "items_per_annotator_total": assignment["items_per_annotator_total"],
            "primary": sampled["primary_meta"],
            "holdout": sampled["holdout_meta"],
            "n_main": len(main),
            "n_calibration": len(cal),
            "git_commit": git_commit_hash(),
            "library_versions": library_versions(),
        },
    )
    _write_csv_template(out / "annotation_template.csv", [it["item_id"] for it in all_items])
    _write_annotator_templates(out / "templates", assignment)
    html_path = out / "annotator_app.html"
    html_path.write_text(
        render_annotator_html(blind_cal + blind_main, orders),
        encoding="utf-8",
    )
    (out / "annotations" / "README.txt").write_text(
        "Place completed files here (one per annotator, assigned items only):\n"
        "  annotator_A.csv\n"
        "  annotator_B.csv\n"
        "  annotator_C.csv\n"
        "  annotator_D.csv\n"
        "  adjudicated.csv  (optional; disagreements on dual-coded main items)\n"
        "Columns: item_id,label,cannot_judge,snippet,notes\n"
        "label in {0,1,2}; cannot_judge true/false.\n"
        "Each annotator files calibration (c001–c010) plus 75 assigned main items.\n",
        encoding="utf-8",
    )
    return {
        "out_dir": str(out),
        "n_main": len(main),
        "n_calibration": len(cal),
        "n_annotators": len(ANNOTATORS),
        "items_per_annotator_total": assignment["items_per_annotator_total"],
        "primary": sampled["primary_meta"],
        "holdout": sampled["holdout_meta"],
    }
