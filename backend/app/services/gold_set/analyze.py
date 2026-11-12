"""Phase 2 gold-set analysis. Refuses to run without valid annotation files."""

from __future__ import annotations

import csv
from pathlib import Path

from app.services.gold_set.constants import ANNOTATORS

REQUIRED_COLS = {"item_id", "label", "cannot_judge", "snippet", "notes"}


class GoldSetAnnotationError(RuntimeError):
    pass


def _read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise GoldSetAnnotationError(f"{path} is not a list of records")
        return data
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_and_validate_annotations(
    path: Path,
    expected_ids: set[str],
    *,
    min_completion: float = 0.90,
) -> list[dict]:
    if not path.is_file():
        raise GoldSetAnnotationError(
            f"Annotation file missing: {path}. "
            "Place real annotator files under experiments/gold_set/annotations/ "
            "and do not fabricate labels."
        )
    rows = _read_table(path)
    cols = set(rows[0].keys()) if rows else set()
    missing = REQUIRED_COLS - cols
    if missing:
        raise GoldSetAnnotationError(f"{path} missing columns {sorted(missing)}")
    by_id = {str(r.get("item_id")): r for r in rows if r.get("item_id")}
    present = set(by_id) & expected_ids
    rate = len(present) / len(expected_ids) if expected_ids else 0.0
    if rate < min_completion:
        raise GoldSetAnnotationError(
            f"{path} covers {rate:.1%} of items (need ≥{min_completion:.0%}). "
            "Refusing to analyse incomplete or invalid annotations."
        )
    return [by_id[i] for i in expected_ids if i in by_id]


def run_gold_set_analyze(root: Path | None = None) -> dict:
    from app.services.gap_sensitivity.io_utils import project_root, load_json

    root = root or project_root()
    pack = root / "experiments/gold_set"
    items_path = pack / "items_blind.json"
    if not items_path.is_file():
        raise GoldSetAnnotationError("Gold-set pack not built. Run build_gold_set.py first.")
    blind = load_json(items_path)
    all_ids = {it["item_id"] for it in (blind.get("items") or [])}
    all_ids |= {it["item_id"] for it in (blind.get("calibration") or [])}
    ann_dir = pack / "annotations"
    from app.services.gold_set.phase2 import run_phase2

    assignment_path = pack / "annotator_assignments.json"
    assignment = load_json(assignment_path) if assignment_path.is_file() else None
    rows_by: dict[str, list[dict]] = {}
    if assignment:
        missing = [
            name for name in ANNOTATORS if not (ann_dir / f"{name}.csv").is_file()
        ]
        if len(missing) == len(ANNOTATORS):
            raise GoldSetAnnotationError(
                f"Annotation file missing: {ann_dir / 'annotator_A.csv'}. "
                "Place real annotator files under experiments/gold_set/annotations/ "
                "and do not fabricate labels."
            )
        if missing:
            raise GoldSetAnnotationError(
                "Four-annotator gold set needs all of annotator_A.csv … annotator_D.csv. "
                "Missing: " + ", ".join(f"{name}.csv" for name in missing)
            )
        for name in ANNOTATORS:
            expected = set(assignment["annotators"][name]["item_ids"])
            rows_by[name] = load_and_validate_annotations(ann_dir / f"{name}.csv", expected)
    else:
        a_path = ann_dir / "annotator_A.csv"
        a_rows = load_and_validate_annotations(a_path, all_ids)
        rows_by["annotator_A"] = a_rows
        b_path = ann_dir / "annotator_B.csv"
        if b_path.is_file():
            rows_by["annotator_B"] = load_and_validate_annotations(b_path, all_ids)

    adj_path = ann_dir / "adjudicated.csv"
    adj_rows = _read_table(adj_path) if adj_path.is_file() else None
    return run_phase2(root, rows_by, adj_rows, assignment=assignment)
