"""Phase 2 gold-set analysis. Refuses to run without valid annotation files."""

from __future__ import annotations

import csv
from pathlib import Path

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
    expected = {it["item_id"] for it in (blind.get("items") or [])}
    expected |= {it["item_id"] for it in (blind.get("calibration") or [])}
    ann_dir = pack / "annotations"
    a_path = ann_dir / "annotator_A.csv"
    b_path = ann_dir / "annotator_B.csv"
    # Single-annotator fallback: only A is required if B is absent — still refuse
    # if A is missing. B missing is allowed only if A exists (test-retest later).
    from app.services.gold_set.phase2 import run_phase2

    a_rows = load_and_validate_annotations(a_path, expected)
    b_rows = None
    if b_path.is_file():
        b_rows = load_and_validate_annotations(b_path, expected)
    adj_path = ann_dir / "adjudicated.csv"
    adj_rows = None
    if adj_path.is_file():
        adj_rows = _read_table(adj_path)
    return run_phase2(root, a_rows, b_rows, adj_rows)
