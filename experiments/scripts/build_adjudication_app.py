"""Build the two-adjudicator page for the 30 main-set disagreements.

Reads the blind pack and the adjudication worksheet. Does not read the answer key.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.gold_set.adjudication_app import render_adjudication_html  # noqa: E402

PACK = ROOT / "experiments/gold_set"
BLIND = PACK / "items_blind.json"
WORKSHEET = PACK / "round2/adjudication_worksheet.csv"
CALIBRATION = PACK / "annotations/adjudicated_calibration.csv"
OUT = PACK / "round2/adjudication_app.html"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    blind = json.loads(BLIND.read_text(encoding="utf-8"))
    by_id = {row["item_id"]: row for row in list(blind.get("items") or []) + list(blind.get("calibration") or [])}
    cases = []
    for row in _rows(WORKSHEET):
        item_id = row["item_id"].strip()
        item = by_id.get(item_id)
        if item is None:
            raise SystemExit(f"Worksheet item {item_id} is not in the blind pack")
        cases.append(
            {
                "item_id": item_id,
                "requirement_text": item.get("requirement_text") or row.get("requirement_text") or "",
                "cv_text": item.get("cv_text") or "",
                "jd_text": item.get("jd_text") or "",
                "annotator_1": row.get("annotator_1") or "",
                "label_1": row.get("label_1") or "",
                "notes_1": row.get("notes_1") or "",
                "annotator_2": row.get("annotator_2") or "",
                "label_2": row.get("label_2") or "",
                "notes_2": row.get("notes_2") or "",
                "requirement_phrase_in_cv": row.get("requirement_phrase_in_cv") or "",
            }
        )
    if len(cases) != 30:
        raise SystemExit(f"Expected 30 disagreement rows, found {len(cases)}")
    calibration = []
    for row in _rows(CALIBRATION):
        item_id = row["item_id"].strip()
        item = by_id.get(item_id)
        if item is None:
            raise SystemExit(f"Calibration item {item_id} is not in the blind pack")
        calibration.append(
            {
                "item_id": item_id,
                "requirement_text": item.get("requirement_text") or "",
                "label": row.get("label") or "",
            }
        )
    if len(calibration) != 10:
        raise SystemExit(f"Expected 10 calibration labels, found {len(calibration)}")
    html = render_adjudication_html(cases, calibration)
    forbidden = ("strong_match", "weak_match", "partial_match", "fit_label", "system_label")
    leaked = [word for word in forbidden if word in html]
    if leaked:
        raise SystemExit(f"Refusing to write the page; it contains {leaked}")
    OUT.write_text(html, encoding="utf-8")
    print(json.dumps({"path": str(OUT.relative_to(ROOT)), "cases": len(cases), "bytes": OUT.stat().st_size}))


if __name__ == "__main__":
    main()
