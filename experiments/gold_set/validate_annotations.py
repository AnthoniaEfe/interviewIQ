"""Pre-flight check for a gold-set annotation CSV. Read-only.

Does not write under annotations/ and does not read the answer key.
Exit status is non-zero when the file is not ready for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REQUIRED_COLS = ("item_id", "label", "cannot_judge", "snippet", "notes")
JUDGEMENT_LABELS = {"0", "1", "2"}
CANNOT_JUDGE_LABELS = {"C", "c"}
MIN_COMPLETION = 0.90
ANSWER_KEY_NAME = "answer_key_do_not_open_before_annotating.json"


class AnnotationValidationError(Exception):
    """The CSV is not ready. The message is meant for a human."""


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_expected(pack: Path) -> tuple[list[str], list[str]]:
    blind_path = pack / "items_blind.json"
    if not blind_path.is_file():
        raise AnnotationValidationError(f"Blind item file missing: {blind_path}")
    payload = json.loads(blind_path.read_text(encoding="utf-8"))
    main = [str(it["item_id"]) for it in (payload.get("items") or []) if it.get("item_id")]
    calibration = [
        str(it["item_id"]) for it in (payload.get("calibration") or []) if it.get("item_id")
    ]
    if not main and not calibration:
        raise AnnotationValidationError(f"No item_id values in {blind_path}")
    return main, calibration


def _row_status(row: dict[str, str]) -> str:
    """Return 'complete', 'incomplete', or raise on an illegal label."""
    label = str(row.get("label", "")).strip()
    cannot = _truthy(row.get("cannot_judge", ""))
    if label in CANNOT_JUDGE_LABELS and not cannot:
        raise AnnotationValidationError(
            f"item {row.get('item_id')}: label {label!r} requires cannot_judge true"
        )
    if label in JUDGEMENT_LABELS:
        return "complete"
    if label in CANNOT_JUDGE_LABELS or cannot:
        return "complete"
    if label == "":
        return "incomplete"
    raise AnnotationValidationError(
        f"item {row.get('item_id')}: label {label!r} is not in {{0, 1, 2, C}} "
        "and cannot_judge is not true"
    )


def validate_annotation_file(csv_path: Path, pack: Path) -> dict:
    if csv_path.name == ANSWER_KEY_NAME or ANSWER_KEY_NAME in csv_path.parts:
        raise AnnotationValidationError("Refusing to read the gold-set answer key.")
    if not csv_path.is_file():
        raise AnnotationValidationError(f"Annotation file does not exist: {csv_path}")

    main_ids, calibration_ids = _load_expected(pack)
    expected = list(dict.fromkeys([*calibration_ids, *main_ids]))
    expected_set = set(expected)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnnotationValidationError(f"{csv_path} has no header row")
        missing_cols = [col for col in REQUIRED_COLS if col not in reader.fieldnames]
        if missing_cols:
            raise AnnotationValidationError(
                f"{csv_path} missing columns {missing_cols}. "
                f"Required: {', '.join(REQUIRED_COLS)}"
            )
        rows = list(reader)

    seen: dict[str, int] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    complete_ids: set[str] = set()
    for row in rows:
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            raise AnnotationValidationError("A row is missing item_id")
        seen[item_id] = seen.get(item_id, 0) + 1
        if seen[item_id] == 2:
            duplicates.append(item_id)
        if item_id not in expected_set:
            unknown.append(item_id)
        if _row_status(row) == "complete":
            complete_ids.add(item_id)

    if duplicates:
        raise AnnotationValidationError(f"Duplicate item_id values: {duplicates}")
    if unknown:
        raise AnnotationValidationError(f"item_id values not in the blind pack: {unknown}")

    missing_calibration = [item_id for item_id in calibration_ids if item_id not in seen]
    if missing_calibration:
        raise AnnotationValidationError(
            "Calibration items missing from the file: " + ", ".join(missing_calibration)
        )

    n_expected = len(expected_set)
    n_complete = len(complete_ids & expected_set)
    rate = n_complete / n_expected if n_expected else 0.0
    if rate < MIN_COMPLETION:
        raise AnnotationValidationError(
            f"Completion {rate:.1%} ({n_complete}/{n_expected}) is below {MIN_COMPLETION:.0%}. "
            "A row counts only when label is 0, 1, or 2, or cannot_judge is true "
            "(label C is also accepted)."
        )

    return {
        "path": str(csv_path),
        "n_expected": n_expected,
        "n_complete": n_complete,
        "completion": rate,
        "n_calibration": len(calibration_ids),
        "calibration_present": True,
        "min_completion": MIN_COMPLETION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a gold-set annotation CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Gold-set pack directory containing items_blind.json",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_annotation_file(args.csv_path, args.pack)
    except AnnotationValidationError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 1
    print(
        f"READY: {report['path']} completion {report['completion']:.1%} "
        f"({report['n_complete']}/{report['n_expected']}), "
        f"calibration items present ({report['n_calibration']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
