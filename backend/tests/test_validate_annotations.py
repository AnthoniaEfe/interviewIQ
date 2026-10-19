"""Pre-flight gold-set annotation validator. Synthetic CSVs stay in tmp_path."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "experiments" / "gold_set" / "validate_annotations.py"
ANSWER_KEY = ROOT / "experiments" / "gold_set" / "answer_key_do_not_open_before_annotating.json"


def _load():
    spec = importlib.util.spec_from_file_location("validate_annotations", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    items = [{"item_id": f"g{i:03d}", "requirement_text": "x", "cv_text": "y"} for i in range(1, 11)]
    calibration = [
        {"item_id": f"c{i:03d}", "requirement_text": "x", "cv_text": "y"} for i in range(1, 3)
    ]
    (pack / "items_blind.json").write_text(
        json.dumps({"items": items, "calibration": calibration}),
        encoding="utf-8",
    )
    return pack


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", "label", "cannot_judge", "snippet", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _complete_rows() -> list[dict]:
    rows = []
    for i in range(1, 3):
        rows.append(
            {
                "item_id": f"c{i:03d}",
                "label": "1",
                "cannot_judge": "false",
                "snippet": "evidence",
                "notes": "",
            }
        )
    for i in range(1, 11):
        rows.append(
            {
                "item_id": f"g{i:03d}",
                "label": "0",
                "cannot_judge": "false",
                "snippet": "",
                "notes": "",
            }
        )
    return rows


def test_ready_file_passes(tmp_path: Path):
    module = _load()
    pack = _pack(tmp_path)
    csv_path = tmp_path / "annotator_A.csv"
    _write(csv_path, _complete_rows())
    report = module.validate_annotation_file(csv_path, pack)
    assert report["completion"] == 1.0
    assert module.main([str(csv_path), "--pack", str(pack)]) == 0


def test_cannot_judge_counts_and_bad_label_fails(tmp_path: Path):
    module = _load()
    pack = _pack(tmp_path)
    rows = _complete_rows()
    rows[-1]["label"] = ""
    rows[-1]["cannot_judge"] = "true"
    csv_path = tmp_path / "annotator_B.csv"
    _write(csv_path, rows)
    report = module.validate_annotation_file(csv_path, pack)
    assert report["n_complete"] == 12

    rows[-1]["label"] = "9"
    rows[-1]["cannot_judge"] = "false"
    _write(csv_path, rows)
    with pytest.raises(module.AnnotationValidationError, match="label"):
        module.validate_annotation_file(csv_path, pack)


def test_completion_threshold_duplicates_unknown_and_calibration(tmp_path: Path):
    module = _load()
    pack = _pack(tmp_path)
    csv_path = tmp_path / "short.csv"
    rows = _complete_rows()[:4]
    _write(csv_path, rows)
    with pytest.raises(module.AnnotationValidationError, match="Completion"):
        module.validate_annotation_file(csv_path, pack)

    dup = _complete_rows() + [_complete_rows()[0]]
    _write(csv_path, dup)
    with pytest.raises(module.AnnotationValidationError, match="Duplicate"):
        module.validate_annotation_file(csv_path, pack)

    unknown = _complete_rows()
    unknown[-1]["item_id"] = "nope"
    _write(csv_path, unknown)
    with pytest.raises(module.AnnotationValidationError, match="not in the blind pack"):
        module.validate_annotation_file(csv_path, pack)

    missing_cal = [row for row in _complete_rows() if row["item_id"] != "c001"]
    _write(csv_path, missing_cal)
    with pytest.raises(module.AnnotationValidationError, match="Calibration"):
        module.validate_annotation_file(csv_path, pack)


def test_missing_file_columns_and_no_side_effects(tmp_path: Path):
    module = _load()
    pack = _pack(tmp_path)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    readme = annotations / "README.txt"
    readme.write_text("untouched", encoding="utf-8")
    missing = tmp_path / "nope.csv"
    with pytest.raises(module.AnnotationValidationError, match="does not exist"):
        module.validate_annotation_file(missing, pack)
    assert module.main([str(missing), "--pack", str(pack)]) == 1

    bad = tmp_path / "bad.csv"
    bad.write_text("item_id,label\ng001,1\n", encoding="utf-8")
    with pytest.raises(module.AnnotationValidationError, match="missing columns"):
        module.validate_annotation_file(bad, pack)

    assert readme.read_text(encoding="utf-8") == "untouched"
    assert list(annotations.iterdir()) == [readme]
    ok = tmp_path / "annotator_A.csv"
    _write(ok, _complete_rows())
    key_before = ANSWER_KEY.stat().st_mtime_ns
    assert module.main([str(ok), "--pack", str(pack)]) == 0
    assert key_before == ANSWER_KEY.stat().st_mtime_ns
