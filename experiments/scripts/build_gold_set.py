"""Build the gold-set annotation pack (no LLM, no analysis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.gold_set.build import build_gold_set  # noqa: E402


INSTRUCTIONS = """
================================================================
GOLD-SET PHASE 1 — human annotation (do not open the answer key)
================================================================

Blind items (safe to load in the annotator):
  experiments/gold_set/items_blind.json
  experiments/gold_set/annotator_app.html
  experiments/gold_set/annotation_template.csv

Guidelines:
  docs/gold_set_guidelines.md
  docs/gold_set_protocol.md

Answer key (DO NOT OPEN until annotation is finished):
  experiments/gold_set/answer_key_do_not_open_before_annotating.json

How to annotate:
  1. Both annotators complete CALIBRATION items (c001…) first, then discuss.
  2. Open annotator_app.html locally (no network). Choose A or B (independent order).
     Keyboard: 0 / 1 / 2 / C. Export CSV.
     Or fill annotation_template.csv.
  3. Place completed files at:
       experiments/gold_set/annotations/annotator_A.csv
       experiments/gold_set/annotations/annotator_B.csv
       experiments/gold_set/annotations/adjudicated.csv   (optional; disagreements)
     Columns: item_id, label, cannot_judge, snippet, notes
     label ∈ {{0,1,2}}; cannot_judge true/false.
  4. If only one annotator: submit annotator_A.csv and, after ≥7 days, a 30-item
     re-annotation (document item ids). Analysis will report test-retest kappa.

Do NOT run gold_set_analyze until those files exist. Do not invent labels.
"""


def main() -> None:
    result = build_gold_set(ROOT)
    print(json.dumps(result, indent=2, default=str))
    print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
