"""Analyse gold-set round 3 (worked-rule dual coding). Does not overwrite round 2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.gap_sensitivity.io_utils import load_json  # noqa: E402
from app.services.gold_set.analyze import (  # noqa: E402
    GoldSetAnnotationError,
    _read_table,
    load_and_validate_annotations,
)
from app.services.gold_set.phase2 import run_phase2  # noqa: E402

ROUND3 = ROOT / "experiments/gold_set/round3"
ANN = ROUND3 / "annotations"
ASSIGN = ROUND3 / "assignment.json"


def main() -> None:
    try:
        assignment = load_json(ASSIGN)
        rows_by: dict[str, list] = {}
        for name, block in assignment["annotators"].items():
            expected = set(block["main_ids"])
            path = ANN / f"{name}.csv"
            rows = load_and_validate_annotations(path, expected, min_completion=0.90)
            # Keep only main rows for consensus; calibration optional in file.
            rows_by[name] = [r for r in rows if str(r.get("item_id")) in expected]
        adj_path = ANN / "adjudicated.csv"
        adj_rows = _read_table(adj_path) if adj_path.is_file() else None
        payload = run_phase2(
            ROOT,
            rows_by,
            adj_rows,
            assignment=assignment,
            analysis_round="round3_worked_rule_llm_dual_coding",
            results_dirname="gold_set_analysis_round3",
            report_path=ROOT / "docs/gold_set_round3_report.md",
            figures_dirname="round3",
        )
        payload["label_source"] = (
            "LLM annotators P1/P2 under experiments/gold_set/round3/training/"
            "worked_rule_brief.md. Exploratory. Does not replace human round 1–2."
        )
        from app.services.gap_sensitivity.io_utils import write_json

        write_json(
            ROOT / "experiments/results/gold_set_analysis_round3/summary.json",
            payload,
        )
        print(
            {
                "n_usable": payload.get("n_usable_consensus"),
                "disposition": payload.get("consensus_disposition"),
                "iaa_qwk": (payload.get("agreement_main") or {}).get("kappa_quadratic"),
                "g1_interpreted": (payload.get("g1_interpretation") or {}).get("interpreted"),
                "g1_supported": ((payload.get("g1") or {}).get("pooled") or {}).get("supported"),
            }
        )
    except GoldSetAnnotationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
