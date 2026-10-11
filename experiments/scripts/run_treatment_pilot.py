"""Run the InterviewIQ treatment pilot end to end.

Usage:
    PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --dry-run
    PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --smoke --confirm-spend --max-cost 4.00
    PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --confirm-spend --max-cost 4.00
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from app.services.treatment_pilot.constants import DEFAULT_MAX_COST_USD  # noqa: E402
from app.services.treatment_pilot.pipeline import run_treatment_pilot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="InterviewIQ treatment pilot (primary-30)")
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "all",
            "dry_run",
            "regression",
            "baseline_profile_aware",
            "treatment_pipeline",
            "selection_ablation",
            "pilot_compare",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--smoke", action="store_true", help="2-pair smoke test")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--source-experiment", default=None)
    parser.add_argument("--gap-experiment", default=None)
    parser.add_argument("--selection", default="competency_weighted_mmr")
    parser.add_argument("--b1-experiment", default=None)
    parser.add_argument("--treatment-experiment", default=None)
    parser.add_argument("--ablation-experiment", default=None)
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip gap/B0 regression (not recommended).",
    )
    args = parser.parse_args()
    stage = "dry_run" if args.dry_run else args.stage
    result = asyncio.run(
        run_treatment_pilot(
            stage=stage,
            dry_run=args.dry_run or stage == "dry_run",
            confirm_spend=args.confirm_spend,
            max_cost=args.max_cost,
            resume=args.resume,
            replicates=args.replicates,
            smoke=args.smoke,
            source_experiment=args.source_experiment,
            gap_experiment=args.gap_experiment,
            selection=args.selection,
            b1_experiment=args.b1_experiment,
            treatment_experiment=args.treatment_experiment,
            ablation_experiment=args.ablation_experiment,
            skip_regression=args.skip_regression,
        )
    )
    print(result.get("status"), result.get("command") or "")


if __name__ == "__main__":
    main()
