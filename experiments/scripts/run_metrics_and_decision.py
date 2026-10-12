"""Run frozen metrics Part A / Part B.

Usage:
    PYTHONPATH=backend python experiments/scripts/run_metrics_and_decision.py --stage metrics_freeze
    PYTHONPATH=backend python experiments/scripts/run_metrics_and_decision.py --stage metrics_pilot
    PYTHONPATH=backend python experiments/scripts/run_metrics_and_decision.py --stage scale_decision
    PYTHONPATH=backend python experiments/scripts/run_metrics_and_decision.py --stage all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from app.config.settings import Settings, get_settings  # noqa: E402
from app.services.eval_metrics.constants import (  # noqa: E402
    DEFAULT_MAX_COST_USD,
    DEFAULT_SCALE_BUDGET_CNY,
)
from app.services.eval_metrics.pipeline import (  # noqa: E402
    run_metrics_freeze,
    run_metrics_pilot,
    run_scale_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen metrics + scale/diagnose decision")
    parser.add_argument(
        "--stage",
        choices=("metrics_freeze", "metrics_pilot", "scale_decision", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--extract-missing", action="store_true")
    parser.add_argument("--with-dimension-labels", action="store_true")
    parser.add_argument("--scale-budget-cny", type=float, default=DEFAULT_SCALE_BUDGET_CNY)
    parser.add_argument("--allow-post-hoc", action="store_true")
    args = parser.parse_args()

    get_settings.cache_clear()
    settings = Settings()

    if args.stage in {"metrics_freeze", "all"}:
        run_metrics_freeze(
            settings=settings,
            dry_run=args.dry_run,
            confirm_spend=args.confirm_spend,
            max_cost=args.max_cost,
            extract_missing=args.extract_missing,
            with_dimension_labels=args.with_dimension_labels,
        )
    if args.stage in {"metrics_pilot", "all"}:
        run_metrics_pilot(settings=settings, allow_post_hoc=args.allow_post_hoc)
    if args.stage in {"scale_decision", "all"}:
        run_scale_decision(
            settings=settings,
            allow_post_hoc=args.allow_post_hoc,
            scale_budget_cny=args.scale_budget_cny,
        )


if __name__ == "__main__":
    main()
