"""Run fit-label control end to end.

Usage:
    PYTHONPATH=backend python experiments/scripts/run_fit_label_control.py --dry-run
    PYTHONPATH=backend python experiments/scripts/run_fit_label_control.py --confirm-spend --max-cost 1.00
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

from app.services.fit_label_control.constants import DEFAULT_MAX_COST_USD  # noqa: E402
from app.services.fit_label_control.pipeline import run_fit_label_control  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit-label control (No Fit negative control)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--smoke", action="store_true", help="Sample n=2 No Fit per split")
    args = parser.parse_args()
    asyncio.run(
        run_fit_label_control(
            dry_run=args.dry_run,
            confirm_spend=args.confirm_spend,
            max_cost=args.max_cost,
            smoke=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
