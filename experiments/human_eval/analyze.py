"""Run the pre-registered main analysis. Refuses pilot, demo, and incomplete exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.human_eval.analyze import HumanEvalAnalysisError, analyze_ratings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--condition-key", type=Path, required=True)
    parser.add_argument("--mode", default="main")
    parser.add_argument("--n-items", type=int, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "human_eval_results.md")
    args = parser.parse_args()
    ratings = json.loads(args.ratings.read_text(encoding="utf-8"))
    if isinstance(ratings, dict):
        ratings = ratings.get("ratings") or []
    key = json.loads(args.condition_key.read_text(encoding="utf-8"))
    try:
        result = analyze_ratings(ratings, key, mode=args.mode, n_items=args.n_items)
    except HumanEvalAnalysisError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(
        "# Human evaluation results\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
