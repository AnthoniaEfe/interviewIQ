"""Build a blinded rating study from stored condition outputs.

Refuses to invent questions. Writes the condition key under
experiments/human_eval/keys/ (git-ignored) and a hash-only manifest next to
this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.human_eval.study import STUDY_SEED, build_blinded_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True, help="JSON list of pair records with T and B1 questions")
    parser.add_argument("--raters", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=STUDY_SEED)
    parser.add_argument("--mode", choices=("pilot", "main"), default="main")
    args = parser.parse_args()
    pairs = json.loads(args.inputs.read_text(encoding="utf-8"))
    built = build_blinded_study(pairs, args.raters, seed=args.seed)
    out_dir = ROOT / "experiments" / "human_eval" / "data" / args.mode
    key_dir = ROOT / "experiments" / "human_eval" / "keys"
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study.json").write_text(json.dumps(built["blinded"], indent=2), encoding="utf-8")
    (key_dir / f"condition_key_{args.mode}.json").write_text(
        json.dumps(built["condition_key"], indent=2),
        encoding="utf-8",
    )
    manifest_path = ROOT / "experiments" / "human_eval" / f"manifest_{args.mode}.json"
    manifest_path.write_text(json.dumps(built["manifest"], indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "n_items": built["manifest"]["n_items"]}, indent=2))


if __name__ == "__main__":
    main()
