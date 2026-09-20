"""Reproducible experiment runner.

Usage:
    python -m experiments.run --config experiments/configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="InterviewIQ experiment runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))

    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")
    out_dir = Path("experiments/results") / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": experiment_id,
        "status": "not_yet_evaluated",
        "note": (
            "Runner scaffold only. Baseline vs InterviewIQ generation "
            "will be executed here once LLM extraction is wired."
        ),
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
