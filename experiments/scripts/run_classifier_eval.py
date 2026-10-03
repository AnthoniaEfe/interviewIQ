"""Run classifier reframing evaluation (local embeddings, no LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config.settings import Settings  # noqa: E402
from app.services.classifier_eval.pipeline import run_classifier_eval  # noqa: E402


def main() -> None:
    smoke = "--smoke" in sys.argv
    payload = run_classifier_eval(root=ROOT, settings=Settings(), smoke=smoke)
    print(json.dumps({k: payload[k] for k in ("experiment_id", "status", "regression_v0", "leakage") if k in payload}, indent=2, default=str))


if __name__ == "__main__":
    main()
