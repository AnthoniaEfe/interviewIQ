"""Analyse gold-set annotations. Refuses if files are missing or invalid."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.gold_set.analyze import GoldSetAnnotationError, run_gold_set_analyze  # noqa: E402


def main() -> None:
    try:
        run_gold_set_analyze(ROOT)
    except GoldSetAnnotationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
