"""Run gap-sensitivity on primary-30 and holdout-30, then combine outputs.

Usage:
    PYTHONPATH=backend python experiments/scripts/run_gap_sensitivity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

load_dotenv(ROOT / ".env", override=True)

import yaml  # noqa: E402

from app.config.settings import Settings, get_settings  # noqa: E402
from app.services.embeddings.service import build_embedding_service  # noqa: E402
from app.services.gap_sensitivity.analyze import run_gap_sensitivity  # noqa: E402
from app.services.gap_sensitivity.combine import combine_results  # noqa: E402
from app.services.gap_sensitivity.io_utils import utc_experiment_id  # noqa: E402
from app.services.gap_sensitivity.report import write_report  # noqa: E402


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gap-sensitivity on both splits")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--primary-source",
        default="exp_20260927T205741Z",
        help="Primary-30 extraction experiment id",
    )
    parser.add_argument(
        "--holdout-source",
        default="exp_20260927T221621Z",
        help="Holdout-30 extraction experiment id",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    settings = Settings()
    print(
        f"[gap_sensitivity] loading embeddings {settings.embedding_model} "
        f"device={settings.embedding_device}"
    )
    embeddings = build_embedding_service(
        settings.embedding_model,
        settings.embedding_device,
    )

    splits = [
        (
            "primary",
            ROOT / "experiments" / "configs" / "primary_hf.yaml",
            args.primary_source,
        ),
        (
            "holdout",
            ROOT / "experiments" / "configs" / "primary_hf_holdout30.yaml",
            args.holdout_source,
        ),
    ]
    out_dirs: dict[str, Path] = {}
    for name, config_path, source in splits:
        config = _load_config(config_path)
        experiment_id = utc_experiment_id()
        out_dir = ROOT / "experiments" / "results" / experiment_id
        print(f"\n=== {name} source={source} -> {experiment_id} ===")
        run_gap_sensitivity(
            settings=settings,
            config=config,
            config_path=config_path,
            experiment_id=experiment_id,
            out_dir=out_dir,
            source_experiment=source,
            smoke=args.smoke,
            embeddings=embeddings,
        )
        out_dirs[name] = out_dir

    combined_dir = ROOT / "experiments" / "results" / "gap_sensitivity_combined"
    figures_dir = ROOT / "docs" / "figures"
    print(f"\n=== combine -> {combined_dir} ===")
    combine_results(
        primary_dir=out_dirs["primary"],
        holdout_dir=out_dirs["holdout"],
        combined_dir=combined_dir,
        figures_dir=figures_dir,
    )
    report_path = ROOT / "docs" / "gap_sensitivity_report.md"
    write_report(
        report_path=report_path,
        primary_dir=out_dirs["primary"],
        holdout_dir=out_dirs["holdout"],
        combined_dir=combined_dir,
        figures_dir=figures_dir,
    )
    print(f"Report -> {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
