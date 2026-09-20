"""Reproducible experiment runner.

Usage:
    PYTHONPATH=backend python -m experiments.run --config experiments/configs/primary_hf.yaml
    PYTHONPATH=backend python -m experiments.run --config experiments/configs/baseline.yaml --smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

load_dotenv(ROOT / ".env", override=True)

from app.config.settings import Settings, get_settings  # noqa: E402
from app.schemas.domain import GenerationMode, LLMUsage  # noqa: E402
from app.services.llm.factory import build_router  # noqa: E402
from app.services.llm.tasks import LLMTask  # noqa: E402
from app.services.question_generation.baseline import PROMPT, generate_baseline  # noqa: E402


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _text_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.txt") if p.is_file())


def load_combinations(config: dict) -> list[dict]:
    """Return list of {profile_id, jd_id, seniority, cv_path, jd_path, label?}."""
    data = config.get("data", {})
    default_seniority = data.get("default_seniority", "mid")

    manifest_rel = data.get("manifest")
    if manifest_rel:
        manifest_path = ROOT / manifest_rel
        if not manifest_path.exists():
            raise SystemExit(
                f"Manifest not found: {manifest_path}. "
                "Run: PYTHONPATH=backend python experiments/scripts/train_fit_classifier.py"
            )
        pairs_dir = ROOT / data.get(
            "pairs_dir",
            "experiments/datasets/resume_job_fit/experiment_pairs",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        combos = []
        for item in manifest:
            combos.append(
                {
                    "profile_id": item["profile_id"],
                    "jd_id": item["jd_id"],
                    "seniority": item.get("seniority", default_seniority),
                    "cv_path": pairs_dir / item["profile_path"],
                    "jd_path": pairs_dir / item["job_path"],
                    "fit_label": item.get("label"),
                    "pair_id": item.get("pair_id"),
                }
            )
        return combos

    profiles_dir = ROOT / data.get("profiles_dir", "experiments/datasets/profiles")
    jobs_dir = ROOT / data.get("jobs_dir", "experiments/datasets/jobs")
    seniorities = data.get("seniorities") or ["junior", "mid", "senior"]
    profiles = _text_files(profiles_dir)
    jobs = _text_files(jobs_dir)
    if not profiles:
        raise SystemExit(f"No profile .txt files in {profiles_dir}")
    if not jobs:
        raise SystemExit(f"No job .txt files in {jobs_dir}")
    return [
        {
            "profile_id": cv.stem,
            "jd_id": jd.stem,
            "seniority": seniority,
            "cv_path": cv,
            "jd_path": jd,
            "fit_label": None,
            "pair_id": None,
        }
        for cv, jd, seniority in product(profiles, jobs, seniorities)
    ]


async def run_baseline_combo(
    *,
    router,
    settings: Settings,
    combo: dict,
    n_questions: int,
    experiment_id: str,
    out_dir: Path,
) -> dict:
    cv_path: Path = combo["cv_path"]
    jd_path: Path = combo["jd_path"]
    seniority = combo["seniority"]
    request_id = f"{combo['profile_id']}__{combo['jd_id']}__{seniority}"
    parsed, usage = await generate_baseline(
        router=router,
        settings=settings,
        cv_text=cv_path.read_text(encoding="utf-8"),
        jd_text=jd_path.read_text(encoding="utf-8"),
        seniority=seniority,
        n_questions=n_questions,
        experiment_id=experiment_id,
        request_id=request_id,
    )
    record = {
        "experiment_id": experiment_id,
        "generation_mode": GenerationMode.BASELINE.value,
        "profile_id": combo["profile_id"],
        "jd_id": combo["jd_id"],
        "pair_id": combo.get("pair_id"),
        "fit_label": combo.get("fit_label"),
        "dataset": "cnamuangtoun/resume-job-description-fit"
        if combo.get("fit_label") is not None
        else None,
        "seniority": seniority,
        "provider": usage.provider,
        "model": usage.model,
        "task_type": LLMTask.QUESTION_GENERATION.value,
        "prompt_version": PROMPT.version,
        "final_question_count_requested": n_questions,
        "questions": [item.model_dump() for item in parsed.questions],
        "usage": LLMUsage(
            provider=usage.provider,
            model=usage.model,
            task_type=LLMTask.QUESTION_GENERATION.value,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            estimated_cost=usage.estimated_cost,
            latency_ms=usage.latency_ms,
            retry_count=usage.retry_count,
            fallback_used=usage.fallback_used,
            fallback_reason=usage.fallback_reason,
        ).model_dump(),
        "fallback_used": usage.fallback_used,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = out_dir / f"{request_id}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


async def async_main(config_path: Path, *, smoke: bool) -> Path:
    config = load_config(config_path)
    get_settings.cache_clear()
    settings = Settings()
    router = build_router(settings)

    n_questions = int(config.get("final_question_count", settings.final_question_count))
    combinations = load_combinations(config)
    if smoke:
        combinations = combinations[:1]

    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "experiments" / "results" / experiment_id
    (out_dir / "baseline").mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    errors: list[dict] = []
    modes = config.get("generation_modes") or ["baseline"]
    if "baseline" not in modes:
        raise SystemExit("This runner currently executes baseline mode only.")

    for combo in combinations:
        label = f"{combo['profile_id']} x {combo['jd_id']} x {combo['seniority']}"
        if combo.get("fit_label"):
            label = f"{label} [{combo['fit_label']}]"
        print(f"[baseline] {label} via {settings.llm_primary_model}")
        try:
            record = await run_baseline_combo(
                router=router,
                settings=settings,
                combo=combo,
                n_questions=n_questions,
                experiment_id=experiment_id,
                out_dir=out_dir / "baseline",
            )
            usage = record["usage"]
            print(
                f"  questions={len(record['questions'])} "
                f"latency_ms={usage['latency_ms']} "
                f"tokens_in={usage['input_tokens']} "
                f"tokens_out={usage['output_tokens']} "
                f"cost_usd={usage['estimated_cost']}"
            )
            results.append(record)
        except Exception as exc:  # noqa: BLE001 — recorded in experiment summary
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            errors.append(
                {
                    "profile_id": combo["profile_id"],
                    "jd_id": combo["jd_id"],
                    "seniority": combo["seniority"],
                    "fit_label": combo.get("fit_label"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    total_cost = sum(r["usage"]["estimated_cost"] for r in results)
    total_latency = sum(r["usage"]["latency_ms"] for r in results)
    summary = {
        "experiment_id": experiment_id,
        "status": "completed" if results and not errors else (
            "failed" if not results else "partial"
        ),
        "generation_mode": GenerationMode.BASELINE.value,
        "primary_dataset": config.get("primary_dataset"),
        "provider": settings.llm_provider,
        "model": settings.llm_primary_model,
        "prompt_version": PROMPT.version,
        "final_question_count_requested": n_questions,
        "combinations_attempted": len(combinations),
        "combinations_ok": len(results),
        "combinations_failed": len(errors),
        "total_estimated_cost_usd": round(total_cost, 8),
        "average_latency_ms": round(total_latency / len(results), 2) if results else None,
        "human_evaluation": "not_yet_evaluated",
        "errors": errors,
        "result_files": [
            f"baseline/{r['profile_id']}__{r['jd_id']}__{r['seniority']}.json" for r in results
        ],
        "config_path": str(config_path),
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {out_dir / 'summary.json'}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="InterviewIQ experiment runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="Run one combination only")
    args = parser.parse_args()
    asyncio.run(async_main(Path(args.config).resolve(), smoke=args.smoke))


if __name__ == "__main__":
    main()
