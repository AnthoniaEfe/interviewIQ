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
from app.schemas.domain import (  # noqa: E402
    CandidateProfile,
    GenerationMode,
    JobDescription,
    LLMUsage,
)
from app.services.embeddings.service import build_embedding_service  # noqa: E402
from app.services.extraction.profiles import (  # noqa: E402
    CV_PROMPT,
    JD_PROMPT,
    extract_candidate,
    extract_job,
)
from app.services.gap_analysis.matrix import build_gap_matrix  # noqa: E402
from app.services.gap_sensitivity.analyze import run_gap_sensitivity  # noqa: E402
from app.services.fit_label_control.pipeline import run_fit_label_control  # noqa: E402
from app.services.llm.factory import build_router  # noqa: E402
from app.services.llm.provider import LLMResult  # noqa: E402
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


def _usage_payload(usage: LLMResult, task_type: str) -> dict:
    return LLMUsage(
        provider=usage.provider,
        model=usage.model,
        task_type=task_type,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        estimated_cost=usage.estimated_cost,
        latency_ms=usage.latency_ms,
        retry_count=usage.retry_count,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
    ).model_dump()


async def run_extraction_combo(
    *,
    router,
    settings: Settings,
    combo: dict,
    experiment_id: str,
    out_dir: Path,
) -> dict:
    cv_path: Path = combo["cv_path"]
    jd_path: Path = combo["jd_path"]
    seniority = combo["seniority"]
    request_id = f"{combo['profile_id']}__{combo['jd_id']}__{seniority}"
    candidate, cv_usage = await extract_candidate(
        router=router,
        settings=settings,
        cv_text=cv_path.read_text(encoding="utf-8"),
        experiment_id=experiment_id,
        request_id=f"{request_id}:cv",
    )
    job, jd_usage = await extract_job(
        router=router,
        settings=settings,
        jd_text=jd_path.read_text(encoding="utf-8"),
        experiment_id=experiment_id,
        request_id=f"{request_id}:jd",
    )
    record = {
        "experiment_id": experiment_id,
        "stage": "extraction",
        "profile_id": combo["profile_id"],
        "jd_id": combo["jd_id"],
        "pair_id": combo.get("pair_id"),
        "fit_label": combo.get("fit_label"),
        "dataset": "cnamuangtoun/resume-job-description-fit"
        if combo.get("fit_label") is not None
        else None,
        "seniority": seniority,
        "provider": cv_usage.provider,
        "model": cv_usage.model,
        "task_types": [
            LLMTask.CV_EXTRACTION.value,
            LLMTask.JD_EXTRACTION.value,
        ],
        "prompt_versions": {
            "candidate_profile": CV_PROMPT.version,
            "job_description": JD_PROMPT.version,
        },
        "candidate": candidate.model_dump(),
        "job": job.model_dump(),
        "cv_usage": _usage_payload(cv_usage, LLMTask.CV_EXTRACTION.value),
        "jd_usage": _usage_payload(jd_usage, LLMTask.JD_EXTRACTION.value),
        "fallback_used": cv_usage.fallback_used or jd_usage.fallback_used,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = out_dir / f"{request_id}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def latest_extraction_dir() -> Path:
    results = ROOT / "experiments" / "results"
    if not results.exists():
        raise SystemExit("No experiments/results directory. Run --stage extraction first.")
    candidates: list[Path] = []
    for path in results.iterdir():
        summary_path = path / "summary.json"
        extraction_dir = path / "extraction"
        if not summary_path.is_file() or not extraction_dir.is_dir():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("stage") == "extraction" and summary.get("status") == "completed":
            candidates.append(path)
    if not candidates:
        raise SystemExit("No completed extraction experiment found. Run --stage extraction first.")
    return sorted(candidates)[-1]


def load_extraction_records(source_dir: Path) -> list[dict]:
    extraction_dir = source_dir / "extraction"
    records = []
    for path in sorted(extraction_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit(f"No extraction artifacts in {extraction_dir}")
    return records


def run_gap_combo(
    *,
    settings: Settings,
    embeddings,
    record: dict,
    experiment_id: str,
    source_experiment: str,
    out_dir: Path,
) -> dict:
    candidate = CandidateProfile.model_validate(record["candidate"])
    job = JobDescription.model_validate(record["job"])
    started = datetime.now(timezone.utc)
    matrix = build_gap_matrix(candidate, job, settings, embeddings)
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    three_category = {
        "strength_matches": matrix.strong_matches,
        "partial_matches": matrix.partial_matches,
        "competency_gaps": matrix.weak_matches + matrix.missing,
    }
    result = {
        "experiment_id": experiment_id,
        "stage": "gap_analysis",
        "source_experiment": source_experiment,
        "profile_id": record.get("profile_id"),
        "jd_id": record.get("jd_id"),
        "pair_id": record.get("pair_id"),
        "fit_label": record.get("fit_label"),
        "dataset": record.get("dataset"),
        "seniority": record.get("seniority"),
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "signals": {
            "tfidf": settings.enable_tfidf,
            "bm25": settings.enable_bm25,
            "semantic": settings.enable_semantic,
        },
        "gap_matrix": matrix.model_dump(),
        "three_category": three_category,
        "counts": {
            "requirements": len(matrix.gaps),
            "strength_matches": len(three_category["strength_matches"]),
            "partial_matches": len(three_category["partial_matches"]),
            "competency_gaps": len(three_category["competency_gaps"]),
        },
        "latency_ms": round(elapsed_ms, 2),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    request_id = (
        f"{record.get('profile_id')}__{record.get('jd_id')}__{record.get('seniority')}"
    )
    (out_dir / f"{request_id}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


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


async def async_main(
    config_path: Path,
    *,
    smoke: bool,
    stage: str = "baseline",
    source_experiment: str | None = None,
    dry_run: bool = False,
    confirm_spend: bool = False,
    max_cost: float | None = None,
    resume: bool = True,
    replicates: int = 1,
    gap_experiment: str | None = None,
    selection: str = "competency_weighted_mmr",
    b1_experiment: str | None = None,
    treatment_experiment: str | None = None,
    ablation_experiment: str | None = None,
) -> Path:
    config = load_config(config_path)
    get_settings.cache_clear()
    settings = Settings()

    n_questions = int(config.get("final_question_count", settings.final_question_count))
    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "experiments" / "results" / experiment_id

    if stage in {
        "treatment_pipeline",
        "baseline_profile_aware",
        "selection_ablation",
        "pilot_compare",
        "treatment_regression",
    }:
        from app.services.treatment_pilot.constants import DEFAULT_MAX_COST_USD
        from app.services.treatment_pilot.pipeline import run_treatment_pilot

        mapped = "regression" if stage == "treatment_regression" else stage
        result = await run_treatment_pilot(
            stage=mapped,
            dry_run=dry_run,
            confirm_spend=confirm_spend,
            max_cost=DEFAULT_MAX_COST_USD if max_cost is None else max_cost,
            resume=resume,
            replicates=replicates,
            smoke=smoke,
            source_experiment=source_experiment,
            gap_experiment=gap_experiment,
            selection=selection,
            b1_experiment=b1_experiment,
            treatment_experiment=treatment_experiment,
            ablation_experiment=ablation_experiment,
        )
        for key in ("b1_dir", "treatment_dir", "ablation_dir"):
            if result.get(key):
                return Path(result[key])
        return ROOT / "experiments" / "results"

    if stage == "gap_analysis":
        (out_dir / "gap_analysis").mkdir(parents=True, exist_ok=True)
        return _run_gap_analysis_stage(
            settings=settings,
            config=config,
            config_path=config_path,
            experiment_id=experiment_id,
            out_dir=out_dir,
            smoke=smoke,
            source_experiment=source_experiment,
        )

    if stage == "fit_label_control":
        from app.services.fit_label_control.constants import DEFAULT_MAX_COST_USD

        result = await run_fit_label_control(
            dry_run=dry_run,
            confirm_spend=confirm_spend,
            max_cost=DEFAULT_MAX_COST_USD if max_cost is None else max_cost,
            smoke=smoke,
        )
        out = result.get("out_dir")
        return Path(out) if out else ROOT / "experiments" / "results"

    if stage == "gold_set_build":
        from app.services.gold_set.build import build_gold_set

        result = build_gold_set(ROOT)
        print(result)
        return ROOT / "experiments" / "gold_set"

    if stage == "gold_set_analyze":
        from app.services.gold_set.analyze import run_gold_set_analyze

        run_gold_set_analyze(ROOT)
        return ROOT / "experiments" / "results" / "gold_set_analysis"

    if stage == "classifier_eval":
        from app.services.classifier_eval.pipeline import run_classifier_eval

        result = run_classifier_eval(root=ROOT, settings=settings, smoke=smoke)
        out = result.get("out_dir")
        return Path(out) if out else ROOT / "experiments" / "results"

    if stage == "gap_sensitivity":
        if not source_experiment:
            raise SystemExit(
                "--source-experiment is required for --stage gap_sensitivity "
                "(extraction experiment id)."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        return run_gap_sensitivity(
            settings=settings,
            config=config,
            config_path=config_path,
            experiment_id=experiment_id,
            out_dir=out_dir,
            source_experiment=source_experiment,
            smoke=smoke,
        )

    router = build_router(settings)
    combinations = load_combinations(config)
    if smoke:
        combinations = combinations[:1]
    if stage == "extraction":
        (out_dir / "extraction").mkdir(parents=True, exist_ok=True)
        return await _run_extraction_stage(
            settings=settings,
            router=router,
            config=config,
            config_path=config_path,
            combinations=combinations,
            experiment_id=experiment_id,
            out_dir=out_dir,
        )

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


async def _run_extraction_stage(
    *,
    settings: Settings,
    router,
    config: dict,
    config_path: Path,
    combinations: list[dict],
    experiment_id: str,
    out_dir: Path,
) -> Path:
    results: list[dict] = []
    errors: list[dict] = []
    for combo in combinations:
        label = f"{combo['profile_id']} x {combo['jd_id']} x {combo['seniority']}"
        if combo.get("fit_label"):
            label = f"{label} [{combo['fit_label']}]"
        print(f"[extraction] {label} via {settings.llm_fast_model}")
        try:
            record = await run_extraction_combo(
                router=router,
                settings=settings,
                combo=combo,
                experiment_id=experiment_id,
                out_dir=out_dir / "extraction",
            )
            cv_usage = record["cv_usage"]
            jd_usage = record["jd_usage"]
            print(
                f"  cv_skills={len(record['candidate']['technical_skills'])} "
                f"jd_required={len(record['job']['required_skills'])} "
                f"latency_ms={cv_usage['latency_ms'] + jd_usage['latency_ms']} "
                f"cost_usd={cv_usage['estimated_cost'] + jd_usage['estimated_cost']}"
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

    total_cost = sum(
        r["cv_usage"]["estimated_cost"] + r["jd_usage"]["estimated_cost"] for r in results
    )
    total_latency = sum(
        r["cv_usage"]["latency_ms"] + r["jd_usage"]["latency_ms"] for r in results
    )
    summary = {
        "experiment_id": experiment_id,
        "status": "completed" if results and not errors else (
            "failed" if not results else "partial"
        ),
        "stage": "extraction",
        "primary_dataset": config.get("primary_dataset"),
        "provider": settings.llm_provider,
        "model": settings.llm_fast_model,
        "prompt_versions": {
            "candidate_profile": CV_PROMPT.version,
            "job_description": JD_PROMPT.version,
        },
        "combinations_attempted": len(combinations),
        "combinations_ok": len(results),
        "combinations_failed": len(errors),
        "total_estimated_cost_usd": round(total_cost, 8),
        "average_latency_ms": round(total_latency / len(results), 2) if results else None,
        "human_evaluation": "not_yet_evaluated",
        "errors": errors,
        "result_files": [
            f"extraction/{r['profile_id']}__{r['jd_id']}__{r['seniority']}.json" for r in results
        ],
        "config_path": str(config_path),
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {out_dir / 'summary.json'}")
    return out_dir


def _run_gap_analysis_stage(
    *,
    settings: Settings,
    config: dict,
    config_path: Path,
    experiment_id: str,
    out_dir: Path,
    smoke: bool,
    source_experiment: str | None,
) -> Path:
    source_dir = (
        ROOT / "experiments" / "results" / source_experiment
        if source_experiment
        else latest_extraction_dir()
    )
    if not (source_dir / "extraction").is_dir():
        raise SystemExit(f"Extraction artifacts not found in {source_dir}")
    records = load_extraction_records(source_dir)
    if smoke:
        records = records[:1]

    print(
        f"[gap_analysis] loading embeddings {settings.embedding_model} "
        f"device={settings.embedding_device}"
    )
    embeddings = build_embedding_service(
        settings.embedding_model,
        settings.embedding_device,
    )
    results: list[dict] = []
    errors: list[dict] = []
    for record in records:
        label = f"{record.get('profile_id')} x {record.get('jd_id')}"
        if record.get("fit_label"):
            label = f"{label} [{record['fit_label']}]"
        print(f"[gap_analysis] {label}")
        try:
            result = run_gap_combo(
                settings=settings,
                embeddings=embeddings,
                record=record,
                experiment_id=experiment_id,
                source_experiment=source_dir.name,
                out_dir=out_dir / "gap_analysis",
            )
            counts = result["counts"]
            print(
                f"  requirements={counts['requirements']} "
                f"strength={counts['strength_matches']} "
                f"partial={counts['partial_matches']} "
                f"gaps={counts['competency_gaps']} "
                f"latency_ms={result['latency_ms']}"
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001 — recorded in experiment summary
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            errors.append(
                {
                    "profile_id": record.get("profile_id"),
                    "jd_id": record.get("jd_id"),
                    "fit_label": record.get("fit_label"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    total_strength = sum(r["counts"]["strength_matches"] for r in results)
    total_partial = sum(r["counts"]["partial_matches"] for r in results)
    total_gaps = sum(r["counts"]["competency_gaps"] for r in results)
    total_req = sum(r["counts"]["requirements"] for r in results)
    total_latency = sum(r["latency_ms"] for r in results)
    summary = {
        "experiment_id": experiment_id,
        "status": "completed" if results and not errors else (
            "failed" if not results else "partial"
        ),
        "stage": "gap_analysis",
        "source_experiment": source_dir.name,
        "primary_dataset": config.get("primary_dataset"),
        "embedding_model": settings.embedding_model,
        "embedding_device": getattr(embeddings, "device", settings.embedding_device),
        "signals": {
            "tfidf": settings.enable_tfidf,
            "bm25": settings.enable_bm25,
            "semantic": settings.enable_semantic,
        },
        "combinations_attempted": len(records),
        "combinations_ok": len(results),
        "combinations_failed": len(errors),
        "requirements_scored": total_req,
        "three_category_totals": {
            "strength_matches": total_strength,
            "partial_matches": total_partial,
            "competency_gaps": total_gaps,
        },
        "average_latency_ms": round(total_latency / len(results), 2) if results else None,
        "human_evaluation": "not_yet_evaluated",
        "errors": errors,
        "result_files": [
            f"gap_analysis/{r['profile_id']}__{r['jd_id']}__{r['seniority']}.json"
            for r in results
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
    parser.add_argument(
        "--stage",
        choices=(
            "extraction",
            "gap_analysis",
            "baseline",
            "gap_sensitivity",
            "fit_label_control",
            "gold_set_build",
            "gold_set_analyze",
            "classifier_eval",
            "treatment_pipeline",
            "baseline_profile_aware",
            "selection_ablation",
            "pilot_compare",
            "treatment_regression",
        ),
        default="baseline",
        help="Pipeline stage to run. Default remains baseline generation.",
    )
    parser.add_argument(
        "--source-experiment",
        default=None,
        help=(
            "Extraction experiment id for gap_analysis, gap_sensitivity, or treatment stages. "
            "Defaults to latest completed extraction for gap_analysis."
        ),
    )
    parser.add_argument(
        "--gap-experiment",
        default=None,
        help="Gap-analysis experiment id for treatment stages (default: pinned primary-30).",
    )
    parser.add_argument(
        "--selection",
        default="competency_weighted_mmr",
        help="Selection strategy for --stage treatment_pipeline.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse disk-cached LLM outputs for treatment stages (default: true).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Generations per condition per pair for treatment stages (default: 1).",
    )
    parser.add_argument(
        "--b1-experiment",
        default=None,
        help="Existing B1 result dir/id for selection_ablation/pilot_compare.",
    )
    parser.add_argument(
        "--treatment-experiment",
        default=None,
        help="Existing treatment result dir/id (with pool/) for ablations/compare.",
    )
    parser.add_argument(
        "--ablation-experiment",
        default=None,
        help="Existing selection_ablation result dir/id for pilot_compare.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For fit_label_control / treatment stages: print pairs and estimated cost, then exit.",
    )
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="Allow paid LLM calls (fit_label_control / treatment stages).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Abort if running cost would exceed this USD cap.",
    )
    args = parser.parse_args()
    asyncio.run(
        async_main(
            Path(args.config).resolve(),
            smoke=args.smoke,
            stage=args.stage,
            source_experiment=args.source_experiment,
            dry_run=args.dry_run,
            confirm_spend=args.confirm_spend,
            max_cost=args.max_cost,
            resume=args.resume,
            replicates=args.replicates,
            gap_experiment=args.gap_experiment,
            selection=args.selection,
            b1_experiment=args.b1_experiment,
            treatment_experiment=args.treatment_experiment,
            ablation_experiment=args.ablation_experiment,
        )
    )


if __name__ == "__main__":
    main()
