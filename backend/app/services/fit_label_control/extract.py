"""Luna extraction with disk cache, smoke test, and spend guard."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription
from app.services.extraction.profiles import CV_PROMPT, JD_PROMPT, extract_candidate, extract_job
from app.services.fit_label_control.constants import (
    EXTRACTION_MODEL,
    PRIOR_PER_PAIR_USD,
    PROMPT_VERSIONS,
    SENIORITY,
)
from app.services.fit_label_control.spend import SpendAborted, SpendGuard
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    project_root,
    utc_experiment_id,
    write_json,
)
from app.services.llm.factory import build_router
from app.services.llm.tasks import LLMTask
from app.services.llm.provider import LLMResult
from app.schemas.domain import LLMUsage


def api_key_present(settings: Settings) -> bool:
    return bool((settings.openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip())


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


def index_extraction_cache(results_dir: Path) -> dict[str, Path]:
    """Map pair_id -> newest extraction JSON with matching prompt version / model."""
    index: dict[str, tuple[str, Path]] = {}
    if not results_dir.is_dir():
        return {}
    for exp_dir in sorted(results_dir.iterdir()):
        extraction = exp_dir / "extraction"
        if not extraction.is_dir():
            continue
        for path in extraction.glob("*.json"):
            if path.name.startswith("_"):
                continue
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pair_id = rec.get("pair_id")
            if not pair_id:
                continue
            prompts = rec.get("prompt_versions") or {}
            model = rec.get("model")
            if prompts.get("candidate_profile") != PROMPT_VERSIONS["candidate_profile"]:
                continue
            if prompts.get("job_description") != PROMPT_VERSIONS["job_description"]:
                continue
            if model and model != EXTRACTION_MODEL:
                continue
            try:
                CandidateProfile.model_validate(rec["candidate"])
                JobDescription.model_validate(rec["job"])
            except Exception:
                continue
            index[str(pair_id)] = (exp_dir.name, path)
    return {k: v[1] for k, v in index.items()}


def _combo_from_manifest_item(
    item: dict, pairs_dir: Path, seniority: str = SENIORITY
) -> dict:
    return {
        "profile_id": item["profile_id"],
        "jd_id": item["jd_id"],
        "seniority": item.get("seniority", seniority),
        "cv_path": pairs_dir / item["profile_path"],
        "jd_path": pairs_dir / item["job_path"],
        "fit_label": item.get("label"),
        "pair_id": item.get("pair_id"),
    }


def load_combos_from_manifest(manifest_path: Path, pairs_dir: Path) -> list[dict]:
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [_combo_from_manifest_item(it, pairs_dir) for it in items]


def reuse_extraction_record(
    source: dict,
    *,
    experiment_id: str,
) -> dict:
    rec = dict(source)
    rec["experiment_id"] = experiment_id
    rec["reused_from_experiment"] = source.get("experiment_id")
    rec["cv_usage"] = dict(source.get("cv_usage") or {})
    rec["jd_usage"] = dict(source.get("jd_usage") or {})
    rec["cv_usage"]["estimated_cost"] = 0.0
    rec["jd_usage"]["estimated_cost"] = 0.0
    rec["extracted_at"] = datetime.now(timezone.utc).isoformat()
    return rec


async def extract_one_combo(
    *,
    router,
    settings: Settings,
    combo: dict,
    experiment_id: str,
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
    CandidateProfile.model_validate(candidate.model_dump())
    JobDescription.model_validate(job.model_dump())
    return {
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
        "reused_from_experiment": None,
    }


def pair_cost(record: dict) -> float:
    return float(record["cv_usage"]["estimated_cost"]) + float(
        record["jd_usage"]["estimated_cost"]
    )


def dry_run_payload(
    combos: list[dict],
    cache: dict[str, Path],
    *,
    prior_per_pair: float = PRIOR_PER_PAIR_USD,
    extra_uncached: int = 0,
) -> dict[str, Any]:
    cached = [c for c in combos if c.get("pair_id") in cache]
    uncached = [c for c in combos if c.get("pair_id") not in cache]
    n_uncached = len(uncached) + extra_uncached
    est = round(prior_per_pair * n_uncached, 8)
    return {
        "n_pairs": len(combos),
        "n_cached": len(cached),
        "n_uncached": n_uncached,
        "cached_pair_ids": [c.get("pair_id") for c in cached],
        "uncached_pair_ids": [c.get("pair_id") for c in uncached],
        "prior_per_pair_usd": prior_per_pair,
        "estimated_cost_usd": est,
        "note": (
            "Estimated cost = documented primary-30 mean per-pair cost "
            f"({prior_per_pair:.8f} USD from exp_20260927T205741Z) × uncached pairs. "
            "Cached pairs are reused from disk and are not billed again."
        ),
    }


async def run_extraction_guarded(
    *,
    combos: list[dict],
    settings: Settings,
    guard: SpendGuard,
    out_dir: Path,
    experiment_id: str | None = None,
    cache: dict[str, Path] | None = None,
    smoke_n: int = 0,
    router=None,
) -> dict[str, Any]:
    """Extract combos with cache reuse and spend abort. Failures are logged, not dropped silently."""
    root = project_root()
    experiment_id = experiment_id or utc_experiment_id()
    cache = cache if cache is not None else index_extraction_cache(root / "experiments/results")
    extraction_dir = out_dir / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    if smoke_n:
        ordered = combos[:smoke_n] + combos[smoke_n:]
    else:
        ordered = list(combos)

    results: list[dict] = []
    errors: list[dict] = []
    reused = 0
    paid = 0

    if router is None and any(c.get("pair_id") not in cache for c in ordered):
        guard.require_confirm()
        router = build_router(settings)

    for combo in ordered:
        pair_id = combo.get("pair_id")
        request_id = f"{combo['profile_id']}__{combo['jd_id']}__{combo['seniority']}"
        out_path = extraction_dir / f"{request_id}.json"
        try:
            if pair_id and pair_id in cache:
                source = json.loads(cache[pair_id].read_text(encoding="utf-8"))
                record = reuse_extraction_record(source, experiment_id=experiment_id)
                reused += 1
            elif out_path.is_file():
                record = json.loads(out_path.read_text(encoding="utf-8"))
                CandidateProfile.model_validate(record["candidate"])
                JobDescription.model_validate(record["job"])
                reused += 1
            else:
                guard.require_confirm()
                guard.check_next_pair()
                if router is None:
                    router = build_router(settings)
                record = await extract_one_combo(
                    router=router,
                    settings=settings,
                    combo=combo,
                    experiment_id=experiment_id,
                )
                cost = pair_cost(record)
                guard.add_actual(cost, pair_id=str(pair_id))
                paid += 1
            CandidateProfile.model_validate(record["candidate"])
            JobDescription.model_validate(record["job"])
            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            results.append(record)
        except SpendAborted:
            errors.append(
                {
                    "pair_id": pair_id,
                    "profile_id": combo["profile_id"],
                    "error_type": "SpendAborted",
                    "error": guard.abort_message,
                }
            )
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "pair_id": pair_id,
                    "profile_id": combo["profile_id"],
                    "jd_id": combo["jd_id"],
                    "seniority": combo["seniority"],
                    "fit_label": combo.get("fit_label"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    total_cost = sum(pair_cost(r) for r in results if not r.get("reused_from_experiment"))
    # Reused records have cost zeroed; paid cost is guard.running_cost.
    summary = {
        "experiment_id": experiment_id,
        "status": (
            "failed_spend"
            if guard.aborted
            else ("completed" if results and not errors else ("failed" if not results else "partial"))
        ),
        "stage": "extraction",
        "model": settings.llm_fast_model,
        "prompt_versions": {
            "candidate_profile": CV_PROMPT.version,
            "job_description": JD_PROMPT.version,
        },
        "combinations_attempted": len(ordered),
        "combinations_ok": len(results),
        "combinations_failed": len(errors),
        "n_reused_from_disk": reused,
        "n_paid_calls": paid,
        "total_estimated_cost_usd": round(guard.running_cost, 8),
        "written_cost_usd": round(total_cost, 8),
        "errors": errors,
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "result_files": [
            f"extraction/{r['profile_id']}__{r['jd_id']}__{r['seniority']}.json" for r in results
        ],
    }
    write_json(out_dir / "summary.json", summary)
    return summary
