"""Optional Luna extraction + local gap analysis for the 90 pairs lacking them."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription, LLMUsage
from app.services.embeddings.service import build_embedding_service
from app.services.eval_metrics.constants import (
    EXTRACTION_PRIOR_PER_PAIR_USD,
    MANIFEST_150_REL,
    PAIRS_150_DIR_REL,
    USD_CNY_RATE,
)
from app.services.eval_metrics.spend import MetricsSpendGuard, SpendAborted
from app.services.extraction.profiles import CV_PROMPT, JD_PROMPT, extract_candidate, extract_job
from app.services.fit_label_control.extract import api_key_present
from app.services.gap_analysis.matrix import build_gap_matrix
from app.services.gap_sensitivity.io_utils import (
    project_root,
    utc_experiment_id,
    write_json,
)
from app.services.llm.factory import build_router
from app.services.llm.provider import LLMResult
from app.services.llm.tasks import LLMTask
from app.services.treatment_pilot.currency import usd_to_cny


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


async def _extract_all(
    *,
    settings: Settings,
    combos: list[dict],
    guard: MetricsSpendGuard,
) -> tuple[str, list[dict]]:
    root = project_root()
    router = build_router(settings)
    ext_id = utc_experiment_id()
    ext_dir = root / "experiments" / "results" / ext_id
    (ext_dir / "extraction").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for combo in combos:
        guard.check_next()
        request_id = f"{combo['profile_id']}__{combo['jd_id']}__{combo['seniority']}"
        candidate, cv_usage = await extract_candidate(
            router=router,
            settings=settings,
            cv_text=combo["cv_path"].read_text(encoding="utf-8"),
            experiment_id=ext_id,
            request_id=f"{request_id}:cv",
        )
        job, jd_usage = await extract_job(
            router=router,
            settings=settings,
            jd_text=combo["jd_path"].read_text(encoding="utf-8"),
            experiment_id=ext_id,
            request_id=f"{request_id}:jd",
        )
        cost = float(cv_usage.estimated_cost or 0) + float(jd_usage.estimated_cost or 0)
        guard.add_actual(cost, label=str(combo["pair_id"]))
        record = {
            "experiment_id": ext_id,
            "stage": "extraction",
            "profile_id": combo["profile_id"],
            "jd_id": combo["jd_id"],
            "pair_id": combo["pair_id"],
            "fit_label": combo.get("fit_label"),
            "seniority": combo["seniority"],
            "provider": cv_usage.provider,
            "model": cv_usage.model,
            "prompt_versions": {
                "candidate_profile": CV_PROMPT.version,
                "job_description": JD_PROMPT.version,
            },
            "candidate": candidate.model_dump(),
            "job": job.model_dump(),
            "cv_usage": _usage_payload(cv_usage, LLMTask.CV_EXTRACTION.value),
            "jd_usage": _usage_payload(jd_usage, LLMTask.JD_EXTRACTION.value),
        }
        write_json(ext_dir / "extraction" / f"{request_id}.json", record)
        results.append(record)
        print(f"[extract_missing] extracted {combo['pair_id']} cost={cost:.6f}")
    write_json(
        ext_dir / "summary.json",
        {
            "experiment_id": ext_id,
            "stage": "extraction",
            "status": "completed" if results else "failed",
            "combinations_ok": len(results),
            "total_estimated_cost_usd": guard.running_cost,
            "purpose": "eval_metrics_extract_missing_90",
        },
    )
    return ext_id, results


def maybe_extract_missing(
    *,
    settings: Settings,
    missing_pair_ids: list[str],
    confirm_spend: bool,
    max_cost: float,
    dry_run: bool,
) -> dict[str, Any]:
    root = project_root()
    info: dict[str, Any] = {
        "requested": True,
        "ran": False,
        "dry_run": dry_run,
        "n_missing": len(missing_pair_ids),
        "n_pairs": 0,
        "cost_usd": 0.0,
        "cost_cny": 0.0,
        "skipped_reason": None,
        "extracted_pair_ids": [],
        "extraction_experiment": None,
        "gap_experiment": None,
    }
    if not missing_pair_ids:
        info["skipped_reason"] = "no_missing_pairs"
        return info

    est = EXTRACTION_PRIOR_PER_PAIR_USD * len(missing_pair_ids)
    info["estimated_cost_usd"] = round(est, 8)
    info["estimated_cost_cny"] = usd_to_cny(est, USD_CNY_RATE)

    if not api_key_present(settings):
        info["skipped_reason"] = "missing_api_key"
        print("[extract_missing] no API key; skipping gated extraction")
        return info

    if dry_run:
        info["skipped_reason"] = "dry_run"
        print(
            f"[extract_missing] dry-run: {len(missing_pair_ids)} pairs, "
            f"est {est:.4f} USD / {info['estimated_cost_cny']:.2f} CNY"
        )
        return info

    guard = MetricsSpendGuard(max_cost=max_cost, confirm_spend=confirm_spend)
    try:
        guard.require_confirm()
    except SpendAborted as exc:
        info["skipped_reason"] = str(exc)
        print(f"[extract_missing] {exc}")
        return info

    manifest = json.loads((root / MANIFEST_150_REL).read_text(encoding="utf-8"))
    by_id = {str(x["pair_id"]): x for x in manifest}
    pairs_dir = root / PAIRS_150_DIR_REL
    combos = []
    for pid in missing_pair_ids:
        item = by_id.get(pid)
        if not item:
            continue
        combos.append(
            {
                "profile_id": item["profile_id"],
                "jd_id": item["jd_id"],
                "seniority": "mid",
                "cv_path": pairs_dir / item["profile_path"],
                "jd_path": pairs_dir / item["job_path"],
                "fit_label": item.get("label"),
                "pair_id": pid,
            }
        )

    try:
        ext_id, results = asyncio.run(
            _extract_all(settings=settings, combos=combos, guard=guard)
        )
    except SpendAborted as exc:
        info["skipped_reason"] = str(exc)
        print(f"[extract_missing] {exc}")
        return info

    emb = build_embedding_service(settings.embedding_model, settings.embedding_device)
    gap_id = utc_experiment_id()
    gap_dir = root / "experiments" / "results" / gap_id
    (gap_dir / "gap_analysis").mkdir(parents=True, exist_ok=True)
    gap_ok = 0
    for record in results:
        candidate = CandidateProfile.model_validate(record["candidate"])
        job = JobDescription.model_validate(record["job"])
        matrix = build_gap_matrix(
            profile=candidate,
            job=job,
            settings=settings,
            embeddings=emb,
        )
        stem = f"{record['profile_id']}__{record['jd_id']}__{record['seniority']}"
        payload = {
            "experiment_id": gap_id,
            "stage": "gap_analysis",
            "source_experiment": ext_id,
            "profile_id": record["profile_id"],
            "jd_id": record["jd_id"],
            "pair_id": record["pair_id"],
            "fit_label": record.get("fit_label"),
            "seniority": record["seniority"],
            "embedding_model": settings.embedding_model,
            "gap_matrix": matrix.model_dump(),
        }
        write_json(gap_dir / "gap_analysis" / f"{stem}.json", payload)
        gap_ok += 1

    write_json(
        gap_dir / "summary.json",
        {
            "experiment_id": gap_id,
            "stage": "gap_analysis",
            "status": "completed" if gap_ok else "failed",
            "source_experiment": ext_id,
            "combinations_ok": gap_ok,
            "purpose": "eval_metrics_extract_missing_90",
        },
    )

    info.update(
        {
            "ran": bool(results),
            "n_pairs": len(results),
            "cost_usd": round(guard.running_cost, 8),
            "cost_cny": usd_to_cny(guard.running_cost, USD_CNY_RATE),
            "extracted_pair_ids": [str(r["pair_id"]) for r in results],
            "extraction_experiment": ext_id,
            "gap_experiment": gap_id,
        }
    )
    return info
