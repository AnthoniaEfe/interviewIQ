"""Resumable batch planning. Identical calls are not paid twice."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def call_cache_key(
    *,
    pair_id: str,
    prompt_version: str,
    model: str,
    config: str,
) -> str:
    payload = json.dumps(
        {
            "pair_id": pair_id,
            "prompt_version": prompt_version,
            "model": model,
            "config": config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pending_jobs(jobs: list[dict[str, str]], cached_keys: set[str]) -> list[dict[str, str]]:
    pending: list[dict[str, str]] = []
    for job in jobs:
        key = call_cache_key(
            pair_id=job["pair_id"],
            prompt_version=job["prompt_version"],
            model=job["model"],
            config=job["config"],
        )
        if key not in cached_keys:
            pending.append({**job, "cache_key": key})
    return pending


async def run_resumable_batch(
    router: Any,
    jobs: list[dict[str, str]],
    cache: dict[str, Any],
    *,
    confirm_spend: bool,
    max_cost_usd: float,
    task: Any,
    schema: type,
    system_prompt: str,
    user_prompt_for: Any,
) -> dict[str, Any]:
    """Run uncached jobs through the router, or plan them when spend is not confirmed.

    `router.complete_structured` is the only model call. A second run with the
    same cache does not call the router again.
    """
    pending = pending_jobs(jobs, set(cache))
    if not confirm_spend:
        return {
            "dry_run": True,
            "calls": 0,
            "n_pending": len(pending),
            "n_cached": len(jobs) - len(pending),
            "spent_usd": 0.0,
        }
    spent = 0.0
    calls = 0
    for job in pending:
        if spent >= max_cost_usd:
            raise RuntimeError(
                f"aborting: spent {spent:.6f} USD has reached max_cost {max_cost_usd:.6f}"
            )
        parsed, usage = await router.complete_structured(
            task=task,
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt_for(job),
            experiment_id=job.get("experiment_id"),
            request_id=job["cache_key"],
            prompt_cache_key=job["cache_key"],
        )
        cost = float(getattr(usage, "estimated_cost", 0.0) or 0.0)
        if spent + cost > max_cost_usd:
            raise RuntimeError(
                f"aborting before recording a call that would exceed max_cost "
                f"({spent + cost:.6f} > {max_cost_usd:.6f})"
            )
        spent += cost
        calls += 1
        cache[job["cache_key"]] = {"parsed": parsed, "cost_usd": cost}
    return {
        "dry_run": False,
        "calls": calls,
        "n_pending": 0,
        "n_cached": len(jobs),
        "spent_usd": spent,
    }
