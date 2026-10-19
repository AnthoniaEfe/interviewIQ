"""Spend gates for the 150-pair run. Paid work requires every gate."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.eval_metrics.freeze import FreezeError, verify_freeze

SCALE_SUMMARY = "experiments/results/exp_20261012T163853Z/summary.json"
PREREG = "docs/adaptive_selection_prereg.md"
USD_CNY = 6.77


def evaluate_paid_gates(root: Path) -> dict:
    """Return whether a paid 150-pair run is allowed, and every failed gate."""
    failures: list[str] = []
    summary_path = root / SCALE_SUMMARY
    verdict = None
    projected_cny = None
    projected_usd = None
    budget_cny = None
    if not summary_path.is_file():
        failures.append(f"scale-decision summary missing: {SCALE_SUMMARY}")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        verdict = (summary.get("decision") or {}).get("verdict")
        cost = summary.get("cost_projection") or {}
        projected_cny = cost.get("total_cny")
        projected_usd = cost.get("total_usd")
        budget_cny = summary.get("scale_budget_cny")
        if verdict != "SCALE":
            failures.append(
                f"scale decision on record is {verdict!r}, not 'SCALE' "
                f"({SCALE_SUMMARY})"
            )
        if projected_cny is None or budget_cny is None:
            failures.append("scale summary has no projected cost or budget")
        elif float(projected_cny) > float(budget_cny):
            failures.append(
                f"projected cost {projected_cny} CNY exceeds budget {budget_cny} CNY"
            )
    try:
        verify_freeze()
    except FreezeError as exc:
        failures.append(f"frozen-metrics hashes do not verify: {exc}")
    prereg = root / PREREG
    if not prereg.is_file():
        failures.append(f"step-5 pre-registration missing: {PREREG}")
    return {
        "allowed": not failures,
        "failures": failures,
        "scale_verdict": verdict,
        "projected_usd": projected_usd,
        "projected_cny": projected_cny,
        "scale_budget_cny": budget_cny,
        "usd_cny_rate": USD_CNY,
        "scale_summary": SCALE_SUMMARY,
    }
