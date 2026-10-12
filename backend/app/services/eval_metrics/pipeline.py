"""Entry points for metrics_freeze / metrics_pilot / scale_decision stages."""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings, get_settings
from app.services.eval_metrics.baseline_run import run_baseline_and_freeze
from app.services.eval_metrics.constants import (
    DEFAULT_MAX_COST_USD,
    DEFAULT_SCALE_BUDGET_CNY,
)
from app.services.eval_metrics.pilot_run import run_metrics_pilot_stage
from app.services.eval_metrics.scale_run import run_scale_decision_stage


def run_metrics_freeze(
    *,
    settings: Settings | None = None,
    dry_run: bool = False,
    confirm_spend: bool = False,
    max_cost: float = DEFAULT_MAX_COST_USD,
    extract_missing: bool = False,
    with_dimension_labels: bool = False,
) -> dict[str, Any]:
    settings = settings or get_settings()
    return run_baseline_and_freeze(
        settings=settings,
        extract_missing=extract_missing,
        confirm_spend=confirm_spend,
        max_cost=max_cost,
        dry_run=dry_run,
        with_dimension_labels=with_dimension_labels,
    )


def run_metrics_pilot(
    *,
    settings: Settings | None = None,
    allow_post_hoc: bool = False,
) -> dict[str, Any]:
    settings = settings or get_settings()
    return run_metrics_pilot_stage(settings=settings, allow_post_hoc=allow_post_hoc)


def run_scale_decision(
    *,
    settings: Settings | None = None,
    allow_post_hoc: bool = False,
    scale_budget_cny: float = DEFAULT_SCALE_BUDGET_CNY,
    pilot_experiment: str | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    return run_scale_decision_stage(
        settings=settings,
        allow_post_hoc=allow_post_hoc,
        scale_budget_cny=scale_budget_cny,
        pilot_experiment=pilot_experiment,
    )
