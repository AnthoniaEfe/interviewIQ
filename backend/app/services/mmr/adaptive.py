"""Adaptive lambda and competency-budget selection.

Pure functions. With every switch off except competency weights, the selected
indices and per-step scores match ``competency_weighted_mmr`` in
``app.services.mmr.select`` (same relevance, same strict-``>`` argmax, same
``remaining`` set iteration order, same redundancy definition, no relevance
normalisation).

Relevance normalisation (``normalize_relevance=True``) is off by default.
When it is on, relevance is min-max scaled across the pool before the loop:

    lo, hi = min(relevance), max(relevance)
    if hi > lo: relevance = (relevance - lo) / (hi - lo)
    elif hi == lo and hi != 0: relevance = 1 for every item
    else: relevance stays 0

Zero-availability dimensions (budget > 0 but no pool question in that
dimension) are excluded from the coverage denominator and recorded in
``shortfalls`` with reason ``zero_availability``. If every budgeted dimension
is unavailable, coverage is defined as 1.0 (nothing feasible remains to cover).

``min_quota`` assigns one slot to each of the five canonical pool dimensions
(Technical, Behavioural, Situational, Cultural/Motivational, Role-Specific),
including dimensions absent from the pool. Other labels are not given a quota.

Reserved slots: at a step, ``unmet`` is the sum over feasible dimensions
(at least one question in the original pool) of ``max(0, budget - selected)``.
The candidate set is restricted when ``unmet >= remaining_slots``. If that
restriction is empty because those dimensions have no questions left, the step
still records ``reserved_slot_active=True`` and fills from all remaining
questions so the selector still returns ``k`` items.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.schemas.domain import InterviewQuestion
from app.schemas.pool import POOL_DIMENSIONS

CANONICAL_DIMENSIONS: tuple[str, ...] = POOL_DIMENSIONS


@dataclass(frozen=True)
class Shortfall:
    dimension: str
    budget: int
    achieved: int
    reason: str


@dataclass(frozen=True)
class AdaptiveConfig:
    use_competency_weights: bool = True
    use_budget: bool = False
    budget_mode: str = "min_quota"
    use_adaptive_lambda: bool = False
    lambda_fixed: float = 0.7
    lambda_min: float = 0.5
    lambda_max: float = 0.9
    lambda_direction: str = "coverage_up"
    normalize_relevance: bool = False
    custom_budgets: dict[str, int] | None = None


@dataclass
class AdaptiveSelection:
    selected_indices: list[int]
    lambda_t: list[float]
    coverage_t: list[float]
    dimension_counts: dict[str, int]
    relevance_scores: list[float]
    redundancy_scores: list[float]
    mmr_scores: list[float]
    competency_weights: list[float]
    reserved_slot_active: list[bool]
    shortfalls: list[Shortfall] = field(default_factory=list)


def should_reserve_slots(unmet: int, remaining_slots: int) -> bool:
    """Reserved-slot restriction is active exactly when unmet >= remaining slots."""
    return unmet >= remaining_slots


def lambda_from_coverage(coverage: float, config: AdaptiveConfig) -> float:
    if not config.use_adaptive_lambda:
        return float(config.lambda_fixed)
    span = float(config.lambda_max) - float(config.lambda_min)
    if config.lambda_direction == "coverage_up":
        return float(config.lambda_min) + span * float(coverage)
    if config.lambda_direction == "coverage_down":
        return float(config.lambda_max) - span * float(coverage)
    raise ValueError(
        f"lambda_direction must be 'coverage_up' or 'coverage_down', got {config.lambda_direction!r}"
    )


def coverage_fraction(
    counts: dict[str, int],
    budgets: dict[str, int],
    available: dict[str, int],
) -> float:
    """Mean over feasible budgeted dimensions of min(1, n_d / b_d)."""
    terms: list[float] = []
    for dimension, budget in budgets.items():
        if budget <= 0:
            continue
        if available.get(dimension, 0) <= 0:
            continue
        terms.append(min(1.0, counts.get(dimension, 0) / float(budget)))
    if not terms:
        return 1.0
    return float(sum(terms) / len(terms))


def resolve_budgets(config: AdaptiveConfig) -> dict[str, int]:
    mode = config.budget_mode
    if mode == "none":
        return {}
    if mode == "min_quota":
        return {dimension: 1 for dimension in CANONICAL_DIMENSIONS}
    if mode == "custom":
        raw = config.custom_budgets or {}
        budgets: dict[str, int] = {}
        for dimension, budget in raw.items():
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise ValueError(
                    f"budget for {dimension!r} must be a non-negative int, got {budget!r}"
                )
            if budget < 0:
                raise ValueError(f"budget for {dimension!r} is negative ({budget})")
            budgets[str(dimension)] = int(budget)
        return budgets
    raise ValueError(
        f"budget_mode must be 'min_quota', 'custom', or 'none', got {mode!r}"
    )


def _validate_call(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    dimensions: list[str],
    config: AdaptiveConfig,
    k: int,
) -> np.ndarray:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"k must be an int, got {k!r}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    n = len(questions)
    if k > n:
        raise ValueError(f"k ({k}) exceeds pool length ({n})")
    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.ndim != 2 or sim.shape != (n, n):
        raise ValueError(
            f"similarity_matrix must be square ({n}, {n}), got shape {sim.shape}"
        )
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != pool length {n}")
    if len(dimensions) != n:
        raise ValueError(f"dimensions length {len(dimensions)} != pool length {n}")
    for name in ("lambda_fixed", "lambda_min", "lambda_max"):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if float(config.lambda_min) > float(config.lambda_max):
        raise ValueError(
            f"lambda_min ({config.lambda_min}) > lambda_max ({config.lambda_max})"
        )
    if config.lambda_direction not in {"coverage_up", "coverage_down"}:
        raise ValueError(
            f"lambda_direction must be 'coverage_up' or 'coverage_down', got {config.lambda_direction!r}"
        )
    if config.use_adaptive_lambda and config.budget_mode == "none":
        raise ValueError(
            "use_adaptive_lambda requires budget_mode != 'none' "
            "(coverage needs budget targets; reserved slots are not required)"
        )
    budgets = resolve_budgets(config)
    total = sum(budgets.values())
    if total > k:
        raise ValueError(f"sum(budgets)={total} exceeds k={k}")
    return sim


def _normalize(relevance: np.ndarray) -> np.ndarray:
    lo = float(relevance.min()) if relevance.size else 0.0
    hi = float(relevance.max()) if relevance.size else 0.0
    if hi > lo:
        return (relevance - lo) / (hi - lo)
    if hi == lo and hi != 0.0:
        return np.ones_like(relevance)
    return relevance


def select_adaptive(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    dimensions: list[str],
    config: AdaptiveConfig,
    *,
    k: int,
) -> AdaptiveSelection:
    """Greedy MMR with optional competency weights, reserved quotas, and adaptive λ."""
    sim = _validate_call(questions, similarity_matrix, weights, dimensions, config, k)
    n = len(questions)
    budgets = resolve_budgets(config)
    dims = [str(d) for d in dimensions]
    available: dict[str, int] = {}
    for dimension in dims:
        available[dimension] = available.get(dimension, 0) + 1
    for dimension in budgets:
        available.setdefault(dimension, 0)

    relevance = np.array(
        [
            float(questions[i].priority)
            * (float(weights[i]) if config.use_competency_weights else 1.0)
            for i in range(n)
        ],
        dtype=float,
    )
    if config.normalize_relevance and n:
        relevance = _normalize(relevance)

    counts: dict[str, int] = {dimension: 0 for dimension in set(budgets) | set(dims)}
    shortfalls: list[Shortfall] = []
    for dimension, budget in budgets.items():
        if budget > 0 and available.get(dimension, 0) <= 0:
            shortfalls.append(
                Shortfall(
                    dimension=dimension,
                    budget=budget,
                    achieved=0,
                    reason="zero_availability",
                )
            )

    selected: list[int] = []
    remaining: set[int] = set(range(n))
    lambda_trace: list[float] = []
    coverage_trace: list[float] = []
    reserved_flags: list[bool] = []
    recorded_rel: list[float] = []
    recorded_red: list[float] = []
    recorded_mmr: list[float] = []
    recorded_w: list[float] = []

    while remaining and len(selected) < k:
        coverage = coverage_fraction(counts, budgets, available)
        lambda_t = lambda_from_coverage(coverage, config)
        feasible = [d for d, budget in budgets.items() if budget > 0 and available.get(d, 0) > 0]
        unmet = sum(max(0, budgets[d] - counts.get(d, 0)) for d in feasible)
        slots_left = k - len(selected)
        reserve = bool(config.use_budget) and should_reserve_slots(unmet, slots_left)
        unmet_dims = {
            d for d in feasible if budgets[d] - counts.get(d, 0) > 0
        }
        best_i = None
        best_score = -1e18
        best_red = 0.0
        # Same iteration as competency_weighted_mmr: `for i in remaining` (set order),
        # strict `>`. Restriction skips ineligible indices without rebuilding the set,
        # so relative order among eligible indices matches the unrestricted scan.
        for i in remaining:
            if reserve and unmet_dims and dims[i] not in unmet_dims:
                continue
            redundancy = 0.0
            if selected:
                redundancy = float(max(sim[i, j] for j in selected))
            score = lambda_t * float(relevance[i]) - (1.0 - lambda_t) * redundancy
            if score > best_score:
                best_score = score
                best_i = i
                best_red = redundancy
        if best_i is None:
            # Restricted set was empty (quotas cannot be met). Fill by the normal rule.
            for i in remaining:
                redundancy = 0.0
                if selected:
                    redundancy = float(max(sim[i, j] for j in selected))
                score = lambda_t * float(relevance[i]) - (1.0 - lambda_t) * redundancy
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_red = redundancy
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
        counts[dims[best_i]] = counts.get(dims[best_i], 0) + 1
        lambda_trace.append(float(lambda_t))
        coverage_trace.append(float(coverage))
        reserved_flags.append(bool(reserve))
        recorded_rel.append(float(relevance[best_i]))
        recorded_red.append(best_red)
        recorded_mmr.append(best_score)
        if config.use_competency_weights:
            recorded_w.append(float(weights[best_i]))
        else:
            recorded_w.append(1.0)

    for dimension, budget in budgets.items():
        if budget <= 0 or available.get(dimension, 0) <= 0:
            continue
        achieved = counts.get(dimension, 0)
        if achieved < budget:
            shortfalls.append(
                Shortfall(
                    dimension=dimension,
                    budget=budget,
                    achieved=achieved,
                    reason="unmet_quota",
                )
            )

    return AdaptiveSelection(
        selected_indices=selected,
        lambda_t=lambda_trace,
        coverage_t=coverage_trace,
        dimension_counts={dimension: counts.get(dimension, 0) for dimension in counts},
        relevance_scores=recorded_rel,
        redundancy_scores=recorded_red,
        mmr_scores=recorded_mmr,
        competency_weights=recorded_w,
        reserved_slot_active=reserved_flags,
        shortfalls=shortfalls,
    )


def named_ablation_grid() -> dict[str, AdaptiveConfig]:
    """Named switches. ``weighted_mmr`` is the parity target for classical weighted MMR."""
    return {
        "plain_mmr": AdaptiveConfig(
            use_competency_weights=False,
            use_budget=False,
            budget_mode="none",
            use_adaptive_lambda=False,
        ),
        "weighted_mmr": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=False,
            budget_mode="none",
            use_adaptive_lambda=False,
        ),
        "weighted_budget": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=True,
            budget_mode="min_quota",
            use_adaptive_lambda=False,
        ),
        "weighted_adaptive": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=False,
            budget_mode="min_quota",
            use_adaptive_lambda=True,
            lambda_direction="coverage_up",
        ),
        "full": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=True,
            budget_mode="min_quota",
            use_adaptive_lambda=True,
            lambda_direction="coverage_up",
        ),
        "full_minus_weights": AdaptiveConfig(
            use_competency_weights=False,
            use_budget=True,
            budget_mode="min_quota",
            use_adaptive_lambda=True,
            lambda_direction="coverage_up",
        ),
        "full_minus_budget": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=False,
            budget_mode="min_quota",
            use_adaptive_lambda=True,
            lambda_direction="coverage_up",
        ),
        "full_minus_adaptive": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=True,
            budget_mode="min_quota",
            use_adaptive_lambda=False,
        ),
        "full_coverage_down": AdaptiveConfig(
            use_competency_weights=True,
            use_budget=True,
            budget_mode="min_quota",
            use_adaptive_lambda=True,
            lambda_direction="coverage_down",
        ),
    }


def factorial_grid() -> dict[str, AdaptiveConfig]:
    """Full 2^3 over (competency weights, reserved budget, adaptive lambda).

    Adaptive lambda needs budget targets, so cells with adaptive lambda on use
    ``budget_mode='min_quota'`` even when reserved slots are off. Cells with both
    budget and adaptive lambda off use ``budget_mode='none'``.
    """
    grid: dict[str, AdaptiveConfig] = {}
    for weights_on in (False, True):
        for budget_on in (False, True):
            for adaptive_on in (False, True):
                name = (
                    f"weights_{int(weights_on)}__budget_{int(budget_on)}__adaptive_{int(adaptive_on)}"
                )
                if adaptive_on or budget_on:
                    mode = "min_quota"
                else:
                    mode = "none"
                grid[name] = AdaptiveConfig(
                    use_competency_weights=weights_on,
                    use_budget=budget_on,
                    budget_mode=mode,
                    use_adaptive_lambda=adaptive_on,
                    lambda_direction="coverage_up",
                )
    return grid
