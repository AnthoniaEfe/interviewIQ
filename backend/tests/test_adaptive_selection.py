"""Adaptive lambda and competency-budget selection. No network."""

from __future__ import annotations

import numpy as np
import pytest

from app.schemas.domain import InterviewQuestion
from app.schemas.pool import POOL_DIMENSIONS
from app.services.mmr.adaptive import (
    AdaptiveConfig,
    factorial_grid,
    lambda_from_coverage,
    named_ablation_grid,
    select_adaptive,
    should_reserve_slots,
)
from app.services.mmr.select import competency_weighted_mmr
from app.services.selection.strategies import ADAPTIVE_SELECTION_STAGE, select_questions


def _questions(priorities: list[float], dimensions: list[str]) -> list[InterviewQuestion]:
    return [
        InterviewQuestion(
            question=f"q{i}",
            competency=f"c{i}",
            question_type=dimensions[i],
            priority=priorities[i],
        )
        for i in range(len(priorities))
    ]


def _random_pool(rng: np.random.Generator, n: int, *, tie: bool = False):
    priorities = rng.random(n)
    weights = rng.uniform(0.2, 1.5, size=n)
    raw = rng.normal(size=(n, 8))
    raw /= np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)
    sim = raw @ raw.T
    np.fill_diagonal(sim, 1.0)
    dims = [POOL_DIMENSIONS[i % len(POOL_DIMENSIONS)] for i in range(n)]
    if tie:
        priorities[1] = priorities[0]
        weights[1] = weights[0]
        sim[1, :] = sim[0, :]
        sim[:, 1] = sim[:, 0]
        sim[1, 1] = 1.0
        sim[0, 0] = 1.0
    questions = _questions(priorities.tolist(), dims)
    return questions, sim, weights.tolist(), dims


def test_weighted_mmr_parity_many_seeds():
    cfg = named_ablation_grid()["weighted_mmr"]
    for seed in range(24):
        rng = np.random.default_rng(seed)
        questions, sim, weights, dims = _random_pool(rng, n=12, tie=(seed % 5 == 0))
        k = 5
        got = select_adaptive(questions, sim, weights, dims, cfg, k=k)
        ref = competency_weighted_mmr(questions, sim, weights, lambda_=0.7, k=k)
        assert got.selected_indices == ref.selected_indices
        assert got.relevance_scores == ref.relevance_scores
        assert got.redundancy_scores == ref.redundancy_scores
        assert got.mmr_scores == ref.mmr_scores
        assert got.competency_weights == ref.competency_weights
        assert got.lambda_t == [0.7] * k
        assert got.reserved_slot_active == [False] * k


def test_plain_mmr_matches_unit_weights():
    cfg = named_ablation_grid()["plain_mmr"]
    rng = np.random.default_rng(7)
    questions, sim, weights, dims = _random_pool(rng, n=10)
    got = select_adaptive(questions, sim, weights, dims, cfg, k=4)
    ref = competency_weighted_mmr(
        questions, sim, [1.0] * len(questions), lambda_=0.7, k=4
    )
    assert got.selected_indices == ref.selected_indices
    assert got.relevance_scores == ref.relevance_scores
    assert got.redundancy_scores == ref.redundancy_scores
    assert got.mmr_scores == ref.mmr_scores
    assert got.competency_weights == ref.competency_weights


def test_adaptive_lambda_bounds_monotone_and_fixed_switch():
    up = AdaptiveConfig(
        use_adaptive_lambda=True,
        budget_mode="min_quota",
        lambda_min=0.5,
        lambda_max=0.9,
        lambda_direction="coverage_up",
    )
    down = AdaptiveConfig(
        use_adaptive_lambda=True,
        budget_mode="min_quota",
        lambda_min=0.5,
        lambda_max=0.9,
        lambda_direction="coverage_down",
    )
    coverages = [0.0, 0.25, 0.5, 1.0]
    up_vals = [lambda_from_coverage(c, up) for c in coverages]
    down_vals = [lambda_from_coverage(c, down) for c in coverages]
    assert up_vals == pytest.approx([0.5, 0.6, 0.7, 0.9])
    assert down_vals == pytest.approx([0.9, 0.8, 0.7, 0.5])
    assert all(0.5 <= v <= 0.9 for v in up_vals + down_vals)
    assert up_vals == sorted(up_vals)
    assert down_vals == sorted(down_vals, reverse=True)

    off = AdaptiveConfig(use_adaptive_lambda=False, budget_mode="none", lambda_fixed=0.7)
    assert lambda_from_coverage(0.0, off) == 0.7
    assert lambda_from_coverage(1.0, off) == 0.7

    dims = [POOL_DIMENSIONS[i % 5] for i in range(15)]
    priorities = [1.0 - i * 0.01 for i in range(15)]
    questions = _questions(priorities, dims)
    sim = np.eye(15)
    weights = [1.0] * 15
    got_up = select_adaptive(questions, sim, weights, dims, up, k=10)
    got_down = select_adaptive(
        questions,
        sim,
        weights,
        dims,
        AdaptiveConfig(
            use_competency_weights=True,
            use_budget=False,
            budget_mode="min_quota",
            use_adaptive_lambda=True,
            lambda_direction="coverage_down",
        ),
        k=10,
    )
    assert all(0.5 <= v <= 0.9 for v in got_up.lambda_t + got_down.lambda_t)
    assert got_up.lambda_t == sorted(got_up.lambda_t)
    assert got_down.lambda_t == sorted(got_down.lambda_t, reverse=True)
    got_off = select_adaptive(questions, sim, weights, dims, off, k=10)
    assert got_off.lambda_t == [0.7] * 10


def test_min_quota_covers_five_dimensions_and_logs_missing():
    dims = []
    priorities = []
    for d_i, dimension in enumerate(POOL_DIMENSIONS):
        for j in range(6):
            dims.append(dimension)
            # Technical dominates relevance so an unrestricted greedy pass stays there.
            priorities.append(1.0 - j * 0.01 if dimension == "Technical" else 0.1 - d_i * 0.01)
    questions = _questions(priorities, dims)
    n = len(questions)
    sim = np.eye(n)
    weights = [1.0] * n
    cfg = AdaptiveConfig(
        use_competency_weights=True,
        use_budget=True,
        budget_mode="min_quota",
        use_adaptive_lambda=False,
        lambda_fixed=1.0,
    )
    got = select_adaptive(questions, sim, weights, dims, cfg, k=10)
    assert len(got.selected_indices) == 10
    selected_dims = {dims[i] for i in got.selected_indices}
    assert selected_dims == set(POOL_DIMENSIONS)
    assert got.shortfalls == []

    missing = [d for d in dims if d != "Cultural/Motivational"]
    # Rebuild priorities/questions aligned with `missing` — drop that dimension's rows.
    keep = [i for i, d in enumerate(dims) if d != "Cultural/Motivational"]
    q2 = [questions[i] for i in keep]
    d2 = [dims[i] for i in keep]
    sim2 = np.eye(len(q2))
    got2 = select_adaptive(q2, sim2, [1.0] * len(q2), d2, cfg, k=10)
    assert len(got2.selected_indices) == 10
    reasons = {(s.dimension, s.reason) for s in got2.shortfalls}
    assert ("Cultural/Motivational", "zero_availability") in reasons


def test_reserved_slots_trigger_exactly_on_unmet_vs_remaining():
    assert should_reserve_slots(4, 4) is True
    assert should_reserve_slots(5, 4) is True
    assert should_reserve_slots(3, 4) is False

    dims = ["Technical"] * 3 + ["Behavioural", "Situational", "Cultural/Motivational"]
    priorities = [1.0, 0.99, 0.98, 0.1, 0.1, 0.1]
    questions = _questions(priorities, dims)
    sim = np.eye(6)
    weights = [1.0] * 6
    tight = AdaptiveConfig(
        use_budget=True,
        budget_mode="custom",
        custom_budgets={
            "Technical": 1,
            "Behavioural": 1,
            "Situational": 1,
            "Cultural/Motivational": 1,
        },
        use_adaptive_lambda=False,
        lambda_fixed=1.0,
    )
    got = select_adaptive(questions, sim, weights, dims, tight, k=4)
    assert got.reserved_slot_active == [True, True, True, True]
    assert [dims[i] for i in got.selected_indices] == [
        "Technical",
        "Behavioural",
        "Situational",
        "Cultural/Motivational",
    ]

    loose_pool_dims = dims + ["Role-Specific"]
    loose_q = _questions(priorities + [0.1], loose_pool_dims)
    loose = AdaptiveConfig(
        use_budget=True,
        budget_mode="custom",
        custom_budgets={
            "Technical": 1,
            "Behavioural": 1,
            "Situational": 1,
            "Cultural/Motivational": 1,
        },
        use_adaptive_lambda=False,
        lambda_fixed=1.0,
    )
    got_loose = select_adaptive(
        loose_q, np.eye(7), [1.0] * 7, loose_pool_dims, loose, k=5
    )
    # unmet starts at 4, remaining slots at 5.
    assert got_loose.reserved_slot_active[0] is False


def test_determinism_and_validation_errors():
    rng = np.random.default_rng(3)
    questions, sim, weights, dims = _random_pool(rng, n=8)
    cfg = named_ablation_grid()["full"]
    a = select_adaptive(questions, sim, weights, dims, cfg, k=5)
    b = select_adaptive(questions, sim, weights, dims, cfg, k=5)
    assert a == b

    with pytest.raises(ValueError, match="exceeds pool"):
        select_adaptive(questions, sim, weights, dims, cfg, k=9)
    with pytest.raises(ValueError, match="lambda_min"):
        select_adaptive(
            questions,
            sim,
            weights,
            dims,
            AdaptiveConfig(lambda_min=0.95, lambda_max=0.2, budget_mode="none"),
            k=3,
        )
    with pytest.raises(ValueError, match="lambda_fixed"):
        select_adaptive(
            questions,
            sim,
            weights,
            dims,
            AdaptiveConfig(lambda_fixed=1.2, budget_mode="none"),
            k=3,
        )
    with pytest.raises(ValueError, match="sum\\(budgets\\)"):
        select_adaptive(
            questions,
            sim,
            weights,
            dims,
            AdaptiveConfig(
                use_budget=True,
                budget_mode="custom",
                custom_budgets={"Technical": 3, "Behavioural": 3},
            ),
            k=5,
        )
    with pytest.raises(ValueError, match="budget_mode"):
        select_adaptive(
            questions,
            sim,
            weights,
            dims,
            AdaptiveConfig(use_adaptive_lambda=True, budget_mode="none"),
            k=5,
        )
    with pytest.raises(ValueError, match="weights length"):
        select_adaptive(questions, sim, weights[:-1], dims, cfg, k=5)


def test_named_configs_on_pool_30_return_10():
    rng = np.random.default_rng(11)
    questions, sim, weights, dims = _random_pool(rng, n=30)
    for name, cfg in named_ablation_grid().items():
        got = select_adaptive(questions, sim, weights, dims, cfg, k=10)
        assert len(got.selected_indices) == 10, name
        assert len(set(got.selected_indices)) == 10
    assert len(factorial_grid()) == 8
    for name, cfg in factorial_grid().items():
        got = select_adaptive(questions, sim, weights, dims, cfg, k=10)
        assert len(got.selected_indices) == 10, name


def test_registry_weighted_mmr_matches_classical():
    rng = np.random.default_rng(4)
    questions, sim, weights, dims = _random_pool(rng, n=9)
    for q, d in zip(questions, dims, strict=True):
        q.question_type = d
    key = f"{ADAPTIVE_SELECTION_STAGE}/weighted_mmr"
    got = select_questions(key, questions, sim, weights, lambda_=0.7, k=4)
    ref = competency_weighted_mmr(questions, sim, weights, lambda_=0.7, k=4)
    assert got.selected_indices == ref.selected_indices
    assert got.mmr_scores == ref.mmr_scores
    assert got.strategy == key
