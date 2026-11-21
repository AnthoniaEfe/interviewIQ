"""Selector T-prime on a hand-built similarity matrix. No embeddings."""

from __future__ import annotations

import numpy as np

from app.schemas.domain import CompetencyType, InterviewQuestion
from app.schemas.pool import POOL_DIMENSIONS
from app.services.selection.tprime import K, select_tprime


def _question(priority: float, target: str) -> InterviewQuestion:
    return InterviewQuestion(
        question=target,
        competency=target,
        competency_type=CompetencyType.TECHNICAL,
        target_gap=target,
        priority=priority,
    )


def test_quota_then_reserved_slots_then_remainder():
    dimensions = list(POOL_DIMENSIONS) + ["Technical"] * 5
    targets = [
        "tech-gap",
        "behaviour-partial",
        "situation-strong",
        "culture-gap",
        "role-gap",
        "tech-gap-2",
        "tech-gap-3",
        "tech-gap-4",
        "tech-gap-5",
        "tech-partial",
    ]
    labels = {
        "tech-gap": "weak_match",
        "behaviour-partial": "partial_match",
        "situation-strong": "strong_match",
        "culture-gap": "missing",
        "role-gap": "weak_match",
        "tech-gap-2": "missing",
        "tech-gap-3": "weak_match",
        "tech-gap-4": "weak_match",
        "tech-gap-5": "missing",
        "tech-partial": "partial_match",
    }
    priorities = [0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.4]
    questions = [_question(priorities[i], target) for i, target in enumerate(targets)]
    weights = [1.0] * len(questions)
    similarity = np.eye(len(questions))
    result = select_tprime(
        questions,
        similarity,
        weights,
        dimensions,
        targets,
        labels,
        use_weights=True,
    )
    assert result.k == K
    assert result.selected_indices == [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]
    assert result.quota_met is True
    # The partial and strong items were already taken as dimension quotas,
    # so the second reserved slot has nothing left and is logged, not filled from a gap.
    assert len(result.relaxations) == 1
    assert result.relaxations[0].phase == "reserved"
    assert result.relaxations[0].reason == "no_partial_or_strong_remaining"
