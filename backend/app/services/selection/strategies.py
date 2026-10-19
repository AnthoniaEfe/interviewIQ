"""Named selection strategies (registry). Coverage-dependent λ can be added later."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.schemas.domain import InterviewQuestion
from app.services.mmr.select import MMRSelection, competency_weighted_mmr


@dataclass
class SelectionResult:
    selected_indices: list[int]
    strategy: str
    relevance_scores: list[float] = field(default_factory=list)
    redundancy_scores: list[float] = field(default_factory=list)
    mmr_scores: list[float] = field(default_factory=list)
    competency_weights: list[float] = field(default_factory=list)
    backfilled: bool = False
    backfill_count: int = 0
    notes: list[str] = field(default_factory=list)


def _ensure_k(
    selected: list[int],
    n: int,
    k: int,
    *,
    order_key: Callable[[int], float] | None = None,
) -> tuple[list[int], bool, int]:
    """Backfill unselected indices until k (or n) reached. Never silently short."""
    if n == 0:
        return [], False, 0
    out = list(selected)
    remaining = [i for i in range(n) if i not in set(out)]
    if order_key is not None:
        remaining.sort(key=order_key, reverse=True)
    backfill = 0
    while len(out) < min(k, n) and remaining:
        out.append(remaining.pop(0))
        backfill += 1
    return out, backfill > 0, backfill


def _from_mmr(
    mmr: MMRSelection,
    *,
    strategy: str,
    n: int,
    k: int,
    questions: list[InterviewQuestion],
    weights: list[float],
) -> SelectionResult:
    selected, backfilled, n_back = _ensure_k(
        mmr.selected_indices,
        n,
        k,
        order_key=lambda i: float(questions[i].priority)
        * float(weights[i] if i < len(weights) else 1.0),
    )
    notes: list[str] = []
    if len(selected) < k:
        notes.append(
            f"pool_shorter_than_k: selected={len(selected)} k={k} n_pool={n}"
        )
    return SelectionResult(
        selected_indices=selected,
        strategy=strategy,
        relevance_scores=list(mmr.relevance_scores),
        redundancy_scores=list(mmr.redundancy_scores),
        mmr_scores=list(mmr.mmr_scores),
        competency_weights=list(mmr.competency_weights),
        backfilled=backfilled,
        backfill_count=n_back,
        notes=notes,
    )


def strategy_competency_weighted_mmr(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
    seed: int = 42,
    query_similarities: np.ndarray | None = None,
) -> SelectionResult:
    del seed, query_similarities
    mmr = competency_weighted_mmr(
        questions, similarity_matrix, weights, lambda_=lambda_, k=k
    )
    return _from_mmr(
        mmr,
        strategy="competency_weighted_mmr",
        n=len(questions),
        k=k,
        questions=questions,
        weights=weights,
    )


def strategy_plain_mmr_priority(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
    seed: int = 42,
    query_similarities: np.ndarray | None = None,
) -> SelectionResult:
    """Classic MMR with relevance = question.priority (all competency weights = 1)."""
    del weights, seed, query_similarities
    ones = [1.0] * len(questions)
    mmr = competency_weighted_mmr(
        questions, similarity_matrix, ones, lambda_=lambda_, k=k
    )
    return _from_mmr(
        mmr,
        strategy="plain_mmr_priority",
        n=len(questions),
        k=k,
        questions=questions,
        weights=ones,
    )


def _mmr_with_relevance(
    relevance: np.ndarray,
    similarity_matrix: np.ndarray,
    *,
    lambda_: float,
    k: int,
) -> list[int]:
    n = int(relevance.shape[0])
    if n == 0 or k <= 0:
        return []
    selected: list[int] = []
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e18
        for i in remaining:
            redundancy = 0.0
            if selected:
                redundancy = float(max(similarity_matrix[i, j] for j in selected))
            score = lambda_ * float(relevance[i]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_i = i
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def strategy_plain_mmr_query(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
    seed: int = 42,
    query_similarities: np.ndarray | None = None,
) -> SelectionResult:
    """MMR with relevance = cosine(question, job-description embedding)."""
    del weights, seed
    n = len(questions)
    if query_similarities is None:
        raise ValueError("plain_mmr_query requires query_similarities")
    rel = np.asarray(query_similarities, dtype=float).reshape(-1)
    if rel.shape[0] != n:
        raise ValueError("query_similarities length must match questions")
    selected = _mmr_with_relevance(rel, similarity_matrix, lambda_=lambda_, k=k)
    selected, backfilled, n_back = _ensure_k(
        selected, n, k, order_key=lambda i: float(rel[i])
    )
    return SelectionResult(
        selected_indices=selected,
        strategy="plain_mmr_query",
        relevance_scores=[float(rel[i]) for i in selected],
        backfilled=backfilled,
        backfill_count=n_back,
        notes=[]
        if len(selected) >= min(k, n)
        else [f"pool_shorter_than_k: selected={len(selected)} k={k}"],
    )


def strategy_top_priority(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
    seed: int = 42,
    query_similarities: np.ndarray | None = None,
) -> SelectionResult:
    del similarity_matrix, weights, lambda_, seed, query_similarities
    n = len(questions)
    order = sorted(
        range(n),
        key=lambda i: (float(questions[i].priority), -i),
        reverse=True,
    )
    selected = order[: min(k, n)]
    backfilled = False
    return SelectionResult(
        selected_indices=selected,
        strategy="top_priority",
        relevance_scores=[float(questions[i].priority) for i in selected],
        backfilled=backfilled,
        backfill_count=0,
        notes=[]
        if len(selected) >= min(k, n)
        else [f"pool_shorter_than_k: selected={len(selected)} k={k}"],
    )


def strategy_random(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
    seed: int = 42,
    query_similarities: np.ndarray | None = None,
) -> SelectionResult:
    del similarity_matrix, weights, lambda_, query_similarities
    n = len(questions)
    rng = np.random.default_rng(seed)
    if n == 0:
        return SelectionResult(selected_indices=[], strategy="random")
    take = min(k, n)
    selected = [int(i) for i in rng.choice(n, size=take, replace=False)]
    return SelectionResult(
        selected_indices=selected,
        strategy="random",
        backfilled=False,
        backfill_count=0,
        notes=[] if take >= k else [f"pool_shorter_than_k: selected={take} k={k}"],
    )


SELECTION_STRATEGIES: dict[str, Callable[..., SelectionResult]] = {
    "competency_weighted_mmr": strategy_competency_weighted_mmr,
    "plain_mmr_priority": strategy_plain_mmr_priority,
    "plain_mmr_query": strategy_plain_mmr_query,
    "top_priority": strategy_top_priority,
    "random": strategy_random,
}

# Stage name for the adaptive-lambda / competency-budget grid.
# Distinct from selection_ablation, which the pilot uses.
ADAPTIVE_SELECTION_STAGE = "adaptive_selection"


def _make_adaptive_strategy(config_name: str) -> Callable[..., SelectionResult]:
    def _strategy(
        questions: list[InterviewQuestion],
        similarity_matrix: np.ndarray,
        weights: list[float],
        *,
        lambda_: float,
        k: int,
        seed: int = 42,
        query_similarities: np.ndarray | None = None,
    ) -> SelectionResult:
        del seed, query_similarities
        from dataclasses import replace

        from app.services.mmr.adaptive import named_ablation_grid, select_adaptive

        cfg = named_ablation_grid()[config_name]
        if not cfg.use_adaptive_lambda:
            cfg = replace(cfg, lambda_fixed=float(lambda_))
        dims = [str(q.question_type) for q in questions]
        result = select_adaptive(
            questions,
            similarity_matrix,
            list(weights),
            dims,
            cfg,
            k=k,
        )
        return SelectionResult(
            selected_indices=list(result.selected_indices),
            strategy=f"{ADAPTIVE_SELECTION_STAGE}/{config_name}",
            relevance_scores=list(result.relevance_scores),
            redundancy_scores=list(result.redundancy_scores),
            mmr_scores=list(result.mmr_scores),
            competency_weights=list(result.competency_weights),
            notes=[
                f"stage={ADAPTIVE_SELECTION_STAGE}",
                f"config={config_name}",
                f"shortfalls={len(result.shortfalls)}",
            ],
        )

    _strategy.__name__ = f"strategy_adaptive_{config_name}"
    return _strategy


def register_adaptive_selection_grid() -> None:
    """Register named adaptive configs under stage ``adaptive_selection``."""
    from app.services.mmr.adaptive import named_ablation_grid

    for name in named_ablation_grid():
        SELECTION_STRATEGIES[f"{ADAPTIVE_SELECTION_STAGE}/{name}"] = _make_adaptive_strategy(
            name
        )


register_adaptive_selection_grid()


def select_questions(
    strategy: str,
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
    seed: int = 42,
    query_similarities: np.ndarray | None = None,
) -> SelectionResult:
    if strategy not in SELECTION_STRATEGIES:
        raise KeyError(
            f"Unknown selection strategy {strategy!r}. "
            f"Known: {sorted(SELECTION_STRATEGIES)}"
        )
    return SELECTION_STRATEGIES[strategy](
        questions,
        similarity_matrix,
        weights,
        lambda_=lambda_,
        k=k,
        seed=seed,
        query_similarities=query_similarities,
    )
