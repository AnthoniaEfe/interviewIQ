"""Gap scoring against an explicit candidate evidence string list."""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import Settings
from app.schemas.domain import (
    JobDescription,
    MatchLabel,
    RequirementPriority,
)
from app.services.embeddings.service import EmbeddingService
from app.services.gap_analysis.matrix import _job_competencies, classify_match
from app.services.retrieval.bm25 import BM25Matcher
from app.services.retrieval.tfidf import TfidfMatcher

WEIGHT_TFIDF = 0.25
WEIGHT_BM25 = 0.25
WEIGHT_SEMANTIC = 0.50
THRESH_STRONG = 0.65
THRESH_PARTIAL = 0.40
THRESH_WEAK = 0.20


@dataclass(frozen=True)
class RequirementScore:
    job_competency: str
    priority: RequirementPriority
    tfidf: float
    bm25_raw: float
    bm25_norm: float
    bm25_norm_fixed: float | None
    semantic: float
    combined: float
    combined_fixed_bm25: float | None
    match_label: MatchLabel
    match_label_fixed_bm25: MatchLabel | None
    match_label_semantic_only: MatchLabel
    matched_candidate: str | None


@dataclass(frozen=True)
class PairScoreResult:
    scores: list[RequirementScore]
    bm25_max: float
    evidence_size: int
    evidence_texts: list[str]


def _label_from_combined(combined: float) -> MatchLabel:
    if combined >= THRESH_STRONG:
        return MatchLabel.STRONG
    if combined >= THRESH_PARTIAL:
        return MatchLabel.PARTIAL
    if combined >= THRESH_WEAK:
        return MatchLabel.WEAK
    return MatchLabel.MISSING


def _safe_bm25_best(
    matcher: BM25Matcher, query: str, documents: list[str]
) -> tuple[int | None, float]:
    """BM25 best-match that tolerates empty-token corpora."""
    try:
        return matcher.best_match(query, documents)
    except ZeroDivisionError:
        return None, 0.0


def score_requirements(
    job: JobDescription,
    cand_texts: list[str],
    settings: Settings,
    embeddings: EmbeddingService | None = None,
    *,
    bm25_max_fixed: float | None = None,
) -> PairScoreResult:
    """Score job requirements against a candidate evidence string list.

    BM25 normalisation:
    - ``bm25_norm`` uses max raw BM25 within this (job, evidence) pair — same as
      production ``build_gap_matrix``. This scale depends on the evidence set.
    - ``bm25_norm_fixed`` / ``combined_fixed_bm25`` optionally divide by
      ``bm25_max_fixed`` (typically Condition A's per-pair max) so lexical
      scores are more comparable across conditions. Semantic-only labels are
      always returned as a scale-free robustness check.
    """
    job_items = _job_competencies(job)
    texts = list(cand_texts) if cand_texts else [""]
    cand_blob = " ".join(t for t in texts if t)

    tfidf_matcher = TfidfMatcher(min_similarity=settings.tfidf_min_similarity)
    bm25_matcher = BM25Matcher(
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        min_score=settings.bm25_min_score,
    )

    semantic_by_label: dict[str, float] = {}
    best_sem_idx: dict[str, int | None] = {}
    if settings.enable_semantic and embeddings is not None:
        job_labels = [label for label, _ in job_items]
        valid_cand = [text for text in texts if text]
        if job_labels and valid_cand:
            sim = embeddings.similarity_matrix(job_labels, valid_cand)
            for i, label in enumerate(job_labels):
                if sim.shape[1] == 0:
                    semantic_by_label[label] = 0.0
                    best_sem_idx[label] = None
                else:
                    j = int(sim[i].argmax())
                    semantic_by_label[label] = float(sim[i, j])
                    best_sem_idx[label] = j

    rows: list[dict] = []
    for label, priority in job_items:
        tfidf_score = 0.0
        if settings.enable_tfidf:
            tfidf_score, _ = tfidf_matcher.similarity(cand_blob, label)

        bm25_score = 0.0
        best_idx: int | None = None
        if settings.enable_bm25:
            best_idx, bm25_score = _safe_bm25_best(bm25_matcher, label, texts)

        semantic_score = semantic_by_label.get(label, 0.0)
        matched = None
        if best_idx is not None and 0 <= best_idx < len(texts) and texts[best_idx]:
            matched = texts[best_idx]
        elif best_sem_idx.get(label) is not None:
            idx = best_sem_idx[label]
            if idx is not None and texts[idx]:
                matched = texts[idx]

        rows.append(
            {
                "label": label,
                "priority": priority,
                "tfidf": float(tfidf_score),
                "bm25": float(bm25_score),
                "semantic": float(semantic_score),
                "matched": matched,
            }
        )

    bm25_max = max((r["bm25"] for r in rows), default=0.0)
    scores: list[RequirementScore] = []
    for row in rows:
        bm25_norm = row["bm25"] / bm25_max if bm25_max > 0 else 0.0
        combined = (
            WEIGHT_TFIDF * row["tfidf"]
            + WEIGHT_BM25 * bm25_norm
            + WEIGHT_SEMANTIC * row["semantic"]
        )
        if settings.enable_gap_analysis:
            # Production path (identical formula to classify_match).
            match_label = classify_match(
                tfidf=row["tfidf"],
                bm25=row["bm25"],
                semantic=row["semantic"],
                bm25_max=bm25_max,
            )
        else:
            match_label = MatchLabel.PARTIAL

        bm25_norm_fixed: float | None = None
        combined_fixed: float | None = None
        label_fixed: MatchLabel | None = None
        if bm25_max_fixed is not None:
            bm25_norm_fixed = (
                row["bm25"] / bm25_max_fixed if bm25_max_fixed > 0 else 0.0
            )
            combined_fixed = (
                WEIGHT_TFIDF * row["tfidf"]
                + WEIGHT_BM25 * bm25_norm_fixed
                + WEIGHT_SEMANTIC * row["semantic"]
            )
            label_fixed = _label_from_combined(combined_fixed)

        scores.append(
            RequirementScore(
                job_competency=row["label"],
                priority=row["priority"],
                tfidf=row["tfidf"],
                bm25_raw=row["bm25"],
                bm25_norm=bm25_norm,
                bm25_norm_fixed=bm25_norm_fixed,
                semantic=row["semantic"],
                combined=combined,
                combined_fixed_bm25=combined_fixed,
                match_label=match_label,
                match_label_fixed_bm25=label_fixed,
                match_label_semantic_only=_label_from_combined(row["semantic"]),
                matched_candidate=row["matched"],
            )
        )

    evidence_size = len([t for t in texts if t])
    return PairScoreResult(
        scores=scores,
        bm25_max=bm25_max,
        evidence_size=evidence_size,
        evidence_texts=texts,
    )


def is_gap_label(label: MatchLabel) -> bool:
    return label in {MatchLabel.WEAK, MatchLabel.MISSING}


def three_category(label: MatchLabel) -> str:
    if label == MatchLabel.STRONG:
        return "strength"
    if label == MatchLabel.PARTIAL:
        return "partial"
    return "gap"
