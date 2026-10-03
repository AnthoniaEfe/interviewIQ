"""LLM-free gap-lite features (not the production extracted matrix)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config.settings import Settings
from app.schemas.domain import JobDescription, NormalizedCompetency, RequirementPriority
from app.services.embeddings.service import EmbeddingService
from app.services.gap_sensitivity.chunking import chunk_cv_text
from app.services.gap_sensitivity.scoring import score_requirements
from app.schemas.domain import MatchLabel


FEATURE_NAMES = [
    "n_req_units",
    "share_strong",
    "share_partial",
    "share_gap",
    "mean_combined",
    "median_combined",
    "min_combined",
    "max_combined",
    "p10_combined",
    "p25_combined",
    "mean_top5_combined",
    "mean_semantic",
    "cv_words",
    "jd_words",
    "jaccard",
]


def _words(text: str) -> list[str]:
    return [w for w in str(text).lower().split() if w]


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_words(a)), set(_words(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def gap_lite_vector(
    resume: str,
    job: str,
    settings: Settings,
    embeddings: EmbeddingService,
) -> np.ndarray:
    reqs = chunk_cv_text(job, min_words=4, max_words=60)
    evid = chunk_cv_text(resume, min_words=4, max_words=60)
    n_req = len(reqs)
    if n_req == 0:
        reqs = [job[:500] or "job"]
        n_req = 1
    jd = JobDescription(
        required_skills=[
            NormalizedCompetency(normalized_label=r, priority=RequirementPriority.REQUIRED)
            for r in reqs
        ]
    )
    scored = score_requirements(jd, evid or [""], settings, embeddings)
    comb = np.asarray([s.combined for s in scored.scores], dtype=float)
    sem = np.asarray([s.semantic for s in scored.scores], dtype=float)
    labels = [s.match_label for s in scored.scores]
    n = max(len(labels), 1)
    n_strong = sum(1 for x in labels if x == MatchLabel.STRONG)
    n_part = sum(1 for x in labels if x == MatchLabel.PARTIAL)
    n_gap = n - n_strong - n_part
    top5 = np.sort(comb)[-min(5, comb.size) :]
    vec = np.asarray(
        [
            float(n_req),
            n_strong / n,
            n_part / n,
            n_gap / n,
            float(np.mean(comb)) if comb.size else 0.0,
            float(np.median(comb)) if comb.size else 0.0,
            float(np.min(comb)) if comb.size else 0.0,
            float(np.max(comb)) if comb.size else 0.0,
            float(np.percentile(comb, 10)) if comb.size else 0.0,
            float(np.percentile(comb, 25)) if comb.size else 0.0,
            float(np.mean(top5)) if top5.size else 0.0,
            float(np.mean(sem)) if sem.size else 0.0,
            float(len(_words(resume))),
            float(len(_words(job))),
            _jaccard(resume, job),
        ],
        dtype=np.float32,
    )
    return vec


def gap_lite_matrix(
    resumes: list[str],
    jobs: list[str],
    settings: Settings,
    embeddings: EmbeddingService,
) -> np.ndarray:
    rows = []
    n = len(resumes)
    for i, (r, j) in enumerate(zip(resumes, jobs, strict=True)):
        if i % 50 == 0:
            print(f"[gap-lite] {i}/{n}", flush=True)
        rows.append(gap_lite_vector(r, j, settings, embeddings))
    return np.vstack(rows)
