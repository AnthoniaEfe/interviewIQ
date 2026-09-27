from __future__ import annotations

from app.config.settings import Settings
from app.schemas.domain import (
    CandidateProfile,
    GapMatrix,
    JobDescription,
    MatchLabel,
    ProfileGap,
    RequirementPriority,
    SimilarityEvidence,
)
from app.services.embeddings.service import EmbeddingService
from app.services.retrieval.bm25 import BM25Matcher
from app.services.retrieval.tfidf import TfidfMatcher


def _candidate_texts(profile: CandidateProfile) -> list[str]:
    extras: list[str] = []
    for skill in (
        profile.technical_skills
        + profile.soft_skills
        + profile.domain_experience
        + profile.certifications
        + profile.competencies
    ):
        extras.append(skill.normalized_label)
    for job in profile.experience:
        extras.extend(job.technologies)
        extras.extend(job.responsibilities)
    seen: set[str] = set()
    out: list[str] = []
    for text in extras:
        key = text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text.strip())
    return out or [""]


def _job_competencies(job: JobDescription) -> list[tuple[str, RequirementPriority]]:
    items: list[tuple[str, RequirementPriority]] = []
    for skill in job.required_skills:
        items.append((skill.normalized_label, RequirementPriority.REQUIRED))
    for skill in job.preferred_skills:
        items.append((skill.normalized_label, RequirementPriority.PREFERRED))
    if items:
        return items
    return [(c.normalized_label, c.priority) for c in job.competencies]


def classify_match(
    *,
    tfidf: float,
    bm25: float,
    semantic: float,
    bm25_max: float,
) -> MatchLabel:
    bm25_norm = bm25 / bm25_max if bm25_max > 0 else 0.0
    combined = 0.25 * tfidf + 0.25 * bm25_norm + 0.50 * semantic
    if combined >= 0.65:
        return MatchLabel.STRONG
    if combined >= 0.40:
        return MatchLabel.PARTIAL
    if combined >= 0.20:
        return MatchLabel.WEAK
    return MatchLabel.MISSING


def compute_weight(
    *,
    priority: RequirementPriority,
    match_label: MatchLabel,
    settings: Settings,
    semantic: float,
    tfidf: float,
    bm25_norm: float,
) -> float:
    if not settings.enable_competency_weighting:
        return 1.0
    base = (
        settings.required_competency_weight
        if priority == RequirementPriority.REQUIRED
        else settings.preferred_competency_weight
    )
    severity = {
        MatchLabel.MISSING: 1.0,
        MatchLabel.WEAK: 0.8,
        MatchLabel.PARTIAL: 0.5,
        MatchLabel.STRONG: 0.25,
    }[match_label]
    evidence = 0.4 * semantic + 0.3 * tfidf + 0.3 * bm25_norm
    return round(base * severity * (0.5 + 0.5 * (1.0 - evidence)), 4)


def build_gap_matrix(
    profile: CandidateProfile,
    job: JobDescription,
    settings: Settings,
    embeddings: EmbeddingService | None = None,
) -> GapMatrix:
    job_items = _job_competencies(job)
    cand_texts = _candidate_texts(profile)
    cand_blob = " ".join(cand_texts)

    tfidf_matcher = TfidfMatcher(min_similarity=settings.tfidf_min_similarity)
    bm25_matcher = BM25Matcher(
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        min_score=settings.bm25_min_score,
    )

    gaps: list[ProfileGap] = []
    bm25_raw: list[float] = []
    rows: list[dict] = []
    semantic_by_label: dict[str, float] = {}
    if settings.enable_semantic and embeddings is not None:
        job_labels = [label for label, _ in job_items]
        valid_cand = [text for text in cand_texts if text]
        if job_labels and valid_cand:
            sim = embeddings.similarity_matrix(job_labels, valid_cand)
            for i, label in enumerate(job_labels):
                semantic_by_label[label] = float(sim[i].max()) if sim.shape[1] else 0.0

    for label, priority in job_items:
        tfidf_score = 0.0
        matched_terms: list[str] = []
        if settings.enable_tfidf:
            tfidf_score, matched_terms = tfidf_matcher.similarity(cand_blob, label)

        bm25_score = 0.0
        best_idx: int | None = None
        if settings.enable_bm25:
            best_idx, bm25_score = bm25_matcher.best_match(label, cand_texts)

        semantic_score = semantic_by_label.get(label, 0.0)

        matched_candidate = cand_texts[best_idx] if best_idx is not None and cand_texts else None
        bm25_raw.append(bm25_score)
        rows.append(
            {
                "label": label,
                "priority": priority,
                "tfidf": tfidf_score,
                "terms": matched_terms,
                "bm25": bm25_score,
                "semantic": semantic_score,
                "matched": matched_candidate,
            }
        )

    bm25_max = max(bm25_raw) if bm25_raw else 0.0
    for row in rows:
        bm25_norm = row["bm25"] / bm25_max if bm25_max > 0 else 0.0
        if settings.enable_gap_analysis:
            match_label = classify_match(
                tfidf=row["tfidf"],
                bm25=row["bm25"],
                semantic=row["semantic"],
                bm25_max=bm25_max,
            )
        else:
            match_label = MatchLabel.PARTIAL
        weight = compute_weight(
            priority=row["priority"],
            match_label=match_label,
            settings=settings,
            semantic=row["semantic"],
            tfidf=row["tfidf"],
            bm25_norm=bm25_norm,
        )
        gaps.append(
            ProfileGap(
                job_competency=row["label"],
                priority=row["priority"],
                match_label=match_label,
                weight=weight,
                evidence=SimilarityEvidence(
                    tfidf_score=row["tfidf"],
                    tfidf_matched_terms=row["terms"],
                    bm25_score=row["bm25"],
                    semantic_score=row["semantic"],
                    matched_candidate_competency=row["matched"],
                ),
                candidate_evidence=[row["matched"]] if row["matched"] else [],
            )
        )

    return GapMatrix(
        gaps=gaps,
        strong_matches=[g.job_competency for g in gaps if g.match_label == MatchLabel.STRONG],
        partial_matches=[g.job_competency for g in gaps if g.match_label == MatchLabel.PARTIAL],
        weak_matches=[g.job_competency for g in gaps if g.match_label == MatchLabel.WEAK],
        missing=[g.job_competency for g in gaps if g.match_label == MatchLabel.MISSING],
    )
