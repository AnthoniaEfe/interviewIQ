"""Score pairs under Condition A / D / semantic-only; production gap JSON; mismatch."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription, MatchLabel
from app.services.embeddings.service import EmbeddingService
from app.services.fit_label_control.constants import PRIORITY_WEIGHT, SEED, SENIORITY
from app.services.fit_label_control.mismatch import mismatched_partner_indices_by_label
from app.services.gap_analysis.matrix import build_gap_matrix
from app.services.gap_sensitivity.analyze import EXPECTED_COUNTS, regression_check
from app.services.gap_sensitivity.evidence import CONDITION_A, CONDITION_D, build_evidence
from app.services.gap_sensitivity.io_utils import write_json
from app.services.gap_sensitivity.metrics import gap_share
from app.services.gap_sensitivity.scoring import (
    PairScoreResult,
    is_gap_label,
    score_requirements,
)


def score_pair(
    record: dict,
    cv_text: str | None,
    condition: str,
    settings: Settings,
    embeddings: EmbeddingService,
    *,
    bm25_max_fixed: float | None = None,
) -> PairScoreResult:
    candidate = CandidateProfile.model_validate(record["candidate"])
    job = JobDescription.model_validate(record["job"])
    evidence = build_evidence(condition, candidate, cv_text)
    return score_requirements(
        job, evidence, settings, embeddings, bm25_max_fixed=bm25_max_fixed
    )


def write_production_gap(
    record: dict,
    *,
    settings: Settings,
    embeddings: EmbeddingService,
    experiment_id: str,
    source_experiment: str,
    out_dir: Path,
) -> dict:
    """Same path as ``experiments.run.run_gap_combo`` (``build_gap_matrix``)."""
    candidate = CandidateProfile.model_validate(record["candidate"])
    job = JobDescription.model_validate(record["job"])
    started = datetime.now(timezone.utc)
    matrix = build_gap_matrix(candidate, job, settings, embeddings)
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    three = {
        "strength_matches": matrix.strong_matches,
        "partial_matches": matrix.partial_matches,
        "competency_gaps": matrix.weak_matches + matrix.missing,
    }
    result = {
        "experiment_id": experiment_id,
        "stage": "gap_analysis",
        "source_experiment": source_experiment,
        "profile_id": record.get("profile_id"),
        "jd_id": record.get("jd_id"),
        "pair_id": record.get("pair_id"),
        "fit_label": record.get("fit_label"),
        "dataset": record.get("dataset"),
        "seniority": record.get("seniority", SENIORITY),
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "signals": {
            "tfidf": settings.enable_tfidf,
            "bm25": settings.enable_bm25,
            "semantic": settings.enable_semantic,
        },
        "gap_matrix": matrix.model_dump(),
        "three_category": three,
        "counts": {
            "requirements": len(matrix.gaps),
            "strength_matches": len(three["strength_matches"]),
            "partial_matches": len(three["partial_matches"]),
            "competency_gaps": len(three["competency_gaps"]),
        },
        "latency_ms": round(elapsed_ms, 2),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    rid = (
        f"{record.get('profile_id')}__{record.get('jd_id')}__"
        f"{record.get('seniority', SENIORITY)}"
    )
    write_json(out_dir / f"{rid}.json", result)
    return result


def _labels(pr: PairScoreResult, *, semantic_only: bool) -> list[MatchLabel]:
    if semantic_only:
        return [s.match_label_semantic_only for s in pr.scores]
    return [s.match_label for s in pr.scores]


def pair_row(
    record: dict,
    pr: PairScoreResult,
    *,
    cv_text: str | None,
    jd_text: str | None,
    semantic_only: bool,
    mismatch_gap_rate: float | None = None,
) -> dict[str, Any]:
    labs = _labels(pr, semantic_only=semantic_only)
    n = len(labs) or 1
    n_strong = sum(1 for x in labs if x == MatchLabel.STRONG)
    n_partial = sum(1 for x in labs if x == MatchLabel.PARTIAL)
    n_weak = sum(1 for x in labs if x == MatchLabel.WEAK)
    n_missing = sum(1 for x in labs if x == MatchLabel.MISSING)
    n_gap = n_weak + n_missing
    weights = [PRIORITY_WEIGHT.get(s.priority.value, 0.8) for s in pr.scores]
    w_gap = [
        w
        for w, s in zip(weights, pr.scores, strict=True)
        if is_gap_label(
            s.match_label_semantic_only if semantic_only else s.match_label
        )
    ]
    n_req_pri = sum(1 for s in pr.scores if s.priority.value == "required")
    n_pref = sum(1 for s in pr.scores if s.priority.value == "preferred")
    comb = [s.semantic if semantic_only else s.combined for s in pr.scores]
    sem = [s.semantic for s in pr.scores]
    tfidf = [s.tfidf for s in pr.scores]
    bm25n = [s.bm25_norm for s in pr.scores]
    return {
        "pair_id": record.get("pair_id"),
        "profile_id": record.get("profile_id"),
        "jd_id": record.get("jd_id"),
        "fit_label": record.get("fit_label"),
        "n_requirements": len(pr.scores),
        "n_strong": n_strong,
        "n_partial": n_partial,
        "n_weak": n_weak,
        "n_missing": n_missing,
        "strong_share": n_strong / n,
        "partial_share": n_partial / n,
        "weak_share": n_weak / n,
        "missing_share": n_missing / n,
        "strength_share": n_strong / n,
        "gap_share": n_gap / n,
        "gap_rate": n_gap / n,
        "priority_weighted_gap_share": (
            (sum(w_gap) / sum(weights)) if sum(weights) else None
        ),
        "mean_combined": float(np.mean(comb)) if comb else None,
        "mean_semantic": float(np.mean(sem)) if sem else None,
        "mean_tfidf": float(np.mean(tfidf)) if tfidf else None,
        "mean_bm25_norm": float(np.mean(bm25n)) if bm25n else None,
        "evidence_strings": pr.evidence_size,
        "jd_length": len(jd_text) if jd_text is not None else None,
        "cv_length": len(cv_text) if cv_text is not None else None,
        "n_required": n_req_pri,
        "n_preferred": n_pref,
        "share_required": (n_req_pri / len(pr.scores)) if pr.scores else None,
        "mismatch_gap_rate": mismatch_gap_rate,
        "true_minus_mismatch_gap": (
            (n_gap / n - mismatch_gap_rate) if mismatch_gap_rate is not None else None
        ),
        "jd_text_hash": None,
    }


def score_records(
    records: list[dict],
    cv_texts: list[str | None],
    jd_texts: list[str | None],
    settings: Settings,
    embeddings: EmbeddingService,
) -> dict[str, Any]:
    a_results = [
        score_pair(r, cv, CONDITION_A, settings, embeddings)
        for r, cv in zip(records, cv_texts, strict=True)
    ]
    d_results = [
        score_pair(
            r,
            cv,
            CONDITION_D,
            settings,
            embeddings,
            bm25_max_fixed=a.bm25_max,
        )
        for r, cv, a in zip(records, cv_texts, a_results, strict=True)
    ]
    labels = [str(r.get("fit_label") or "unknown") for r in records]
    if len(records) >= 2 and all(
        labels.count(lab) >= 2 for lab in set(labels)
    ):
        partners = mismatched_partner_indices_by_label(
            labels, seed=SEED, n_permutations=5
        )
        evidence_a = [
            build_evidence(
                CONDITION_A,
                CandidateProfile.model_validate(records[i]["candidate"]),
                cv_texts[i],
            )
            for i in range(len(records))
        ]
        n_perm, n = partners.shape
        mm_rates = np.zeros(n, dtype=np.float64)
        for p in range(n_perm):
            for i in range(n):
                j = int(partners[p, i])
                job = JobDescription.model_validate(records[i]["job"])
                scored = score_requirements(
                    job,
                    evidence_a[j],
                    settings,
                    embeddings,
                    bm25_max_fixed=a_results[i].bm25_max,
                )
                mm_rates[i] += float(gap_share(_labels(scored, semantic_only=False)) or 0.0)
        mm_rates /= n_perm
    else:
        partners = np.zeros((0, len(records)), dtype=np.int64)
        mm_rates = np.full(len(records), np.nan)

    rows_a = [
        pair_row(
            rec,
            pr,
            cv_text=cv,
            jd_text=jd,
            semantic_only=False,
            mismatch_gap_rate=float(mm_rates[i]) if np.isfinite(mm_rates[i]) else None,
        )
        for i, (rec, pr, cv, jd) in enumerate(
            zip(records, a_results, cv_texts, jd_texts, strict=True)
        )
    ]
    rows_d = [
        pair_row(rec, pr, cv_text=cv, jd_text=jd, semantic_only=False)
        for rec, pr, cv, jd in zip(records, d_results, cv_texts, jd_texts, strict=True)
    ]
    rows_sem = [
        pair_row(rec, pr, cv_text=cv, jd_text=jd, semantic_only=True)
        for rec, pr, cv, jd in zip(records, a_results, cv_texts, jd_texts, strict=True)
    ]
    return {
        "A": {"results": a_results, "rows": rows_a},
        "D": {"results": d_results, "rows": rows_d},
        "semantic": {"results": a_results, "rows": rows_sem},
        "mismatch_partners": partners,
        "mismatch_gap_rates": mm_rates,
    }


def regression_fitpos(
    records: list[dict],
    stored_gaps: dict[str, dict],
    settings: Settings,
    embeddings: EmbeddingService,
    split: str,
) -> dict[str, Any]:
    reg = regression_check(records, stored_gaps, settings, embeddings)
    expected = EXPECTED_COUNTS.get(split)
    counts_match = expected is not None and reg.get("counts") == expected
    reg["expected_counts"] = expected
    reg["counts_match_expected"] = counts_match
    reg["split"] = split
    return reg
