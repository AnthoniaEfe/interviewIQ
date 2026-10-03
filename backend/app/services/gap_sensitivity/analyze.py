"""Orchestrate gap-sensitivity analysis for one split (and helpers for combine)."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription, MatchLabel
from app.services.embeddings.service import EmbeddingService, build_embedding_service
from app.services.gap_sensitivity.controls import mismatched_partner_indices
from app.services.gap_sensitivity.evidence import (
    CONDITION_A,
    CONDITION_DESCRIPTIONS,
    MAIN_CONDITIONS,
    build_evidence,
)
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    load_extraction_records,
    load_gap_records,
    project_root,
    request_id,
    resolve_cv_text,
    write_json,
)
from app.services.gap_sensitivity.metrics import (
    bootstrap_mean_ci,
    bootstrap_pooled_gap_share_ci,
    gap_share,
    label_counts,
    per_pair_gap_rates,
    spearman_dose_response,
    summarise_rates,
    three_category_counts,
    transition_matrix,
    wilcoxon_paired,
)
from app.services.gap_sensitivity.scoring import (
    PairScoreResult,
    score_requirements,
    three_category,
)

SEED = 42
BOOTSTRAP_RESAMPLES = 2000
MISMATCH_PERMS = 5
SCORE_TOL = 1e-4

EXPECTED_COUNTS = {
    "primary-30": {
        "requirements": 633,
        "strong_match": 51,
        "partial_match": 175,
        "weak_match": 245,
        "missing": 162,
    },
    "holdout-30": {
        "requirements": 515,
        "strong_match": 33,
        "partial_match": 108,
        "weak_match": 206,
        "missing": 168,
    },
}


def _split_name(config: dict) -> str:
    name = (config.get("experiment_name") or "").lower()
    if "holdout" in name:
        return "holdout-30"
    return "primary-30"


def regression_check(
    records: list[dict],
    stored_gaps: dict[str, dict],
    settings: Settings,
    embeddings: EmbeddingService,
) -> dict[str, Any]:
    """Replay Condition A (production evidence) and compare to stored gap JSONs."""
    label_tally: dict[str, int] = defaultdict(int)
    n_req = 0
    n_label_mismatch = 0
    n_score_mismatch = 0
    max_abs_diff = 0.0
    pair_results: list[dict] = []

    for record in records:
        rid = request_id(record) + ".json"
        if rid not in stored_gaps:
            return {
                "passed": False,
                "error": f"Missing stored gap artifact for {rid}",
            }
        stored = stored_gaps[rid]
        candidate = CandidateProfile.model_validate(record["candidate"])
        job = JobDescription.model_validate(record["job"])
        evidence = build_evidence(CONDITION_A, candidate, None)
        scored = score_requirements(job, evidence, settings, embeddings)
        stored_list = stored["gap_matrix"]["gaps"]
        if len(scored.scores) != len(stored_list):
            return {
                "passed": False,
                "error": (
                    f"Requirement count mismatch for {rid}: "
                    f"{len(scored.scores)} vs {len(stored_list)}"
                ),
            }
        stored_bm25 = [float(g["evidence"]["bm25_score"] or 0.0) for g in stored_list]
        smax = max(stored_bm25) if stored_bm25 else 0.0
        for sc, sg in zip(scored.scores, stored_list, strict=True):
            n_req += 1
            label_tally[sc.match_label.value] += 1
            if sc.match_label.value != sg["match_label"]:
                n_label_mismatch += 1
            ev = sg["evidence"]
            sb = (float(ev["bm25_score"] or 0.0) / smax) if smax > 0 else 0.0
            scomb = (
                0.25 * float(ev["tfidf_score"] or 0.0)
                + 0.25 * sb
                + 0.50 * float(ev["semantic_score"] or 0.0)
            )
            diff = abs(sc.combined - scomb)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > SCORE_TOL:
                n_score_mismatch += 1
        pair_results.append(
            {
                "request_id": request_id(record),
                "n_requirements": len(scored.scores),
                "evidence_size": scored.evidence_size,
            }
        )

    counts = {
        "requirements": n_req,
        "strong_match": int(label_tally.get("strong_match", 0)),
        "partial_match": int(label_tally.get("partial_match", 0)),
        "weak_match": int(label_tally.get("weak_match", 0)),
        "missing": int(label_tally.get("missing", 0)),
    }
    return {
        "passed": n_label_mismatch == 0 and n_score_mismatch == 0,
        "counts": counts,
        "n_label_mismatch": n_label_mismatch,
        "n_score_mismatch": n_score_mismatch,
        "max_abs_combined_diff": max_abs_diff,
        "tolerance": SCORE_TOL,
        "pairs": pair_results,
    }


def _score_all_pairs(
    *,
    records: list[dict],
    cv_texts: list[str | None],
    conditions: tuple[str, ...],
    settings: Settings,
    embeddings: EmbeddingService,
) -> dict[str, list[PairScoreResult]]:
    """Score every pair under every condition (true matching)."""
    # First pass: Condition A to capture per-pair bm25_max for fixed-norm variant.
    a_results: list[PairScoreResult] = []
    for record, cv in zip(records, cv_texts, strict=True):
        candidate = CandidateProfile.model_validate(record["candidate"])
        job = JobDescription.model_validate(record["job"])
        evidence = build_evidence(CONDITION_A, candidate, cv)
        a_results.append(score_requirements(job, evidence, settings, embeddings))

    out: dict[str, list[PairScoreResult]] = {CONDITION_A: a_results}
    for cond in conditions:
        if cond == CONDITION_A:
            continue
        cond_results: list[PairScoreResult] = []
        for i, (record, cv) in enumerate(zip(records, cv_texts, strict=True)):
            candidate = CandidateProfile.model_validate(record["candidate"])
            job = JobDescription.model_validate(record["job"])
            evidence = build_evidence(cond, candidate, cv)
            cond_results.append(
                score_requirements(
                    job,
                    evidence,
                    settings,
                    embeddings,
                    bm25_max_fixed=a_results[i].bm25_max,
                )
            )
        out[cond] = cond_results
    return out


def _score_mismatched(
    *,
    records: list[dict],
    cv_texts: list[str | None],
    true_results: dict[str, list[PairScoreResult]],
    conditions: tuple[str, ...],
    settings: Settings,
    embeddings: EmbeddingService,
    seed: int,
    n_permutations: int,
) -> dict[str, list[list[PairScoreResult]]]:
    """For each condition, list over permutations of per-pair mismatched scores."""
    n = len(records)
    partners = mismatched_partner_indices(n, seed=seed, n_permutations=n_permutations)
    # Precompute evidence sets per condition per pair
    evidence_by_cond: dict[str, list[list[str]]] = {}
    for cond in conditions:
        evidence_by_cond[cond] = [
            build_evidence(
                cond,
                CandidateProfile.model_validate(records[i]["candidate"]),
                cv_texts[i],
            )
            for i in range(n)
        ]

    out: dict[str, list[list[PairScoreResult]]] = {c: [] for c in conditions}
    for p in range(n_permutations):
        for cond in conditions:
            perm_results: list[PairScoreResult] = []
            for i in range(n):
                j = int(partners[p, i])
                job = JobDescription.model_validate(records[i]["job"])
                evidence = evidence_by_cond[cond][j]
                bm25_fixed = true_results[CONDITION_A][i].bm25_max
                perm_results.append(
                    score_requirements(
                        job,
                        evidence,
                        settings,
                        embeddings,
                        bm25_max_fixed=bm25_fixed,
                    )
                )
            out[cond].append(perm_results)
    return out


def _pair_labels(results: list[PairScoreResult], *, semantic_only: bool = False) -> list[list[MatchLabel]]:
    out = []
    for pr in results:
        if semantic_only:
            out.append([s.match_label_semantic_only for s in pr.scores])
        else:
            out.append([s.match_label for s in pr.scores])
    return out


def _flat_labels(results: list[PairScoreResult], *, semantic_only: bool = False) -> list[MatchLabel]:
    labels: list[MatchLabel] = []
    for pr in results:
        for s in pr.scores:
            labels.append(s.match_label_semantic_only if semantic_only else s.match_label)
    return labels


def _mean_signals(results: list[PairScoreResult]) -> dict[str, float | None]:
    tfidf, bm25n, sem, comb = [], [], [], []
    for pr in results:
        for s in pr.scores:
            tfidf.append(s.tfidf)
            bm25n.append(s.bm25_norm)
            sem.append(s.semantic)
            comb.append(s.combined)
    if not comb:
        return {"tfidf": None, "bm25_norm": None, "semantic": None, "combined": None}
    return {
        "tfidf": float(np.mean(tfidf)),
        "bm25_norm": float(np.mean(bm25n)),
        "semantic": float(np.mean(sem)),
        "combined": float(np.mean(comb)),
    }


def _evidence_size_stats(results: list[PairScoreResult]) -> dict[str, float | None]:
    sizes = np.asarray([r.evidence_size for r in results], dtype=np.float64)
    if sizes.size == 0:
        return {"mean": None, "median": None, "n": 0}
    return {
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "n": int(sizes.size),
    }


def _condition_metrics(
    results: list[PairScoreResult],
    *,
    semantic_only: bool = False,
) -> dict[str, Any]:
    pair_labs = _pair_labels(results, semantic_only=semantic_only)
    flat = _flat_labels(results, semantic_only=semantic_only)
    rates = per_pair_gap_rates(pair_labs)
    return {
        "label_counts": label_counts(flat),
        "three_category_counts": three_category_counts(flat),
        "pooled_gap_share": gap_share(flat),
        "pooled_gap_share_bootstrap": bootstrap_pooled_gap_share_ci(
            pair_labs, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
        "per_pair_gap_rate": summarise_rates(rates),
        "per_pair_gap_rate_bootstrap": bootstrap_mean_ci(
            rates, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
        "evidence_size": _evidence_size_stats(results),
        "mean_signals": None if semantic_only else _mean_signals(results),
        "dose_response": spearman_dose_response(
            np.asarray([r.evidence_size for r in results], dtype=np.float64),
            rates,
        ),
    }


def _breakdown(
    records: list[dict],
    results: list[PairScoreResult],
) -> dict[str, Any]:
    by_fit: dict[str, list[MatchLabel]] = defaultdict(list)
    by_priority: dict[str, list[MatchLabel]] = defaultdict(list)
    for record, pr in zip(records, results, strict=True):
        fit = record.get("fit_label") or "unknown"
        for s in pr.scores:
            by_fit[str(fit)].append(s.match_label)
            by_priority[s.priority.value].append(s.match_label)
    return {
        "by_fit_label": {
            k: {
                "label_counts": label_counts(v),
                "three_category_counts": three_category_counts(v),
                "pooled_gap_share": gap_share(v),
                "n_requirements": len(v),
            }
            for k, v in sorted(by_fit.items())
        },
        "by_priority": {
            k: {
                "label_counts": label_counts(v),
                "three_category_counts": three_category_counts(v),
                "pooled_gap_share": gap_share(v),
                "n_requirements": len(v),
            }
            for k, v in sorted(by_priority.items())
        },
    }


def _avg_mismatch_gap_rates(
    mismatch_results: list[list[PairScoreResult]],
) -> tuple[np.ndarray, float | None]:
    """Average per-pair gap rate across permutations; also pooled gap share avg."""
    if not mismatch_results:
        return np.asarray([]), None
    n_pairs = len(mismatch_results[0])
    acc = np.zeros(n_pairs, dtype=np.float64)
    pooled_shares = []
    for perm in mismatch_results:
        labs = _pair_labels(perm)
        acc += per_pair_gap_rates(labs)
        flat = _flat_labels(perm)
        pooled_shares.append(gap_share(flat))
    acc /= len(mismatch_results)
    pooled_vals = [p for p in pooled_shares if p is not None]
    pooled = float(np.mean(pooled_vals)) if pooled_vals else None
    return acc, pooled


def _write_pair_json(
    out_dir: Path,
    record: dict,
    condition_results: dict[str, PairScoreResult],
    experiment_id: str,
    source_experiment: str,
) -> str:
    payload = {
        "experiment_id": experiment_id,
        "stage": "gap_sensitivity",
        "source_experiment": source_experiment,
        "profile_id": record.get("profile_id"),
        "jd_id": record.get("jd_id"),
        "pair_id": record.get("pair_id"),
        "fit_label": record.get("fit_label"),
        "seniority": record.get("seniority"),
        "conditions": {},
    }
    for cond, pr in condition_results.items():
        payload["conditions"][cond] = {
            "evidence_size": pr.evidence_size,
            "bm25_max": pr.bm25_max,
            "requirements": [
                {
                    "job_competency": s.job_competency,
                    "priority": s.priority.value,
                    "match_label": s.match_label.value,
                    "match_label_semantic_only": s.match_label_semantic_only.value,
                    "tfidf": s.tfidf,
                    "bm25_raw": s.bm25_raw,
                    "bm25_norm": s.bm25_norm,
                    "bm25_norm_fixed": s.bm25_norm_fixed,
                    "semantic": s.semantic,
                    "combined": s.combined,
                    "combined_fixed_bm25": s.combined_fixed_bm25,
                    "matched_candidate": s.matched_candidate,
                    "three_category": three_category(s.match_label),
                }
                for s in pr.scores
            ],
        }
    fname = f"{request_id(record)}.json"
    write_json(out_dir / "pairs" / fname, payload)
    return fname


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_gap_sensitivity(
    *,
    settings: Settings,
    config: dict,
    config_path: Path,
    experiment_id: str,
    out_dir: Path,
    source_experiment: str,
    stored_gap_experiment: str | None = None,
    smoke: bool = False,
    conditions: tuple[str, ...] = MAIN_CONDITIONS,
    embeddings: EmbeddingService | None = None,
) -> Path:
    """Run full gap-sensitivity analysis for one config/split."""
    root = project_root()
    source_dir = root / "experiments" / "results" / source_experiment
    if not (source_dir / "extraction").is_dir():
        raise SystemExit(f"Extraction artifacts not found in {source_dir}")

    split = _split_name(config)
    if stored_gap_experiment is None:
        stored_gap_experiment = {
            "primary-30": "exp_20260927T215232Z",
            "holdout-30": "exp_20260927T223303Z",
        }.get(split)
    if not stored_gap_experiment:
        raise SystemExit(f"No stored gap experiment configured for split {split}")

    stored_gap_dir = (
        root / "experiments" / "results" / stored_gap_experiment / "gap_analysis"
    )
    if not stored_gap_dir.is_dir():
        raise SystemExit(f"Stored gap artifacts not found in {stored_gap_dir}")

    records = load_extraction_records(source_dir)
    if smoke:
        records = records[:2]
    stored_gaps = load_gap_records(stored_gap_dir)

    pairs_dir = root / config.get("data", {}).get(
        "pairs_dir", "experiments/datasets/resume_job_fit/experiment_pairs"
    )
    cv_texts = [resolve_cv_text(r, pairs_dir) for r in records]
    missing_cv = sum(1 for t in cv_texts if t is None)
    if missing_cv:
        print(f"[gap_sensitivity] WARNING: {missing_cv} pairs missing raw CV text")

    if embeddings is None:
        print(
            f"[gap_sensitivity] loading embeddings {settings.embedding_model} "
            f"device={settings.embedding_device}"
        )
        embeddings = build_embedding_service(
            settings.embedding_model,
            settings.embedding_device,
        )

    print(f"[gap_sensitivity] regression check (Condition A = production evidence)")
    regression = regression_check(records, stored_gaps, settings, embeddings)
    expected = EXPECTED_COUNTS.get(split)
    counts_match = expected is not None and regression.get("counts") == expected
    regression["expected_counts"] = expected
    regression["counts_match_expected"] = counts_match
    regression["split"] = split
    if not regression["passed"] or (expected and not counts_match and not smoke):
        summary = {
            "experiment_id": experiment_id,
            "status": "failed_regression",
            "stage": "gap_sensitivity",
            "split": split,
            "source_experiment": source_experiment,
            "stored_gap_experiment": stored_gap_experiment,
            "regression": regression,
            "git_commit": git_commit_hash(),
            "library_versions": library_versions(),
            "embedding_model": settings.embedding_model,
            "embedding_device": getattr(embeddings, "device", settings.embedding_device),
            "seed": SEED,
            "config_path": str(config_path),
            "stopped": True,
            "stop_reason": (
                "Condition A regression did not reproduce stored gap results"
            ),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "regression.json", regression)
        print("[gap_sensitivity] REGRESSION FAILED — stopping as specified")
        return out_dir

    print("[gap_sensitivity] scoring conditions:", ", ".join(conditions))
    true_results = _score_all_pairs(
        records=records,
        cv_texts=cv_texts,
        conditions=conditions,
        settings=settings,
        embeddings=embeddings,
    )

    print(
        f"[gap_sensitivity] mismatched control ({MISMATCH_PERMS} permutations, seed={SEED})"
    )
    mismatch_results = _score_mismatched(
        records=records,
        cv_texts=cv_texts,
        true_results=true_results,
        conditions=conditions,
        settings=settings,
        embeddings=embeddings,
        seed=SEED,
        n_permutations=MISMATCH_PERMS,
    )

    # Per-pair outputs
    pair_files = []
    for i, record in enumerate(records):
        cond_map = {c: true_results[c][i] for c in conditions}
        pair_files.append(
            _write_pair_json(
                out_dir,
                record,
                cond_map,
                experiment_id,
                source_experiment,
            )
        )

    # Aggregate metrics
    condition_metrics: dict[str, Any] = {}
    semantic_metrics: dict[str, Any] = {}
    for cond in conditions:
        condition_metrics[cond] = _condition_metrics(true_results[cond])
        semantic_metrics[cond] = _condition_metrics(
            true_results[cond], semantic_only=True
        )

    # Size-adjusted effects vs mismatched
    size_adjusted: dict[str, Any] = {}
    for cond in conditions:
        true_rates = per_pair_gap_rates(_pair_labels(true_results[cond]))
        mm_rates, mm_pooled = _avg_mismatch_gap_rates(mismatch_results[cond])
        true_pooled = gap_share(_flat_labels(true_results[cond]))
        size_adjusted[cond] = {
            "true_mean_per_pair_gap_rate": float(np.nanmean(true_rates))
            if true_rates.size
            else None,
            "mismatched_mean_per_pair_gap_rate": float(np.nanmean(mm_rates))
            if mm_rates.size
            else None,
            "size_adjusted_effect_mean_per_pair": (
                float(np.nanmean(true_rates) - np.nanmean(mm_rates))
                if true_rates.size and mm_rates.size
                else None
            ),
            "true_pooled_gap_share": true_pooled,
            "mismatched_pooled_gap_share": mm_pooled,
            "size_adjusted_effect_pooled_share": (
                (true_pooled - mm_pooled)
                if true_pooled is not None and mm_pooled is not None
                else None
            ),
            "wilcoxon_true_vs_mismatched": wilcoxon_paired(true_rates, mm_rates),
        }

    # Wilcoxon vs A
    vs_a: dict[str, Any] = {}
    a_rates = per_pair_gap_rates(_pair_labels(true_results[CONDITION_A]))
    for cond in conditions:
        if cond == CONDITION_A:
            continue
        rates = per_pair_gap_rates(_pair_labels(true_results[cond]))
        vs_a[cond] = wilcoxon_paired(rates, a_rates)

    # Transitions A -> others (requirement-aligned by pair order + competency)
    transitions: dict[str, Any] = {}
    a_flat = _flat_labels(true_results[CONDITION_A])
    for cond in conditions:
        if cond == CONDITION_A:
            continue
        # Align by walking pairs in lockstep (same job requirements order)
        from_labels: list[MatchLabel] = []
        to_labels: list[MatchLabel] = []
        for a_pr, c_pr in zip(
            true_results[CONDITION_A], true_results[cond], strict=True
        ):
            assert len(a_pr.scores) == len(c_pr.scores)
            for a_s, c_s in zip(a_pr.scores, c_pr.scores, strict=True):
                assert a_s.job_competency == c_s.job_competency
                from_labels.append(a_s.match_label)
                to_labels.append(c_s.match_label)
        transitions[f"{CONDITION_A}->{cond}"] = transition_matrix(from_labels, to_labels)

    # Breakdowns for A and best-evidence (lowest pooled gap share among D/E/C)
    gap_shares = {
        c: condition_metrics[c]["pooled_gap_share"] for c in conditions
    }
    best_cond = min(
        (c for c in conditions if gap_shares[c] is not None),
        key=lambda c: gap_shares[c],
        default=CONDITION_A,
    )
    breakdowns = {
        CONDITION_A: _breakdown(records, true_results[CONDITION_A]),
        best_cond: _breakdown(records, true_results[best_cond]),
        "best_evidence_condition": best_cond,
    }

    # CSV tables
    overview_rows = []
    for cond in conditions:
        m = condition_metrics[cond]
        lc = m["label_counts"]
        tc = m["three_category_counts"]
        overview_rows.append(
            {
                "split": split,
                "condition": cond,
                "n_requirements": sum(lc.values()),
                "strong": lc["strong_match"],
                "partial": lc["partial_match"],
                "weak": lc["weak_match"],
                "missing": lc["missing"],
                "strength_share": tc["strength"] / sum(tc.values()) if sum(tc.values()) else None,
                "partial_share": tc["partial"] / sum(tc.values()) if sum(tc.values()) else None,
                "gap_share": m["pooled_gap_share"],
                "gap_share_ci_low": m["pooled_gap_share_bootstrap"]["ci_low"],
                "gap_share_ci_high": m["pooled_gap_share_bootstrap"]["ci_high"],
                "mean_per_pair_gap_rate": m["per_pair_gap_rate"]["mean"],
                "mean_per_pair_gap_rate_ci_low": m["per_pair_gap_rate_bootstrap"]["ci_low"],
                "mean_per_pair_gap_rate_ci_high": m["per_pair_gap_rate_bootstrap"]["ci_high"],
                "median_per_pair_gap_rate": m["per_pair_gap_rate"]["median"],
                "iqr_per_pair_gap_rate": m["per_pair_gap_rate"]["iqr"],
                "evidence_size_mean": m["evidence_size"]["mean"],
                "evidence_size_median": m["evidence_size"]["median"],
                "mean_tfidf": (m["mean_signals"] or {}).get("tfidf"),
                "mean_bm25_norm": (m["mean_signals"] or {}).get("bm25_norm"),
                "mean_semantic": (m["mean_signals"] or {}).get("semantic"),
                "mean_combined": (m["mean_signals"] or {}).get("combined"),
                "spearman_rho_size_gap": m["dose_response"]["rho"],
                "spearman_p": m["dose_response"]["p_value"],
                "mismatched_gap_share": size_adjusted[cond]["mismatched_pooled_gap_share"],
                "size_adjusted_effect_pooled": size_adjusted[cond][
                    "size_adjusted_effect_pooled_share"
                ],
                "size_adjusted_effect_per_pair": size_adjusted[cond][
                    "size_adjusted_effect_mean_per_pair"
                ],
            }
        )
    _write_csv(out_dir / "tables" / "overview_by_condition.csv", overview_rows)

    wilcoxon_rows = []
    for cond, w in vs_a.items():
        wilcoxon_rows.append(
            {
                "split": split,
                "comparison": f"{cond} vs {CONDITION_A}",
                "n_pairs": w["n_pairs"],
                "n_nonzero": w["n_nonzero"],
                "statistic": w["statistic"],
                "p_value": w["p_value"],
                "rank_biserial": w["rank_biserial"],
            }
        )
    for cond, sa in size_adjusted.items():
        w = sa["wilcoxon_true_vs_mismatched"]
        wilcoxon_rows.append(
            {
                "split": split,
                "comparison": f"{cond} true vs mismatched",
                "n_pairs": w["n_pairs"],
                "n_nonzero": w["n_nonzero"],
                "statistic": w["statistic"],
                "p_value": w["p_value"],
                "rank_biserial": w["rank_biserial"],
            }
        )
    _write_csv(out_dir / "tables" / "wilcoxon.csv", wilcoxon_rows)

    # Transition CSVs
    for key, mat in transitions.items():
        rows = []
        for a_lab, row in mat.items():
            for b_lab, count in row.items():
                rows.append(
                    {
                        "split": split,
                        "transition": key,
                        "from_label": a_lab,
                        "to_label": b_lab,
                        "count": count,
                    }
                )
        safe = key.replace("->", "_to_")
        _write_csv(out_dir / "tables" / f"transition_{safe}.csv", rows)

    device = getattr(embeddings, "device", settings.embedding_device)
    summary = {
        "experiment_id": experiment_id,
        "status": "completed",
        "stage": "gap_sensitivity",
        "split": split,
        "source_experiment": source_experiment,
        "stored_gap_experiment": stored_gap_experiment,
        "conditions": list(conditions),
        "condition_descriptions": {
            c: CONDITION_DESCRIPTIONS.get(c, "") for c in conditions
        },
        "regression": regression,
        "metrics": condition_metrics,
        "semantic_only_metrics": semantic_metrics,
        "size_adjusted": size_adjusted,
        "wilcoxon_vs_A": vs_a,
        "transitions": transitions,
        "breakdowns": breakdowns,
        "bm25_normalisation_note": (
            "Primary labels use within-condition per-pair bm25_max (production). "
            "bm25_norm_fixed / combined_fixed_bm25 use Condition A bm25_max. "
            "Semantic-only metrics remove BM25/TF-IDF scale effects."
        ),
        "seeds": {
            "global": SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "mismatch_permutations": MISMATCH_PERMS,
        },
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "embedding_model": settings.embedding_model,
        "embedding_device": device,
        "n_pairs": len(records),
        "missing_cv_text_pairs": missing_cv,
        "result_files": [f"pairs/{f}" for f in pair_files],
        "config_path": str(config_path),
        "config": config,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "regression.json", regression)
    print(f"[gap_sensitivity] Summary -> {out_dir / 'summary.json'}")
    return out_dir
