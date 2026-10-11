"""Local regression checks before any paid call."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription
from app.services.embeddings.service import EmbeddingService
from app.services.gap_analysis.matrix import build_gap_matrix
from app.services.gap_sensitivity.io_utils import load_extraction_records, request_id
from app.services.treatment_pilot.constants import B0_EXPERIMENTS, GAP_EXPERIMENT

SCORE_TOL = 1e-4
EXPECTED_COUNTS = {
    "requirements": 633,
    "strong_match": 51,
    "partial_match": 175,
    "weak_match": 245,
    "missing": 162,
}


def _combined(tfidf: float, bm25: float, semantic: float, bm25_max: float) -> float:
    bm25_norm = bm25 / bm25_max if bm25_max > 0 else 0.0
    return 0.25 * tfidf + 0.25 * bm25_norm + 0.50 * semantic


def reproduce_gap_matrices(
    *,
    root: Path,
    settings: Settings,
    embeddings: EmbeddingService,
    extraction_experiment: str,
    gap_experiment: str = GAP_EXPERIMENT,
) -> dict[str, Any]:
    """Recompute gap matrices from stored extraction; compare to stored gaps."""
    ext_dir = root / "experiments" / "results" / extraction_experiment
    gap_dir = root / "experiments" / "results" / gap_experiment / "gap_analysis"
    records = load_extraction_records(ext_dir)
    label_tally: dict[str, int] = defaultdict(int)
    n_req = 0
    n_label_mismatch = 0
    n_score_mismatch = 0
    n_weight_mismatch = 0
    max_abs_diff = 0.0
    missing_artifacts: list[str] = []

    for record in records:
        rid = request_id(record) + ".json"
        path = gap_dir / rid
        if not path.is_file():
            missing_artifacts.append(rid)
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        candidate = CandidateProfile.model_validate(record["candidate"])
        job = JobDescription.model_validate(record["job"])
        matrix = build_gap_matrix(candidate, job, settings, embeddings)
        stored_gaps = stored["gap_matrix"]["gaps"]
        if len(matrix.gaps) != len(stored_gaps):
            return {
                "passed": False,
                "error": (
                    f"Requirement count mismatch for {rid}: "
                    f"{len(matrix.gaps)} vs {len(stored_gaps)}"
                ),
            }
        stored_bm25 = [float(g["evidence"].get("bm25_score") or 0.0) for g in stored_gaps]
        recomputed_bm25 = [float(g.evidence.bm25_score or 0.0) for g in matrix.gaps]
        smax = max(stored_bm25) if stored_bm25 else 0.0
        rmax = max(recomputed_bm25) if recomputed_bm25 else 0.0
        for g, sg in zip(matrix.gaps, stored_gaps, strict=True):
            n_req += 1
            label_tally[g.match_label.value] += 1
            if g.job_competency != sg["job_competency"]:
                return {
                    "passed": False,
                    "error": (
                        f"job_competency order mismatch for {rid}: "
                        f"{g.job_competency!r} vs {sg['job_competency']!r}"
                    ),
                }
            if g.match_label.value != sg["match_label"]:
                n_label_mismatch += 1
            ev = sg["evidence"]
            r_comb = _combined(
                float(g.evidence.tfidf_score or 0.0),
                float(g.evidence.bm25_score or 0.0),
                float(g.evidence.semantic_score or 0.0),
                rmax,
            )
            s_comb = _combined(
                float(ev.get("tfidf_score") or 0.0),
                float(ev.get("bm25_score") or 0.0),
                float(ev.get("semantic_score") or 0.0),
                smax,
            )
            diff = abs(r_comb - s_comb)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > SCORE_TOL:
                n_score_mismatch += 1
            if abs(float(g.weight) - float(sg["weight"])) > SCORE_TOL:
                n_weight_mismatch += 1

    counts = {
        "requirements": n_req,
        "strong_match": int(label_tally.get("strong_match", 0)),
        "partial_match": int(label_tally.get("partial_match", 0)),
        "weak_match": int(label_tally.get("weak_match", 0)),
        "missing": int(label_tally.get("missing", 0)),
    }
    counts_match = counts == EXPECTED_COUNTS
    passed = (
        not missing_artifacts
        and n_label_mismatch == 0
        and n_score_mismatch == 0
        and n_weight_mismatch == 0
        and counts_match
        and max_abs_diff <= SCORE_TOL
    )
    return {
        "passed": passed,
        "counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "counts_match_expected": counts_match,
        "n_label_mismatch": n_label_mismatch,
        "n_score_mismatch": n_score_mismatch,
        "n_weight_mismatch": n_weight_mismatch,
        "max_abs_diff": max_abs_diff,
        "missing_artifacts": missing_artifacts,
        "n_pairs": len(records),
    }


def validate_b0_records(root: Path, primary_stems: set[str]) -> dict[str, Any]:
    exceptions: list[dict] = []
    per_run: dict[str, dict] = {}
    for exp_id in B0_EXPERIMENTS:
        d = root / "experiments" / "results" / exp_id / "baseline"
        n = 0
        for path in sorted(d.glob("*.json")):
            if path.name.startswith("_"):
                continue
            if path.stem not in primary_stems:
                continue
            n += 1
            rec = json.loads(path.read_text(encoding="utf-8"))
            qs = rec.get("questions") or []
            if len(qs) != 10:
                exceptions.append(
                    {
                        "experiment": exp_id,
                        "file": path.name,
                        "reason": "count",
                        "n": len(qs),
                    }
                )
            for i, q in enumerate(qs):
                text = q.get("question") if isinstance(q, dict) else q
                if not str(text or "").strip():
                    exceptions.append(
                        {
                            "experiment": exp_id,
                            "file": path.name,
                            "reason": "empty",
                            "index": i,
                        }
                    )
        per_run[exp_id] = {"n_primary_pairs": n}
    return {
        "passed": len(exceptions) == 0,
        "exceptions": exceptions,
        "per_run": per_run,
    }
