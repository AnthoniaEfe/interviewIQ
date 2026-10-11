"""Model-free descriptive diagnostics for the treatment pilot."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.domain import GapMatrix, MatchLabel
from app.services.embeddings.service import EmbeddingService
from app.services.fit_label_control.stats import holm_adjust
from app.services.gap_sensitivity.metrics import bootstrap_mean_ci, wilcoxon_paired
from app.services.mmr.select import redundant_pairs
from app.services.treatment_pilot.constants import (
    B0_EXPERIMENTS,
    BOOTSTRAP_RESAMPLES,
    COVERAGE_THRESHOLDS,
    PRIMARY_COVERAGE_THRESHOLD,
    SEED,
    SIM_THRESHOLD,
)

STOPLIST = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "your",
    "you",
    "we",
    "our",
    "this",
    "that",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "how",
    "what",
    "when",
    "where",
    "why",
    "which",
    "who",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "about",
    "describe",
    "explain",
    "give",
    "example",
    "experience",
    "role",
    "team",
    "work",
    "project",
    "using",
    "use",
    "used",
}


def _question_texts(record: dict) -> list[str]:
    qs = record.get("questions") or []
    out = []
    for q in qs:
        if isinstance(q, dict):
            out.append(str(q.get("question") or "").strip())
        else:
            out.append(str(q).strip())
    return [t for t in out if t]


def redundancy_metrics(texts: list[str], embeddings: EmbeddingService) -> dict[str, Any]:
    n = len(texts)
    if n < 2:
        return {
            "mean_pairwise_sim": None,
            "mean_max_sim": None,
            "n_redundant_pairs": 0,
            "n_questions": n,
        }
    sim = embeddings.similarity_matrix(texts, texts)
    pairs = []
    max_sims = []
    for i in range(n):
        row_max = -1.0
        for j in range(n):
            if i == j:
                continue
            s = float(sim[i, j])
            if j > i:
                pairs.append(s)
            row_max = max(row_max, s)
        max_sims.append(row_max)
    red = redundant_pairs(sim, SIM_THRESHOLD)
    return {
        "mean_pairwise_sim": float(np.mean(pairs)) if pairs else None,
        "mean_max_sim": float(np.mean(max_sims)) if max_sims else None,
        "n_redundant_pairs": len(red),
        "n_questions": n,
    }


def _profile_terms(candidate: dict, job: dict) -> set[str]:
    terms: set[str] = set()

    def add(text: str | None) -> None:
        if not text:
            return
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{1,}", text):
            low = tok.lower()
            if low not in STOPLIST and len(low) > 2:
                terms.add(low)

    for key in (
        "technical_skills",
        "soft_skills",
        "domain_experience",
        "certifications",
        "competencies",
    ):
        for item in candidate.get(key) or []:
            if isinstance(item, dict):
                add(item.get("normalized_label"))
            else:
                add(str(item))
    for exp in candidate.get("experience") or []:
        if isinstance(exp, dict):
            add(exp.get("employer"))
            add(exp.get("title"))
            for t in exp.get("technologies") or []:
                add(str(t))
    for key in ("required_skills", "preferred_skills", "competencies", "soft_skills"):
        for item in job.get(key) or []:
            if isinstance(item, dict):
                add(item.get("normalized_label"))
            else:
                add(str(item))
    add(job.get("title"))
    return terms


def lexical_specificity(texts: list[str], terms: set[str]) -> float | None:
    if not texts:
        return None
    hits = 0
    for t in texts:
        tokens = {tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{1,}", t)}
        if tokens & terms:
            hits += 1
    return hits / len(texts)


def coverage_proxy(
    texts: list[str],
    requirements: list[str],
    gap_labels: list[str],
    embeddings: EmbeddingService,
    thresholds: tuple[float, ...] = COVERAGE_THRESHOLDS,
) -> dict[str, Any]:
    """Assign each question to best-matching requirement by cosine (exploratory)."""
    n_req = len(requirements)
    if not texts or n_req == 0:
        return {
            "n_requirements": n_req,
            "by_threshold": {},
            "label": "proxy: embedding assignment, not human judgement",
        }
    sim = embeddings.similarity_matrix(texts, requirements)  # (nq, nreq)
    by_thr: dict[str, Any] = {}
    for thr in thresholds:
        covered: set[int] = set()
        cat_counts = Counter()
        for i in range(sim.shape[0]):
            j = int(np.argmax(sim[i]))
            if float(sim[i, j]) >= thr:
                covered.add(j)
        for j in covered:
            lab = gap_labels[j] if j < len(gap_labels) else "unknown"
            if lab in {MatchLabel.WEAK.value, MatchLabel.MISSING.value}:
                cat_counts["gap"] += 1
            elif lab == MatchLabel.PARTIAL.value:
                cat_counts["partial"] += 1
            elif lab == MatchLabel.STRONG.value:
                cat_counts["strength"] += 1
            else:
                cat_counts["other"] += 1
        n_cov = len(covered)
        by_thr[str(thr)] = {
            "n_covered": n_cov,
            "coverage_rate": n_cov / n_req if n_req else None,
            "share_gap": (cat_counts["gap"] / n_cov) if n_cov else None,
            "share_partial": (cat_counts["partial"] / n_cov) if n_cov else None,
            "share_strength": (cat_counts["strength"] / n_cov) if n_cov else None,
        }
    return {
        "n_requirements": n_req,
        "by_threshold": by_thr,
        "label": "proxy: embedding assignment, not human judgement",
    }


def metadata_distribution(rich_questions: list[dict], gap_matrix: GapMatrix) -> dict[str, Any]:
    dim = Counter()
    gap_cat = Counter()
    by_name = {g.job_competency: g.match_label.value for g in gap_matrix.gaps}
    for q in rich_questions:
        dim[q.get("question_type") or q.get("dimension") or "unknown"] += 1
        tg = q.get("target_gap")
        if tg == "profile_probe":
            gap_cat["profile_probe"] += 1
        elif tg in by_name:
            lab = by_name[tg]
            if lab in {MatchLabel.WEAK.value, MatchLabel.MISSING.value}:
                gap_cat["gap"] += 1
            elif lab == MatchLabel.PARTIAL.value:
                gap_cat["partial"] += 1
            elif lab == MatchLabel.STRONG.value:
                gap_cat["strength"] += 1
            else:
                gap_cat["other"] += 1
        else:
            gap_cat["unmapped"] += 1
    n = len(rich_questions) or 1
    return {
        "dimension": dict(dim),
        "target_gap_category": dict(gap_cat),
        "dimension_share": {k: v / n for k, v in dim.items()},
        "target_gap_category_share": {k: v / n for k, v in gap_cat.items()},
    }


def jaccard(a: set[int], b: set[int]) -> float | None:
    if not a and not b:
        return None
    u = a | b
    if not u:
        return None
    return len(a & b) / len(u)


def load_condition_records(dir_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not dir_path.is_dir():
        return out
    for path in sorted(dir_path.glob("*.json")):
        if path.name.startswith("_"):
            continue
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def load_b0_replicates(root: Path, primary_stems: set[str]) -> dict[str, list[dict]]:
    """stem -> list of 3 B0 records."""
    by_stem: dict[str, list[dict]] = {s: [] for s in primary_stems}
    for exp_id in B0_EXPERIMENTS:
        d = root / "experiments" / "results" / exp_id / "baseline"
        for path in d.glob("*.json"):
            if path.stem not in primary_stems:
                continue
            by_stem[path.stem].append(json.loads(path.read_text(encoding="utf-8")))
    return by_stem


def b0_variability(
    b0_by_stem: dict[str, list[dict]], embeddings: EmbeddingService
) -> dict[str, Any]:
    """Mean pairwise embedding similarity and question-text overlap across B0 replicates."""
    pair_sims: list[float] = []
    overlaps: list[float] = []
    for stem, recs in b0_by_stem.items():
        if len(recs) < 2:
            continue
        sets = []
        embs = []
        for rec in recs:
            texts = _question_texts(rec)
            sets.append({t.lower() for t in texts})
            if texts:
                embs.append(embeddings.embed_batch(texts).mean(axis=0))
        # text overlap: mean Jaccard across replicate pairs
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                overlaps.append(len(sets[i] & sets[j]) / len(u) if u else 0.0)
        if len(embs) >= 2:
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    a, b = embs[i], embs[j]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    pair_sims.append(float(np.dot(a, b) / denom) if denom else 0.0)
    return {
        "mean_replicate_embedding_similarity": float(np.mean(pair_sims)) if pair_sims else None,
        "mean_question_text_jaccard": float(np.mean(overlaps)) if overlaps else None,
        "n_pair_comparisons_sim": len(pair_sims),
        "n_pair_comparisons_overlap": len(overlaps),
    }


def diagnose_condition(
    *,
    condition: str,
    records: dict[str, dict],
    extraction_by_stem: dict[str, dict],
    gap_by_stem: dict[str, dict],
    embeddings: EmbeddingService,
    has_metadata: bool,
) -> dict[str, Any]:
    per_pair: list[dict] = []
    for stem, rec in sorted(records.items()):
        texts = _question_texts(rec)
        gap_rec = gap_by_stem.get(stem) or gap_by_stem.get(stem.split("__rep")[0])
        ext = extraction_by_stem.get(stem) or extraction_by_stem.get(stem.split("__rep")[0])
        gaps = []
        reqs: list[str] = []
        labels: list[str] = []
        if gap_rec:
            gm = GapMatrix.model_validate(gap_rec["gap_matrix"])
            gaps = gm.gaps
            reqs = [g.job_competency for g in gaps]
            labels = [g.match_label.value for g in gaps]
        red = redundancy_metrics(texts, embeddings)
        cov = coverage_proxy(texts, reqs, labels, embeddings)
        terms = _profile_terms(ext.get("candidate") or {}, ext.get("job") or {}) if ext else set()
        lex = lexical_specificity(texts, terms)
        meta = None
        if has_metadata:
            rich = rec.get("questions_rich") or []
            if gap_rec:
                meta = metadata_distribution(
                    rich, GapMatrix.model_validate(gap_rec["gap_matrix"])
                )
        else:
            meta = "not available"
        per_pair.append(
            {
                "stem": stem,
                "n_questions": len(texts),
                "schema_valid": len(texts) == 10 and all(texts),
                "n_requirements": len(reqs),
                "redundancy": red,
                "coverage": cov,
                "lexical_specificity": lex,
                "metadata": meta,
            }
        )

    def col(getter) -> np.ndarray:
        vals = []
        for row in per_pair:
            v = getter(row)
            vals.append(float(v) if v is not None else np.nan)
        return np.asarray(vals, dtype=np.float64)

    red_mean = col(lambda r: r["redundancy"]["mean_pairwise_sim"])
    cov_045 = col(
        lambda r: (r["coverage"]["by_threshold"].get(str(PRIMARY_COVERAGE_THRESHOLD)) or {}).get(
            "coverage_rate"
        )
    )
    lex_arr = col(lambda r: r["lexical_specificity"])
    return {
        "condition": condition,
        "n_pairs": len(per_pair),
        "n_schema_valid": sum(1 for r in per_pair if r["schema_valid"]),
        "redundancy_mean_pairwise": bootstrap_mean_ci(
            red_mean, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
        "coverage_0.45": bootstrap_mean_ci(
            cov_045, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
        "lexical_specificity": bootstrap_mean_ci(
            lex_arr, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
        "per_pair": per_pair,
    }


def paired_comparisons(
    diagnostics: dict[str, dict],
) -> dict[str, Any]:
    """T vs B1, T vs B0, T vs B2 on redundancy and coverage."""
    t = diagnostics.get("T")
    if not t:
        return {}
    t_stems = {r["stem"]: r for r in t["per_pair"]}

    def align(other_name: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        other = diagnostics.get(other_name)
        if not other:
            return np.array([]), np.array([])
        xs, ys = [], []
        for row in other["per_pair"]:
            stem = row["stem"]
            if stem not in t_stems:
                continue
            if metric == "redundancy":
                xv = t_stems[stem]["redundancy"]["mean_pairwise_sim"]
                yv = row["redundancy"]["mean_pairwise_sim"]
            else:
                xv = (t_stems[stem]["coverage"]["by_threshold"].get("0.45") or {}).get(
                    "coverage_rate"
                )
                yv = (row["coverage"]["by_threshold"].get("0.45") or {}).get("coverage_rate")
            if xv is None or yv is None:
                continue
            xs.append(float(xv))
            ys.append(float(yv))
        return np.asarray(xs), np.asarray(ys)

    comparisons = {}
    pvals_red = []
    pvals_cov = []
    keys = []
    for name in ("B1", "B0", "B2"):
        if name not in diagnostics:
            continue
        keys.append(name)
        tx, ox = align(name, "redundancy")
        # For redundancy, lower is better → test whether T - other is negative.
        w_red = wilcoxon_paired(tx, ox) if tx.size else {
            "p_value": None,
            "rank_biserial": None,
            "n_pairs": 0,
        }
        diff_red = tx - ox if tx.size else np.array([])
        ci_red = bootstrap_mean_ci(diff_red, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED)
        tx2, ox2 = align(name, "coverage")
        w_cov = wilcoxon_paired(tx2, ox2) if tx2.size else {
            "p_value": None,
            "rank_biserial": None,
            "n_pairs": 0,
        }
        diff_cov = tx2 - ox2 if tx2.size else np.array([])
        ci_cov = bootstrap_mean_ci(diff_cov, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED)
        comparisons[f"T_vs_{name}"] = {
            "redundancy": {"wilcoxon": w_red, "diff_ci": ci_red},
            "coverage_0.45": {"wilcoxon": w_cov, "diff_ci": ci_cov},
        }
        pvals_red.append(w_red.get("p_value"))
        pvals_cov.append(w_cov.get("p_value"))

    adj_red = holm_adjust(pvals_red)
    adj_cov = holm_adjust(pvals_cov)
    for i, name in enumerate(keys):
        comparisons[f"T_vs_{name}"]["redundancy"]["holm_p"] = adj_red[i]
        comparisons[f"T_vs_{name}"]["coverage_0.45"]["holm_p"] = adj_cov[i]
    return comparisons


def decision_rule(diagnostics: dict[str, dict], comparisons: dict[str, Any]) -> dict[str, Any]:
    """Advisory: scale only if T ≥ B1 on redundancy (lower) and coverage (higher)."""
    t = diagnostics.get("T")
    b1 = diagnostics.get("B1")
    if not t or not b1:
        return {
            "recommend_scale": False,
            "reason": "missing T or B1 diagnostics",
            "case": "incomplete",
        }
    t_red = (t["redundancy_mean_pairwise"] or {}).get("mean")
    b1_red = (b1["redundancy_mean_pairwise"] or {}).get("mean")
    t_cov = (t["coverage_0.45"] or {}).get("mean")
    b1_cov = (b1["coverage_0.45"] or {}).get("mean")
    red_ok = (
        t_red is not None and b1_red is not None and t_red <= b1_red + 1e-12
    )
    cov_ok = (
        t_cov is not None and b1_cov is not None and t_cov >= b1_cov - 1e-12
    )
    cmp = comparisons.get("T_vs_B1") or {}
    recommend = bool(red_ok and cov_ok)
    return {
        "recommend_scale": recommend,
        "redundancy_t_mean": t_red,
        "redundancy_b1_mean": b1_red,
        "redundancy_t_le_b1": red_ok,
        "coverage_t_mean": t_cov,
        "coverage_b1_mean": b1_cov,
        "coverage_t_ge_b1": cov_ok,
        "paired_diff_ci_redundancy": (cmp.get("redundancy") or {}).get("diff_ci"),
        "paired_diff_ci_coverage": (cmp.get("coverage_0.45") or {}).get("diff_ci"),
        "case": (
            "T at least as good as B1 on both proxies — advisory recommend scale"
            if recommend
            else "T not at least as good as B1 on both proxies — diagnose components first"
        ),
        "note": (
            "Advisory only; proxies are not quality; decision to scale remains the researcher's."
        ),
    }


def selection_overlaps(
    t_records: dict[str, dict],
    ablation_root: Path,
) -> dict[str, Any]:
    out: dict[str, list[float]] = {"T_B2": [], "T_S1": [], "T_S2": []}
    for stem, t_rec in t_records.items():
        t_idx = set((t_rec.get("selection") or {}).get("selected_indices") or [])
        if not t_idx and t_rec.get("questions_rich"):
            # fallback: use question text sets
            t_set = {q["question"] for q in t_rec["questions_rich"]}
        else:
            t_set = None
        for cond, key in (("B2", "T_B2"), ("S1", "T_S1"), ("S2", "T_S2")):
            path = ablation_root / cond / f"{stem}.json"
            if not path.is_file():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            o_idx = set(other.get("selected_indices") or [])
            if t_idx and o_idx:
                j = jaccard(t_idx, o_idx)
            else:
                o_set = {q["question"] for q in (other.get("questions_rich") or [])}
                if t_set is None:
                    t_set = {q.get("question") for q in (t_rec.get("questions_rich") or [])}
                inter = len(t_set & o_set)
                union = len(t_set | o_set) or 1
                j = inter / union
            if j is not None:
                out[key].append(float(j))
    summary = {}
    for k, vals in out.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[k] = bootstrap_mean_ci(arr, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED)
    return summary
