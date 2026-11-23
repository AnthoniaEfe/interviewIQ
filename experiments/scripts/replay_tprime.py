"""Replay selector T-prime on stored primary-30 pools.

No LLM calls. Cosine matrices were not persisted by the generation run.
This script recomputes them locally with the frozen embedding model
``all-MiniLM-L6-v2``, saves them, and then applies the pre-registered
selector. Metric formulas are the frozen M3.P / M4.P functions. Every
number is tagged post-hoc. Selector parameters are not changed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from app.schemas.pool import POOL_DIMENSIONS, PoolQuestionItem, pool_item_to_interview_question  # noqa: E402
from app.services.eval_metrics.compute import compute_full_metrics  # noqa: E402
from app.services.embeddings.service import SentenceTransformerEmbeddingService  # noqa: E402
from app.services.gap_sensitivity.io_utils import git_commit_hash, write_json  # noqa: E402
from app.services.selection.tprime import select_tprime  # noqa: E402

POOL_DIR = ROOT / "experiments/results/exp_20261011T130619Z/pool"
TREATMENT_DIR = ROOT / "experiments/results/exp_20261011T130619Z/treatment"
B2_DIR = ROOT / "experiments/results/exp_20261011T131945Z/B2"
B1_DIR = ROOT / "experiments/results/exp_20261011T130310Z/baseline_profile_aware"
GAP_DIR = ROOT / "experiments/results/exp_20260927T215232Z/gap_analysis"
FREEZE_PATH = ROOT / "experiments/metrics/metrics_freeze.json"
OUT_DIR = ROOT / "experiments/results/selector_tprime_replay"
SIM_DIR = OUT_DIR / "similarity"
PREREG = "docs/selector_tprime_prereg.md"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GAP_LABELS = frozenset({"weak_match", "missing"})
STORED_T_MEANS = {"M3.P": 0.40166666666666667, "M4.P": 7.315436631080094}
STORED_B2_MEANS = {"M3.P": 0.4616666666666666, "M4.P": 7.491812181038271}
STORED_B1_MEANS = {"M3.P": 0.3441666666666667, "M4.P": 7.267037979677226}


def _prereg_commit() -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", PREREG],
        cwd=ROOT,
        text=True,
    ).strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _gap_share(target_gaps: list[str], labels: dict[str, str], indices: list[int]) -> float | None:
    if not indices:
        return None
    hits = 0
    for index in indices:
        if labels.get(target_gaps[index]) in GAP_LABELS:
            hits += 1
    return hits / len(indices)


def _dimension_counts(dimensions: list[str], indices: list[int]) -> dict[str, int]:
    counts = {dimension: 0 for dimension in POOL_DIMENSIONS}
    for index in indices:
        dimension = dimensions[index]
        if dimension in counts:
            counts[dimension] += 1
    return counts


def _question_texts(record: dict) -> list[str]:
    texts = []
    for item in record.get("questions") or []:
        if isinstance(item, dict):
            text = str(item.get("question") or "").strip()
        else:
            text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def _requirements(gap_payload: dict) -> list[dict]:
    gaps = ((gap_payload.get("gap_matrix") or {}).get("gaps")) or []
    return [dict(gap) for gap in gaps]


def _labels(requirements: list[dict]) -> dict[str, str]:
    return {
        str(row.get("job_competency")): str(row.get("match_label") or "")
        for row in requirements
        if row.get("job_competency")
    }


def _metrics(texts: list[str], embeddings, requirements: list[dict], tau: float | None) -> dict:
    raw = compute_full_metrics(texts, embeddings, requirements=requirements, tau=tau)
    return {
        "M1.P": raw.get("M1.P"),
        "M2.P": raw.get("M2.P"),
        "M3.P": raw.get("M3.P"),
        "M4.P": raw.get("M4.P"),
        "assignment_available": bool(raw.get("assignment_available")),
        "n_questions": len(texts),
    }


def _condition_summary(rows: list[dict], key: str) -> dict:
    return {
        "n": len(rows),
        "mean_gap_share": _mean([row[key]["gap_share"] for row in rows]),
        "pools_quota_met": int(sum(1 for row in rows if row[key]["quota_met"] is True)),
        "quota_defined": all(row[key]["quota_met"] is not None for row in rows),
        "mean_dimension_counts": {
            dimension: _mean(
                [
                    None
                    if row[key]["dimension_counts"] is None
                    else row[key]["dimension_counts"].get(dimension)
                    for row in rows
                ]
            )
            for dimension in POOL_DIMENSIONS
        },
        "mean_M1.P": _mean([row[key]["metrics"]["M1.P"] for row in rows]),
        "mean_M2.P": _mean([row[key]["metrics"]["M2.P"] for row in rows]),
        "mean_M3.P": _mean([row[key]["metrics"]["M3.P"] for row in rows]),
        "mean_M4.P": _mean([row[key]["metrics"]["M4.P"] for row in rows]),
    }


def _select_record(
    *,
    result,
    target_gaps: list[str],
    dimensions: list[str],
    labels: dict[str, str],
    texts: list[str],
    embeddings,
    requirements: list[dict],
    tau: float | None,
) -> dict:
    indices = list(result.selected_indices)
    selected_texts = [texts[i] for i in indices]
    return {
        "selected_indices": indices,
        "n_selected": len(indices),
        "quota_met": bool(result.quota_met),
        "dimension_counts": dict(result.dimension_counts),
        "gap_share": _gap_share(target_gaps, labels, indices),
        "relaxations": [
            {
                "phase": item.phase,
                "reason": item.reason,
                "dimension": item.dimension,
                "reserved_index": item.reserved_index,
            }
            for item in result.relaxations
        ],
        "metrics": _metrics(selected_texts, embeddings, requirements, tau),
        "use_weights": True,
    }


def _stored_record(
    indices: list[int] | None,
    texts: list[str],
    *,
    target_gaps: list[str] | None,
    dimensions: list[str] | None,
    labels: dict[str, str] | None,
    embeddings,
    requirements: list[dict],
    tau: float | None,
    quota_defined: bool,
) -> dict:
    gap = None
    counts = None
    quota = None
    if indices is not None and target_gaps is not None and labels is not None and dimensions is not None:
        gap = _gap_share(target_gaps, labels, indices)
        counts = _dimension_counts(dimensions, indices)
        if quota_defined:
            quota = all(counts[dimension] >= 1 for dimension in POOL_DIMENSIONS)
    return {
        "selected_indices": indices,
        "n_selected": len(texts),
        "quota_met": quota,
        "dimension_counts": counts,
        "gap_share": gap,
        "metrics": _metrics(texts, embeddings, requirements, tau),
    }


def main() -> int:
    commit = _prereg_commit()
    if not commit:
        raise SystemExit(f"Refusing to run: {PREREG} has no commit yet")
    pool_paths = sorted(p for p in POOL_DIR.glob("*.json") if not p.name.startswith("_"))
    if len(pool_paths) != 30:
        raise SystemExit(f"Refusing to run: expected 30 pools, found {len(pool_paths)}")
    freeze = _load_json(FREEZE_PATH)
    tau_map = dict(freeze.get("tau_p") or {})

    embeddings = SentenceTransformerEmbeddingService(EMBEDDING_MODEL, device="auto")
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in pool_paths:
        pool = _load_json(path)
        pair_id = str(pool.get("pair_id") or "")
        stem = path.stem
        items = [PoolQuestionItem.model_validate(item) for item in pool.get("valid_questions") or []]
        questions = [pool_item_to_interview_question(item) for item in items]
        weights = [float(value) for value in pool.get("weights") or []]
        dimensions = [item.dimension for item in items]
        target_gaps = [item.target_gap for item in items]
        texts = [item.question for item in items]
        if not (len(weights) == len(dimensions) == len(target_gaps) == len(texts) == len(questions)):
            raise SystemExit(f"Refusing to run: {path.name} pool fields have unequal lengths")
        gap_path = GAP_DIR / f"{stem}.json"
        treatment_path = TREATMENT_DIR / f"{stem}.json"
        b2_path = B2_DIR / f"{stem}.json"
        b1_path = B1_DIR / f"{stem}.json"
        for required in (gap_path, treatment_path, b2_path, b1_path):
            if not required.is_file():
                raise SystemExit(f"Refusing to run: missing {required}")
        gap_payload = _load_json(gap_path)
        requirements = _requirements(gap_payload)
        labels = _labels(requirements)
        tau = tau_map.get(pair_id)
        if tau is None:
            raise SystemExit(f"Refusing to run: no frozen tau_p for {pair_id}")
        similarity = embeddings.similarity_matrix(texts, texts)
        np.savez_compressed(SIM_DIR / f"{stem}.npz", similarity=similarity)

        treatment = _load_json(treatment_path)
        b2 = _load_json(b2_path)
        b1 = _load_json(b1_path)
        t_indices = [int(i) for i in (treatment.get("selection") or {}).get("selected_indices") or []]
        b2_indices = [int(i) for i in b2.get("selected_indices") or []]
        t_result = select_tprime(
            questions,
            similarity,
            weights,
            dimensions,
            target_gaps,
            labels,
            use_weights=True,
        )
        tprime = _select_record(
            result=t_result,
            target_gaps=target_gaps,
            dimensions=dimensions,
            labels=labels,
            texts=texts,
            embeddings=embeddings,
            requirements=requirements,
            tau=float(tau),
        )
        rows.append(
            {
                "pair_id": pair_id,
                "stem": stem,
                "n_pool": len(texts),
                "tau_p": float(tau),
                "Tprime": tprime,
                "T": _stored_record(
                    t_indices,
                    _question_texts(treatment),
                    target_gaps=target_gaps,
                    dimensions=dimensions,
                    labels=labels,
                    embeddings=embeddings,
                    requirements=requirements,
                    tau=float(tau),
                    quota_defined=False,
                ),
                "B2": _stored_record(
                    b2_indices,
                    _question_texts(b2),
                    target_gaps=target_gaps,
                    dimensions=dimensions,
                    labels=labels,
                    embeddings=embeddings,
                    requirements=requirements,
                    tau=float(tau),
                    quota_defined=False,
                ),
                "B1": _stored_record(
                    None,
                    _question_texts(b1),
                    target_gaps=None,
                    dimensions=None,
                    labels=None,
                    embeddings=embeddings,
                    requirements=requirements,
                    tau=float(tau),
                    quota_defined=False,
                ),
            }
        )

    tprime_summary = _condition_summary(rows, "Tprime")
    t_summary = _condition_summary(rows, "T")
    b2_summary = _condition_summary(rows, "B2")
    b1_summary = _condition_summary(rows, "B1")

    m3_t = t_summary["mean_M3.P"]
    m4_t = t_summary["mean_M4.P"]
    m3_tp = tprime_summary["mean_M3.P"]
    m4_tp = tprime_summary["mean_M4.P"]
    m3_b2 = b2_summary["mean_M3.P"]
    clause_quota = tprime_summary["pools_quota_met"] >= 24
    clause_gap = (
        tprime_summary["mean_gap_share"] is not None and tprime_summary["mean_gap_share"] < 0.95
    )
    clause_coverage = (
        m3_tp is not None and m3_t is not None and m4_tp is not None and m4_t is not None
        and m3_tp >= m3_t - 0.02
        and m4_tp >= m4_t - 0.25
    )
    success = bool(clause_quota and clause_gap and clause_coverage)
    fallback_triggered = (
        m3_b2 is not None and m3_tp is not None and (m3_b2 - m3_tp) > 0.05
    )

    ablation_summary = None
    if fallback_triggered:
        for row, path in zip(rows, pool_paths):
            pool = _load_json(path)
            items = [PoolQuestionItem.model_validate(item) for item in pool.get("valid_questions") or []]
            questions = [pool_item_to_interview_question(item) for item in items]
            weights = [float(value) for value in pool.get("weights") or []]
            dimensions = [item.dimension for item in items]
            target_gaps = [item.target_gap for item in items]
            texts = [item.question for item in items]
            gap_payload = _load_json(GAP_DIR / f"{path.stem}.json")
            requirements = _requirements(gap_payload)
            labels = _labels(requirements)
            similarity = np.load(SIM_DIR / f"{path.stem}.npz")["similarity"]
            result = select_tprime(
                questions,
                similarity,
                weights,
                dimensions,
                target_gaps,
                labels,
                use_weights=False,
            )
            record = _select_record(
                result=result,
                target_gaps=target_gaps,
                dimensions=dimensions,
                labels=labels,
                texts=texts,
                embeddings=embeddings,
                requirements=requirements,
                tau=float(row["tau_p"]),
            )
            record["use_weights"] = False
            row["Tprime_unweighted"] = record
        ablation_summary = _condition_summary(rows, "Tprime_unweighted")

    summary = {
        "status": "completed",
        "label": "post-hoc",
        "exploratory": True,
        "confirmatory": False,
        "prereg_commit": commit,
        "prereg_path": PREREG,
        "code_commit": git_commit_hash(),
        "n_pools": len(rows),
        "llm_calls": 0,
        "cost_usd": 0.0,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": embeddings.device,
        "embeddings_computed_locally": True,
        "protocol_note": (
            "Pre-registration section 3 said this replay would not encode text and "
            "would leave M3.P and M4.P uncomputed if vectors were absent. The "
            "generation run did not persist pool cosine matrices. The pre-registered "
            "MMR score cannot be applied without them. This run recomputed cosine "
            "matrices locally with the frozen model all-MiniLM-L6-v2, saved them "
            "under similarity/, and scored M3.P and M4.P with stored tau_p and the "
            "unchanged metrics functions. Selector parameters were not changed."
        ),
        "stored_frozen_means_for_reference": {
            "T": STORED_T_MEANS,
            "B2": STORED_B2_MEANS,
            "B1": STORED_B1_MEANS,
            "source": "experiments/results/exp_20261012T163833Z/summary.json",
        },
        "recomputed_means": {
            "Tprime": tprime_summary,
            "T": t_summary,
            "B2": b2_summary,
            "B1": b1_summary,
            "Tprime_unweighted": ablation_summary,
        },
        "success_rule": {
            "quota_at_least_24": clause_quota,
            "gap_share_below_0.95": clause_gap,
            "coverage_and_diversity_within_margin_of_T": clause_coverage,
            "all_three": success,
            "margins": {"M3.P": 0.02, "M4.P": 0.25},
            "comparison": "recomputed paired means from this run, not the stored frozen means",
        },
        "fallback": {
            "triggered": fallback_triggered,
            "rule": "mean M3.P(B2) - mean M3.P(Tprime) > 0.05",
            "delta_M3_B2_minus_Tprime": None if m3_b2 is None or m3_tp is None else m3_b2 - m3_tp,
            "ablation": "Tprime with competency weights removed" if fallback_triggered else None,
        },
        "similarity_dir": str(SIM_DIR.relative_to(ROOT)),
    }
    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "per_pool.json", {"rows": rows})
    print(json.dumps({
        "status": summary["status"],
        "success": success,
        "Tprime": {
            "quota_met": tprime_summary["pools_quota_met"],
            "gap_share": tprime_summary["mean_gap_share"],
            "M3.P": tprime_summary["mean_M3.P"],
            "M4.P": tprime_summary["mean_M4.P"],
        },
        "T": {"gap_share": t_summary["mean_gap_share"], "M3.P": t_summary["mean_M3.P"], "M4.P": t_summary["mean_M4.P"]},
        "B2": {"gap_share": b2_summary["mean_gap_share"], "M3.P": b2_summary["mean_M3.P"], "M4.P": b2_summary["mean_M4.P"]},
        "B1": {"M3.P": b1_summary["mean_M3.P"], "M4.P": b1_summary["mean_M4.P"]},
        "fallback_triggered": fallback_triggered,
        "ablation": None if ablation_summary is None else {
            "quota_met": ablation_summary["pools_quota_met"],
            "gap_share": ablation_summary["mean_gap_share"],
            "M3.P": ablation_summary["mean_M3.P"],
            "M4.P": ablation_summary["mean_M4.P"],
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
