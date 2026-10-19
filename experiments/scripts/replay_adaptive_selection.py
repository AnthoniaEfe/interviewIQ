"""Exploratory replay of the adaptive-selection grid on stored pilot pools.

No LLM calls. Refuses to run if a pool lacks dimension or a priority in (0, 1].
Does not tune lambda_min, lambda_max, or quotas. Primary-30 only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.schemas.domain import InterviewQuestion  # noqa: E402
from app.schemas.pool import POOL_DIMENSIONS, pool_item_to_interview_question, PoolQuestionItem  # noqa: E402
from app.services.eval_metrics.compute import compute_full_metrics  # noqa: E402
from app.services.eval_metrics.freeze import FreezeError, verify_freeze  # noqa: E402
from app.services.gap_sensitivity.io_utils import write_json  # noqa: E402
from app.services.mmr.adaptive import named_ablation_grid, select_adaptive  # noqa: E402

POOL_DIR = ROOT / "experiments/results/exp_20261011T130619Z/pool"
GAP_DIR = ROOT / "experiments/results/exp_20260927T215232Z/gap_analysis"
OUT_DIR = ROOT / "experiments/adaptive_selection_replay"
FIG_DIR = ROOT / "docs/figures"
K = 10
DPI = 200


def _git_prereg() -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "docs/adaptive_selection_prereg.md"],
        cwd=ROOT,
        text=True,
    ).strip()


def _mean_pairwise(sim: np.ndarray, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    vals = []
    for a_i, a in enumerate(indices):
        for b in indices[a_i + 1 :]:
            vals.append(float(sim[a, b]))
    return float(np.mean(vals)) if vals else None


def _load_pool(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("valid_questions")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"Refusing to run: {path.name} has no valid_questions")
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not str(item.get("dimension") or "").strip():
            raise SystemExit(f"Refusing to run: {path.name} item {index} lacks dimension")
        try:
            priority = float(item.get("priority"))
        except (TypeError, ValueError):
            raise SystemExit(f"Refusing to run: {path.name} item {index} has invalid priority")
        if not (0.0 < priority <= 1.0):
            raise SystemExit(
                f"Refusing to run: {path.name} item {index} priority {priority} is not in (0, 1]"
            )
    weights = payload.get("weights")
    if not isinstance(weights, list) or len(weights) != len(items):
        raise SystemExit(
            f"Refusing to run: {path.name} weights missing or length {0 if not isinstance(weights, list) else len(weights)} != {len(items)}"
        )
    questions: list[InterviewQuestion] = []
    dimensions: list[str] = []
    for item in items:
        parsed = PoolQuestionItem.model_validate(item)
        questions.append(pool_item_to_interview_question(parsed))
        dimensions.append(parsed.dimension)
    return {
        "path": path,
        "pair_id": payload.get("pair_id"),
        "questions": questions,
        "dimensions": dimensions,
        "weights": [float(w) for w in weights],
        "texts": [q.question for q in questions],
    }


def _requirements(pair_id: str | None) -> list[dict] | None:
    if not pair_id:
        return None
    matches = list(GAP_DIR.glob(f"{pair_id}__{pair_id}__*.json"))
    if not matches:
        return None
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    gaps = ((payload.get("gap_matrix") or {}).get("gaps")) or []
    return [dict(gap) for gap in gaps]


def _scalar_metrics(raw: dict) -> dict:
    keep = (
        "M1.P",
        "M1.S1",
        "M1.S2",
        "M4.P",
        "M2.P",
        "M2.S1",
        "M2.S2",
        "M3.P",
        "M3.S1",
        "M3.S2",
        "assignment_available",
        "n_questions",
    )
    return {key: raw.get(key) for key in keep}


def _figures(rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    full_rows = [row for row in rows if row["config"] == "full"]
    if full_rows:
        traces = np.asarray([row["lambda_t"] for row in full_rows], dtype=float)
        steps = np.arange(1, traces.shape[1] + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, traces.mean(axis=0), color="#2A6F97", label="mean lambda")
        ax.fill_between(
            steps,
            traces.min(axis=0),
            traces.max(axis=0),
            color="#2A6F97",
            alpha=0.2,
            label="min–max across pools",
        )
        ax.set_xlabel("selection step")
        ax.set_ylabel("lambda_t")
        ax.set_ylim(0.45, 0.95)
        ax.set_title("Fig 60. Adaptive lambda trajectory (full, exploratory primary-30)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig60_adaptive_lambda_trajectory.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    configs = list(named_ablation_grid())
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(configs))
    width = 0.15
    for offset, dimension in enumerate(POOL_DIMENSIONS):
        means = []
        for config in configs:
            subset = [row for row in rows if row["config"] == config]
            means.append(
                float(np.mean([row["dimension_counts"].get(dimension, 0) for row in subset]))
                if subset
                else 0.0
            )
        ax.bar(x + (offset - 2) * width, means, width=width, label=dimension)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("mean selected count")
    ax.set_title("Fig 61. Dimension counts by config (exploratory primary-30)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig61_dimension_counts_by_config.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prereg = _git_prereg()
    if not prereg:
        raise SystemExit("Refusing to run: adaptive_selection_prereg.md has no commit yet")
    pool_paths = sorted(p for p in POOL_DIR.glob("*.json") if not p.name.startswith("_"))
    if not pool_paths:
        raise SystemExit(f"No stored pools in {POOL_DIR}")
    pools = [_load_pool(path) for path in pool_paths]

    freeze_ok = False
    freeze_error = None
    tau_map: dict = {}
    try:
        freeze = verify_freeze()
        freeze_ok = not bool(freeze.get("post_hoc"))
        tau_map = dict(freeze.get("tau_p") or {})
        if freeze.get("post_hoc"):
            freeze_error = "freeze hash mismatch (post_hoc); metrics not treated as frozen"
            freeze_ok = False
    except FreezeError as exc:
        freeze_error = str(exc)
        freeze_ok = False

    from app.services.embeddings.service import SentenceTransformerEmbeddingService

    embeddings = SentenceTransformerEmbeddingService("all-MiniLM-L6-v2", device="cpu")
    grid = named_ablation_grid()
    rows: list[dict] = []
    for pool in pools:
        sim = embeddings.similarity_matrix(pool["texts"], pool["texts"])
        requirements = _requirements(pool["pair_id"])
        tau = tau_map.get(pool["pair_id"]) if freeze_ok else None
        for name, config in grid.items():
            result = select_adaptive(
                pool["questions"],
                sim,
                pool["weights"],
                pool["dimensions"],
                config,
                k=K,
            )
            selected_texts = [pool["texts"][i] for i in result.selected_indices]
            record = {
                "pair_id": pool["pair_id"],
                "config": name,
                "n_pool": len(pool["questions"]),
                "n_selected": len(result.selected_indices),
                "selected_indices": result.selected_indices,
                "lambda_t": result.lambda_t,
                "coverage_t": result.coverage_t,
                "dimension_counts": result.dimension_counts,
                "reserved_slot_active": result.reserved_slot_active,
                "shortfalls": [s.__dict__ for s in result.shortfalls],
                "mean_pairwise_similarity": _mean_pairwise(sim, result.selected_indices),
                "budget_met": len(result.shortfalls) == 0,
            }
            if freeze_ok:
                metrics = compute_full_metrics(
                    selected_texts,
                    embeddings,
                    requirements=requirements,
                    tau=float(tau) if tau is not None else None,
                )
                record["frozen_metrics"] = _scalar_metrics(metrics)
            rows.append(record)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _figures(rows)
    by_config: dict[str, dict] = {}
    for name in grid:
        subset = [row for row in rows if row["config"] == name]
        diag = {
            "n_pools": len(subset),
            "mean_pairwise_similarity": float(
                np.mean([row["mean_pairwise_similarity"] for row in subset])
            ),
            "pools_budget_met": int(sum(1 for row in subset if row["budget_met"])),
            "mean_dimension_counts": {
                dimension: float(
                    np.mean([row["dimension_counts"].get(dimension, 0) for row in subset])
                )
                for dimension in POOL_DIMENSIONS
            },
        }
        if freeze_ok:
            diag["frozen_metric_means"] = {}
            for key in ("M1.P", "M2.P", "M3.P", "M4.P"):
                vals = [
                    row["frozen_metrics"].get(key)
                    for row in subset
                    if row.get("frozen_metrics", {}).get(key) is not None
                ]
                diag["frozen_metric_means"][key] = float(np.mean(vals)) if vals else None
        by_config[name] = diag

    summary = {
        "exploratory": True,
        "label": "exploratory primary-30 replay; not a confirmatory test",
        "prereg_commit": prereg,
        "prereg_path": "docs/adaptive_selection_prereg.md",
        "pool_dir": str(POOL_DIR.relative_to(ROOT)),
        "n_pools": len(pools),
        "k": K,
        "metric_status": "frozen_eval_metrics" if freeze_ok else "non_frozen_diagnostics",
        "freeze_error": freeze_error,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_calls": 0,
        "cost_usd": 0.0,
        "primary_comparison": "full vs weighted_mmr",
        "budget_met_definition": (
            "pools_budget_met counts pools whose shortfall list is empty. "
            "Configs with no quotas (plain_mmr, weighted_mmr) are vacuously met. "
            "A zero_availability shortfall means a canonical dimension had no pool question."
        ),
        "by_config": by_config,
        "figures": [
            "docs/figures/fig60_adaptive_lambda_trajectory.png",
            "docs/figures/fig61_dimension_counts_by_config.png",
        ],
    }
    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "per_pool.json", {"rows": rows})
    print(json.dumps({k: summary[k] for k in ("metric_status", "n_pools", "prereg_commit", "freeze_error")}, indent=2))


if __name__ == "__main__":
    main()
