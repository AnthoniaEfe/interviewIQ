"""Combine primary + holdout gap-sensitivity results; write figures and report assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.domain import MatchLabel
from app.services.gap_sensitivity.audit import (
    build_audit_rows,
    build_false_gap_examples,
    write_audit_sheet,
    write_false_gap_examples,
)
from app.services.gap_sensitivity.evidence import CONDITION_A, CONDITION_D, MAIN_CONDITIONS
from app.services.gap_sensitivity.figures import (
    figure_17_gap_shares,
    figure_18_gap_rate_distributions,
    figure_19_true_vs_mismatched,
    figure_20_dose_response,
    figure_21_transition_heatmap,
)
from app.services.gap_sensitivity.io_utils import (
    load_extraction_records,
    project_root,
    resolve_cv_text,
    write_json,
)
from app.services.gap_sensitivity.scoring import PairScoreResult, RequirementScore
from app.schemas.domain import RequirementPriority


def _load_summary(exp_dir: Path) -> dict[str, Any]:
    return json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))


def _reconstruct_results(
    exp_dir: Path, conditions: list[str]
) -> tuple[list[dict], dict[str, list[PairScoreResult]], list[str | None]]:
    """Rebuild lightweight PairScoreResult lists from written pair JSON + extraction."""
    summary = _load_summary(exp_dir)
    source = project_root() / "experiments" / "results" / summary["source_experiment"]
    records = load_extraction_records(source)
    pairs_dir = project_root() / summary.get("config", {}).get("data", {}).get(
        "pairs_dir", "experiments/datasets/resume_job_fit/experiment_pairs"
    )
    cv_texts = [resolve_cv_text(r, pairs_dir) for r in records]

    true_results: dict[str, list[PairScoreResult]] = {c: [] for c in conditions}
    for rec in records:
        rid = f"{rec.get('profile_id')}__{rec.get('jd_id')}__{rec.get('seniority')}.json"
        payload = json.loads((exp_dir / "pairs" / rid).read_text(encoding="utf-8"))
        for cond in conditions:
            block = payload["conditions"][cond]
            scores = []
            for r in block["requirements"]:
                scores.append(
                    RequirementScore(
                        job_competency=r["job_competency"],
                        priority=RequirementPriority(r["priority"]),
                        tfidf=float(r["tfidf"]),
                        bm25_raw=float(r["bm25_raw"]),
                        bm25_norm=float(r["bm25_norm"]),
                        bm25_norm_fixed=r.get("bm25_norm_fixed"),
                        semantic=float(r["semantic"]),
                        combined=float(r["combined"]),
                        combined_fixed_bm25=r.get("combined_fixed_bm25"),
                        match_label=MatchLabel(r["match_label"]),
                        match_label_fixed_bm25=None,
                        match_label_semantic_only=MatchLabel(
                            r["match_label_semantic_only"]
                        ),
                        matched_candidate=r.get("matched_candidate"),
                    )
                )
            true_results[cond].append(
                PairScoreResult(
                    scores=scores,
                    bm25_max=float(block["bm25_max"]),
                    evidence_size=int(block["evidence_size"]),
                    evidence_texts=[],
                )
            )
    return records, true_results, cv_texts


def _per_pair_arrays(
    true_results: dict[str, list[PairScoreResult]], conditions: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sizes: dict[str, np.ndarray] = {}
    rates: dict[str, np.ndarray] = {}
    for cond in conditions:
        s, r = [], []
        for pr in true_results[cond]:
            s.append(pr.evidence_size)
            if not pr.scores:
                r.append(float("nan"))
            else:
                n_gap = sum(
                    1
                    for x in pr.scores
                    if x.match_label in {MatchLabel.WEAK, MatchLabel.MISSING}
                )
                r.append(n_gap / len(pr.scores))
        sizes[cond] = np.asarray(s, dtype=np.float64)
        rates[cond] = np.asarray(r, dtype=np.float64)
    return sizes, rates


def combine_results(
    *,
    primary_dir: Path,
    holdout_dir: Path,
    combined_dir: Path,
    figures_dir: Path,
) -> Path:
    primary = _load_summary(primary_dir)
    holdout = _load_summary(holdout_dir)
    if primary.get("status") != "completed" or holdout.get("status") != "completed":
        raise SystemExit(
            "Cannot combine: one or both splits failed "
            f"(primary={primary.get('status')}, holdout={holdout.get('status')})"
        )

    conditions = list(primary.get("conditions") or MAIN_CONDITIONS)
    combined_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Cross-split overview CSV
    rows = []
    for summary in (primary, holdout):
        split = summary["split"]
        for cond in conditions:
            m = summary["metrics"][cond]
            lc = m["label_counts"]
            tc = m["three_category_counts"]
            total = sum(tc.values()) or 1
            sa = summary["size_adjusted"][cond]
            rows.append(
                {
                    "split": split,
                    "condition": cond,
                    "n_requirements": sum(lc.values()),
                    "strong": lc["strong_match"],
                    "partial": lc["partial_match"],
                    "weak": lc["weak_match"],
                    "missing": lc["missing"],
                    "gap_share": m["pooled_gap_share"],
                    "gap_share_ci_low": m["pooled_gap_share_bootstrap"]["ci_low"],
                    "gap_share_ci_high": m["pooled_gap_share_bootstrap"]["ci_high"],
                    "mean_per_pair_gap_rate": m["per_pair_gap_rate"]["mean"],
                    "mean_per_pair_ci_low": m["per_pair_gap_rate_bootstrap"]["ci_low"],
                    "mean_per_pair_ci_high": m["per_pair_gap_rate_bootstrap"]["ci_high"],
                    "strength_share": tc["strength"] / total,
                    "partial_share": tc["partial"] / total,
                    "evidence_size_mean": m["evidence_size"]["mean"],
                    "mismatched_gap_share": sa["mismatched_pooled_gap_share"],
                    "size_adjusted_effect_pooled": sa[
                        "size_adjusted_effect_pooled_share"
                    ],
                    "spearman_rho": m["dose_response"]["rho"],
                    "spearman_p": m["dose_response"]["p_value"],
                }
            )
    with (combined_dir / "overview_by_condition.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Wilcoxon combined
    wrows = []
    for summary in (primary, holdout):
        for cond, w in summary["wilcoxon_vs_A"].items():
            wrows.append(
                {
                    "split": summary["split"],
                    "comparison": f"{cond} vs A",
                    **w,
                }
            )
        for cond, sa in summary["size_adjusted"].items():
            w = sa["wilcoxon_true_vs_mismatched"]
            wrows.append(
                {
                    "split": summary["split"],
                    "comparison": f"{cond} true vs mismatched",
                    **w,
                }
            )
    with (combined_dir / "wilcoxon.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(wrows[0].keys()))
        writer.writeheader()
        writer.writerows(wrows)

    write_json(
        combined_dir / "summary.json",
        {
            "stage": "gap_sensitivity_combined",
            "primary_experiment": primary["experiment_id"],
            "holdout_experiment": holdout["experiment_id"],
            "primary": {
                "experiment_id": primary["experiment_id"],
                "regression": primary["regression"],
                "metrics": primary["metrics"],
                "size_adjusted": primary["size_adjusted"],
                "wilcoxon_vs_A": primary["wilcoxon_vs_A"],
            },
            "holdout": {
                "experiment_id": holdout["experiment_id"],
                "regression": holdout["regression"],
                "metrics": holdout["metrics"],
                "size_adjusted": holdout["size_adjusted"],
                "wilcoxon_vs_A": holdout["wilcoxon_vs_A"],
            },
            "conditions": conditions,
        },
    )

    # Reconstruct for audit / figures
    p_records, p_true, p_cvs = _reconstruct_results(primary_dir, conditions)
    h_records, h_true, h_cvs = _reconstruct_results(holdout_dir, conditions)

    audit_rows = build_audit_rows(
        split="primary-30",
        records=p_records,
        cv_texts=p_cvs,
        true_results=p_true,
        n_pairs=10,
    ) + build_audit_rows(
        split="holdout-30",
        records=h_records,
        cv_texts=h_cvs,
        true_results=h_true,
        n_pairs=10,
    )
    write_audit_sheet(
        combined_dir / "audit_sheet.csv",
        combined_dir / "audit_sheet.md",
        audit_rows,
    )
    examples = build_false_gap_examples(
        split_results=[
            ("primary-30", p_records, p_true),
            ("holdout-30", h_records, h_true),
        ],
        top_n=25,
    )
    write_false_gap_examples(combined_dir / "false_gap_examples.md", examples)

    # Figures
    figure_17_gap_shares(
        figures_dir / "fig17_gap_category_shares.png",
        primary,
        holdout,
        conditions,
    )

    p_sizes, p_rates = _per_pair_arrays(p_true, conditions)
    h_sizes, h_rates = _per_pair_arrays(h_true, conditions)
    p_ci = {
        c: (
            primary["metrics"][c]["per_pair_gap_rate_bootstrap"]["mean"],
            primary["metrics"][c]["per_pair_gap_rate_bootstrap"]["ci_low"],
            primary["metrics"][c]["per_pair_gap_rate_bootstrap"]["ci_high"],
        )
        for c in conditions
    }
    h_ci = {
        c: (
            holdout["metrics"][c]["per_pair_gap_rate_bootstrap"]["mean"],
            holdout["metrics"][c]["per_pair_gap_rate_bootstrap"]["ci_low"],
            holdout["metrics"][c]["per_pair_gap_rate_bootstrap"]["ci_high"],
        )
        for c in conditions
    }
    figure_18_gap_rate_distributions(
        figures_dir / "fig18_per_pair_gap_rate.png",
        p_rates,
        h_rates,
        p_ci,
        h_ci,
        conditions,
    )
    figure_19_true_vs_mismatched(
        figures_dir / "fig19_true_vs_mismatched.png",
        primary,
        holdout,
        conditions,
    )
    figure_20_dose_response(
        figures_dir / "fig20_size_vs_gap_rate.png",
        p_sizes,
        p_rates,
        h_sizes,
        h_rates,
        {
            CONDITION_A: primary["metrics"][CONDITION_A]["dose_response"]["rho"],
            CONDITION_D: primary["metrics"][CONDITION_D]["dose_response"]["rho"],
        },
        {
            CONDITION_A: holdout["metrics"][CONDITION_A]["dose_response"]["rho"],
            CONDITION_D: holdout["metrics"][CONDITION_D]["dose_response"]["rho"],
        },
    )
    figure_21_transition_heatmap(
        figures_dir / "fig21_A_to_D_transition.png",
        primary["transitions"][f"{CONDITION_A}->{CONDITION_D}"],
        holdout["transitions"][f"{CONDITION_A}->{CONDITION_D}"],
    )

    return combined_dir
