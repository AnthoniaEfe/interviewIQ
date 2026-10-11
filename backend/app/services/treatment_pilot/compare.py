"""pilot_compare: diagnostics, statistics, cost tables, figures, report fill-in."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.services.embeddings.service import EmbeddingService, build_embedding_service
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    load_extraction_records,
    load_gap_records,
    request_id,
    write_json,
)
from app.services.treatment_pilot.constants import (
    B0_EXPERIMENTS,
    B0_MEAN_COST_USD,
    B0_MEAN_LATENCY_S,
    COMBINED_DIR_NAME,
    EXTRACTION_COST_PER_PAIR_USD,
    EXTRACTION_EXPERIMENT,
    EXTRACTION_LATENCY_PER_PAIR_S,
    GAP_COST_PER_PAIR_USD,
    GAP_EXPERIMENT,
    GAP_LATENCY_PER_PAIR_S,
    USD_CNY_RATE,
)
from app.services.treatment_pilot.currency import usd_to_cny
from app.services.treatment_pilot.diagnostics import (
    b0_variability,
    decision_rule,
    diagnose_condition,
    load_b0_replicates,
    load_condition_records,
    paired_comparisons,
    selection_overlaps,
)
from app.services.treatment_pilot.figures import (
    figure_40_cost_latency,
    figure_41_redundancy,
    figure_42_coverage,
    figure_43_selection,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean_cost_latency(records: list[dict], cost_key="cost_usd", lat_key="usage") -> tuple[float | None, float | None]:
    if not records:
        return None, None
    costs = []
    lats = []
    for r in records:
        if cost_key in r and r[cost_key] is not None:
            costs.append(float(r[cost_key]))
        elif isinstance(r.get("usage"), dict):
            costs.append(float(r["usage"].get("estimated_cost") or 0))
        if isinstance(r.get("usage"), dict):
            lats.append(float(r["usage"].get("latency_ms") or 0) / 1000.0)
        elif r.get("selection_latency_ms") is not None:
            lats.append(float(r["selection_latency_ms"]) / 1000.0)
    return (
        float(np.mean(costs)) if costs else None,
        float(np.mean(lats)) if lats else None,
    )


def _stage_cost_rows(
    *,
    b0_cost: float,
    b0_lat: float,
    b1_cost: float | None,
    b1_lat: float | None,
    pool_cost: float | None,
    pool_lat: float | None,
    sel_lat: float | None,
    n_pairs: int,
) -> list[dict[str, Any]]:
    def row(stage, model, measured_on, cost, lat):
        cny = usd_to_cny(cost) if cost is not None else None
        return {
            "stage": stage,
            "model": model,
            "measured_on": measured_on,
            "cost_per_pair_usd": None if cost is None else round(cost, 8),
            "cost_per_pair_cny": cny,
            "cost_150_pairs_cny": None if cny is None else round(cny * 150, 4),
            "latency_per_pair_s": None if lat is None else round(lat, 4),
            "n_pairs": n_pairs,
        }

    rows = [
        row("B0_generation", "gpt-5.6-terra", "stored_150_mean", b0_cost, b0_lat),
        row("B1_generation", "gpt-5.6-terra", "pilot", b1_cost, b1_lat),
        row("T_pool_generation", "gpt-5.6-terra", "pilot", pool_cost, pool_lat),
        row("selection_local", None, "pilot", 0.0, sel_lat),
        row(
            "extraction_attributed",
            "gpt-5.6-luna",
            "stored_primary30",
            EXTRACTION_COST_PER_PAIR_USD,
            EXTRACTION_LATENCY_PER_PAIR_S,
        ),
        row(
            "gap_analysis_attributed",
            None,
            "stored_primary30",
            GAP_COST_PER_PAIR_USD,
            GAP_LATENCY_PER_PAIR_S,
        ),
    ]
    return rows


def _pipeline_costs(
    b0_gen: float,
    b1_gen: float | None,
    pool: float | None,
    b0_lat: float,
    b1_lat: float | None,
    pool_lat: float | None,
    sel_lat: float | None,
) -> dict[str, dict[str, float]]:
    ext_cny = usd_to_cny(EXTRACTION_COST_PER_PAIR_USD) or 0.0
    gap_cny = usd_to_cny(GAP_COST_PER_PAIR_USD) or 0.0
    b0_cny = usd_to_cny(b0_gen) or 0.0
    b1_cny = usd_to_cny(b1_gen) or 0.0
    pool_cny = usd_to_cny(pool) or 0.0
    return {
        "B0": {
            "extraction_cny": 0.0,
            "gap_analysis_cny": 0.0,
            "generation_cny": b0_cny,
            "selection_cny": 0.0,
            "total_usd": b0_gen,
            "total_cny": b0_cny,
            "latency_s": b0_lat,
        },
        "B1": {
            "extraction_cny": ext_cny,
            "gap_analysis_cny": 0.0,
            "generation_cny": b1_cny,
            "selection_cny": 0.0,
            "total_usd": (EXTRACTION_COST_PER_PAIR_USD + (b1_gen or 0.0)),
            "total_cny": ext_cny + b1_cny,
            "latency_s": EXTRACTION_LATENCY_PER_PAIR_S + (b1_lat or 0.0),
        },
        "T": {
            "extraction_cny": ext_cny,
            "gap_analysis_cny": gap_cny,
            "generation_cny": pool_cny,
            "selection_cny": 0.0,
            "total_usd": (
                EXTRACTION_COST_PER_PAIR_USD
                + GAP_COST_PER_PAIR_USD
                + (pool or 0.0)
            ),
            "total_cny": ext_cny + gap_cny + pool_cny,
            "latency_s": (
                EXTRACTION_LATENCY_PER_PAIR_S
                + GAP_LATENCY_PER_PAIR_S
                + (pool_lat or 0.0)
                + (sel_lat or 0.0)
            ),
        },
        "B2": {
            "extraction_cny": ext_cny,
            "gap_analysis_cny": gap_cny,
            "generation_cny": pool_cny,
            "selection_cny": 0.0,
            "total_usd": (
                EXTRACTION_COST_PER_PAIR_USD
                + GAP_COST_PER_PAIR_USD
                + (pool or 0.0)
            ),
            "total_cny": ext_cny + gap_cny + pool_cny,
            "latency_s": (
                EXTRACTION_LATENCY_PER_PAIR_S
                + GAP_LATENCY_PER_PAIR_S
                + (pool_lat or 0.0)
                + (sel_lat or 0.0)
            ),
        },
    }


def run_pilot_compare(
    *,
    root: Path,
    settings: Settings,
    b1_dir: Path | None,
    treatment_dir: Path | None,
    ablation_dir: Path | None,
    extraction_experiment: str = EXTRACTION_EXPERIMENT,
    gap_experiment: str = GAP_EXPERIMENT,
    embeddings: EmbeddingService | None = None,
) -> dict[str, Any]:
    combined = root / "experiments" / "results" / COMBINED_DIR_NAME
    combined.mkdir(parents=True, exist_ok=True)
    tables = combined / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    figs = root / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    if embeddings is None:
        embeddings = build_embedding_service(
            settings.embedding_model,
            settings.embedding_device,
            use_hashing_fallback=settings.embedding_model == "hashing-test",
        )

    ext_records = load_extraction_records(
        root / "experiments" / "results" / extraction_experiment
    )
    extraction_by_stem = {request_id(r): r for r in ext_records}
    gap_by_stem_files = load_gap_records(
        root / "experiments" / "results" / gap_experiment / "gap_analysis"
    )
    gap_by_stem = {k.replace(".json", ""): v for k, v in gap_by_stem_files.items()}
    primary_stems = set(extraction_by_stem)

    b0_by_stem = load_b0_replicates(root, primary_stems)
    # B0 mean record per stem (for diagnostics): average not meaningful for text;
    # use first replicate for per-question diagnostics and report variability separately.
    b0_records = {s: recs[0] for s, recs in b0_by_stem.items() if recs}
    b0_mean_records: dict[str, dict] = {}
    for s, recs in b0_by_stem.items():
        if not recs:
            continue
        # For coverage/redundancy on "B0 mean", diagnose each replicate and average metrics later.
        b0_mean_records[s] = recs[0]

    variability = b0_variability(b0_by_stem, embeddings)

    # Per-replicate B0 diagnostics then average means
    b0_rep_diags = []
    for i, exp_id in enumerate(B0_EXPERIMENTS):
        recs = {}
        d = root / "experiments" / "results" / exp_id / "baseline"
        for stem in primary_stems:
            path = d / f"{stem}.json"
            if path.is_file():
                recs[stem] = json.loads(path.read_text(encoding="utf-8"))
        b0_rep_diags.append(
            diagnose_condition(
                condition=f"B0_rep{i}",
                records=recs,
                extraction_by_stem=extraction_by_stem,
                gap_by_stem=gap_by_stem,
                embeddings=embeddings,
                has_metadata=False,
            )
        )

    diagnostics: dict[str, dict] = {}
    # Aggregate B0: mean of replicate per-pair metrics
    if b0_rep_diags:
        stems = sorted(primary_stems)
        agg_per_pair = []
        for stem in stems:
            reds, covs, lexs = [], [], []
            schema = True
            n_req = None
            for diag in b0_rep_diags:
                row = next((r for r in diag["per_pair"] if r["stem"] == stem), None)
                if not row:
                    continue
                schema = schema and row["schema_valid"]
                n_req = row["n_requirements"]
                if row["redundancy"]["mean_pairwise_sim"] is not None:
                    reds.append(row["redundancy"]["mean_pairwise_sim"])
                thr = row["coverage"]["by_threshold"].get("0.45") or {}
                if thr.get("coverage_rate") is not None:
                    covs.append(thr["coverage_rate"])
                if row["lexical_specificity"] is not None:
                    lexs.append(row["lexical_specificity"])
            # rebuild a synthetic per_pair row using means
            base = diagnose_condition(
                condition="B0",
                records={stem: b0_records[stem]} if stem in b0_records else {},
                extraction_by_stem=extraction_by_stem,
                gap_by_stem=gap_by_stem,
                embeddings=embeddings,
                has_metadata=False,
            )
            row0 = base["per_pair"][0] if base["per_pair"] else {
                "stem": stem,
                "n_questions": 10,
                "schema_valid": schema,
                "n_requirements": n_req,
                "redundancy": {
                    "mean_pairwise_sim": float(np.mean(reds)) if reds else None,
                    "mean_max_sim": None,
                    "n_redundant_pairs": None,
                    "n_questions": 10,
                },
                "coverage": {
                    "n_requirements": n_req,
                    "by_threshold": {
                        "0.45": {
                            "coverage_rate": float(np.mean(covs)) if covs else None,
                            "n_covered": None,
                            "share_gap": None,
                            "share_partial": None,
                            "share_strength": None,
                        }
                    },
                    "label": "proxy: embedding assignment, not human judgement",
                },
                "lexical_specificity": float(np.mean(lexs)) if lexs else None,
                "metadata": "not available",
            }
            if reds:
                row0["redundancy"]["mean_pairwise_sim"] = float(np.mean(reds))
            if covs:
                row0["coverage"]["by_threshold"]["0.45"]["coverage_rate"] = float(
                    np.mean(covs)
                )
            # average shares across replicates when available
            for thr_key in ("0.35", "0.45", "0.55"):
                shares = {"share_gap": [], "share_partial": [], "share_strength": [], "n_covered": [], "coverage_rate": []}
                for diag in b0_rep_diags:
                    row = next((r for r in diag["per_pair"] if r["stem"] == stem), None)
                    if not row:
                        continue
                    thr = row["coverage"]["by_threshold"].get(thr_key) or {}
                    for k in shares:
                        if thr.get(k) is not None:
                            shares[k].append(thr[k])
                row0["coverage"]["by_threshold"][thr_key] = {
                    k: (float(np.mean(v)) if v else None) for k, v in shares.items()
                }
            if lexs:
                row0["lexical_specificity"] = float(np.mean(lexs))
            row0["metadata"] = "not available"
            agg_per_pair.append(row0)
        from app.services.gap_sensitivity.metrics import bootstrap_mean_ci
        from app.services.treatment_pilot.constants import BOOTSTRAP_RESAMPLES, SEED

        red_arr = np.asarray(
            [
                r["redundancy"]["mean_pairwise_sim"]
                if r["redundancy"]["mean_pairwise_sim"] is not None
                else np.nan
                for r in agg_per_pair
            ],
            dtype=np.float64,
        )
        cov_arr = np.asarray(
            [
                (r["coverage"]["by_threshold"].get("0.45") or {}).get("coverage_rate")
                if (r["coverage"]["by_threshold"].get("0.45") or {}).get("coverage_rate")
                is not None
                else np.nan
                for r in agg_per_pair
            ],
            dtype=np.float64,
        )
        lex_arr = np.asarray(
            [
                r["lexical_specificity"] if r["lexical_specificity"] is not None else np.nan
                for r in agg_per_pair
            ],
            dtype=np.float64,
        )
        diagnostics["B0"] = {
            "condition": "B0",
            "n_pairs": len(agg_per_pair),
            "n_schema_valid": sum(1 for r in agg_per_pair if r["schema_valid"]),
            "redundancy_mean_pairwise": bootstrap_mean_ci(
                red_arr, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
            ),
            "coverage_0.45": bootstrap_mean_ci(
                cov_arr, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
            ),
            "lexical_specificity": bootstrap_mean_ci(
                lex_arr, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
            ),
            "per_pair": agg_per_pair,
            "replicates": [
                {
                    "experiment": B0_EXPERIMENTS[i],
                    "redundancy_mean_pairwise": d["redundancy_mean_pairwise"],
                    "coverage_0.45": d["coverage_0.45"],
                }
                for i, d in enumerate(b0_rep_diags)
            ],
        }

    cond_dirs = {}
    if b1_dir and (b1_dir / "baseline_profile_aware").is_dir():
        cond_dirs["B1"] = b1_dir / "baseline_profile_aware"
    if treatment_dir and (treatment_dir / "treatment").is_dir():
        cond_dirs["T"] = treatment_dir / "treatment"
    if ablation_dir:
        for c in ("B2", "S0", "S1", "S2"):
            if (ablation_dir / c).is_dir():
                cond_dirs[c] = ablation_dir / c

    for cond, d in cond_dirs.items():
        recs = load_condition_records(d)
        diagnostics[cond] = diagnose_condition(
            condition=cond,
            records=recs,
            extraction_by_stem=extraction_by_stem,
            gap_by_stem=gap_by_stem,
            embeddings=embeddings,
            has_metadata=cond in {"T", "B2", "S0", "S1", "S2"},
        )

    comparisons = paired_comparisons(diagnostics)
    decision = decision_rule(diagnostics, comparisons)

    overlaps = {}
    if "T" in cond_dirs and ablation_dir:
        overlaps = selection_overlaps(load_condition_records(cond_dirs["T"]), ablation_dir)

    # Costs
    b1_recs = list(load_condition_records(cond_dirs["B1"]).values()) if "B1" in cond_dirs else []
    t_recs = list(load_condition_records(cond_dirs["T"]).values()) if "T" in cond_dirs else []
    b1_cost, b1_lat = _mean_cost_latency(b1_recs)
    pool_cost, pool_lat = _mean_cost_latency(t_recs)
    sel_lats = [float(r.get("selection_latency_ms") or 0) / 1000 for r in t_recs]
    sel_lat = float(np.mean(sel_lats)) if sel_lats else None
    if ablation_dir and (ablation_dir / "B2").is_dir():
        b2 = list(load_condition_records(ablation_dir / "B2").values())
        b2_sel = [float(r.get("selection_latency_ms") or 0) / 1000 for r in b2]
        if b2_sel:
            sel_lat = float(np.mean(([sel_lat] if sel_lat else []) + b2_sel))

    stage_rows = _stage_cost_rows(
        b0_cost=B0_MEAN_COST_USD,
        b0_lat=B0_MEAN_LATENCY_S,
        b1_cost=b1_cost,
        b1_lat=b1_lat,
        pool_cost=pool_cost,
        pool_lat=pool_lat,
        sel_lat=sel_lat,
        n_pairs=len(primary_stems),
    )
    pipeline = _pipeline_costs(
        B0_MEAN_COST_USD, b1_cost, pool_cost, B0_MEAN_LATENCY_S, b1_lat, pool_lat, sel_lat
    )

    _write_csv(root / "docs" / "tables" / "stage_cost_latency.csv", stage_rows)
    _write_csv(tables / "stage_cost_latency.csv", stage_rows)

    # Diagnostics summary table
    diag_rows = []
    for cond, d in diagnostics.items():
        diag_rows.append(
            {
                "condition": cond,
                "n_pairs": d["n_pairs"],
                "n_schema_valid": d["n_schema_valid"],
                "redundancy_mean": (d["redundancy_mean_pairwise"] or {}).get("mean"),
                "redundancy_ci_low": (d["redundancy_mean_pairwise"] or {}).get("ci_low"),
                "redundancy_ci_high": (d["redundancy_mean_pairwise"] or {}).get("ci_high"),
                "coverage_0.45_mean": (d["coverage_0.45"] or {}).get("mean"),
                "coverage_0.45_ci_low": (d["coverage_0.45"] or {}).get("ci_low"),
                "coverage_0.45_ci_high": (d["coverage_0.45"] or {}).get("ci_high"),
                "lexical_mean": (d["lexical_specificity"] or {}).get("mean"),
                "lexical_ci_low": (d["lexical_specificity"] or {}).get("ci_low"),
                "lexical_ci_high": (d["lexical_specificity"] or {}).get("ci_high"),
            }
        )
    _write_csv(tables / "diagnostics_summary.csv", diag_rows)

    # Figures
    try:
        figure_40_cost_latency(figs / "fig40_pipeline_cost_latency.png", stage_rows, pipeline)
        figure_41_redundancy(figs / "fig41_redundancy.png", diagnostics)
        figure_42_coverage(figs / "fig42_coverage_proxy.png", diagnostics)
        figure_43_selection(figs / "fig43_selection_variants.png", diagnostics, overlaps)
    except Exception as exc:  # noqa: BLE001
        print(f"[pilot_compare] figure warning: {exc}")

    summary = {
        "experiment_id": COMBINED_DIR_NAME,
        "status": "completed",
        "stage": "pilot_compare",
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "embedding_model": settings.embedding_model,
        "usd_cny_rate": USD_CNY_RATE,
        "b0_variability": variability,
        "diagnostics_summary": diag_rows,
        "comparisons": comparisons,
        "overlaps": overlaps,
        "decision": decision,
        "pipeline_cost_per_pair": pipeline,
        "stage_cost_rows": stage_rows,
        "actual_b1_cost_usd": None if not b1_recs else round(sum(float(r.get("cost_usd") or 0) for r in b1_recs), 8),
        "actual_pool_cost_usd": None if not t_recs else round(sum(float(r.get("cost_usd") or 0) for r in t_recs), 8),
    }
    write_json(combined / "summary.json", summary)
    write_json(combined / "diagnostics.json", {
        k: {kk: vv for kk, vv in v.items() if kk != "per_pair"}
        | {"per_pair_n": len(v.get("per_pair") or [])}
        for k, v in diagnostics.items()
    })
    # Keep full per-pair separately
    write_json(combined / "diagnostics_full.json", diagnostics)

    _append_report_results(root, summary, diagnostics, comparisons, decision, variability, overlaps, pipeline)
    return summary


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _append_report_results(
    root: Path,
    summary: dict,
    diagnostics: dict,
    comparisons: dict,
    decision: dict,
    variability: dict,
    overlaps: dict,
    pipeline: dict,
) -> None:
    report = root / "docs" / "treatment_pilot_report.md"
    if not report.is_file():
        return
    text = report.read_text(encoding="utf-8")

    def replace_section(title: str, body: str) -> None:
        nonlocal text
        marker = f"## {title}"
        start = text.find(marker)
        if start < 0:
            text = text.rstrip() + f"\n\n{marker}\n\n{body}\n"
            return
        rest = text[start + len(marker) :]
        # next ## at beginning of line
        import re

        m = re.search(r"\n## ", rest)
        end = start + len(marker) + (m.start() if m else len(rest))
        text = text[:start] + f"## {title}\n\n{body}\n" + text[end:]

    # Costs
    actual_b1 = summary.get("actual_b1_cost_usd")
    actual_pool = summary.get("actual_pool_cost_usd")
    total = None
    if actual_b1 is not None and actual_pool is not None:
        total = actual_b1 + actual_pool
    cost_body = (
        f"- Actual B1 spend: {_fmt(actual_b1, 6)} USD / {_fmt(usd_to_cny(actual_b1), 4)} CNY\n"
        f"- Actual T pool spend: {_fmt(actual_pool, 6)} USD / {_fmt(usd_to_cny(actual_pool), 4)} CNY\n"
        f"- Actual paid total: {_fmt(total, 6)} USD / {_fmt(usd_to_cny(total), 4)} CNY\n"
        f"- USD_CNY_RATE: {USD_CNY_RATE}\n"
        f"- Git commit: `{summary.get('git_commit')}`\n"
    )
    replace_section("Actual cost and failures", cost_body)

    # Results tables
    lines = [
        "### Stage cost / latency",
        "",
        "| stage | model | cost/pair USD | cost/pair CNY | cost 150 CNY | latency s | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.get("stage_cost_rows") or []:
        lines.append(
            f"| {r['stage']} | {r['model']} | {_fmt(r['cost_per_pair_usd'], 6)} | "
            f"{_fmt(r['cost_per_pair_cny'], 4)} | {_fmt(r['cost_150_pairs_cny'], 2)} | "
            f"{_fmt(r['latency_per_pair_s'], 3)} | {r['n_pairs']} |"
        )
    lines += ["", "### Full-pipeline cost per pair", ""]
    for cond, p in pipeline.items():
        lines.append(
            f"- **{cond}**: {_fmt(p.get('total_usd'), 6)} USD / {_fmt(p.get('total_cny'), 4)} CNY; "
            f"latency {_fmt(p.get('latency_s'), 2)} s"
        )
    lines += ["", "### B0 replicate variability", ""]
    lines.append(f"- Mean pairwise embedding similarity across replicates: {_fmt(variability.get('mean_replicate_embedding_similarity'))}")
    lines.append(f"- Mean question-text Jaccard across replicates: {_fmt(variability.get('mean_question_text_jaccard'))}")
    lines += ["", "### Diagnostics (means with bootstrap 95% CIs)", ""]
    lines.append("| condition | redundancy mean [CI] | coverage@0.45 mean [CI] | lexical mean [CI] |")
    lines.append("|---|---|---|---|")
    for row in summary.get("diagnostics_summary") or []:
        lines.append(
            f"| {row['condition']} | {_fmt(row['redundancy_mean'])} "
            f"[{_fmt(row['redundancy_ci_low'])}, {_fmt(row['redundancy_ci_high'])}] | "
            f"{_fmt(row['coverage_0.45_mean'])} "
            f"[{_fmt(row['coverage_0.45_ci_low'])}, {_fmt(row['coverage_0.45_ci_high'])}] | "
            f"{_fmt(row['lexical_mean'])} "
            f"[{_fmt(row['lexical_ci_low'])}, {_fmt(row['lexical_ci_high'])}] |"
        )
    lines += ["", "### Paired comparisons (exploratory)", ""]
    for key, cmp in comparisons.items():
        red = cmp.get("redundancy") or {}
        cov = cmp.get("coverage_0.45") or {}
        lines.append(
            f"- **{key}** redundancy: rank-biserial={_fmt((red.get('wilcoxon') or {}).get('rank_biserial'))}, "
            f"p={_fmt((red.get('wilcoxon') or {}).get('p_value'))}, Holm p={_fmt(red.get('holm_p'))}; "
            f"diff CI=[{_fmt((red.get('diff_ci') or {}).get('ci_low'))}, {_fmt((red.get('diff_ci') or {}).get('ci_high'))}]"
        )
        lines.append(
            f"  coverage@0.45: rank-biserial={_fmt((cov.get('wilcoxon') or {}).get('rank_biserial'))}, "
            f"p={_fmt((cov.get('wilcoxon') or {}).get('p_value'))}, Holm p={_fmt(cov.get('holm_p'))}; "
            f"diff CI=[{_fmt((cov.get('diff_ci') or {}).get('ci_low'))}, {_fmt((cov.get('diff_ci') or {}).get('ci_high'))}]"
        )
    lines += ["", "### Selection overlaps (Jaccard means)", ""]
    for k, v in (overlaps or {}).items():
        lines.append(f"- {k}: {_fmt(v.get('mean'))} [{_fmt(v.get('ci_low'))}, {_fmt(v.get('ci_high'))}]")
    lines += [
        "",
        "### Figures",
        "",
        "- `docs/figures/fig40_pipeline_cost_latency.png`",
        "- `docs/figures/fig41_redundancy.png`",
        "- `docs/figures/fig42_coverage_proxy.png`",
        "- `docs/figures/fig43_selection_variants.png`",
    ]
    replace_section("Results", "\n".join(lines))

    replace_section(
        "Pilot decision-rule outcome",
        f"- **Case:** {decision.get('case')}\n"
        f"- recommend_scale: `{decision.get('recommend_scale')}`\n"
        f"- redundancy T≤B1: `{decision.get('redundancy_t_le_b1')}` "
        f"(T={_fmt(decision.get('redundancy_t_mean'))}, B1={_fmt(decision.get('redundancy_b1_mean'))})\n"
        f"- coverage T≥B1: `{decision.get('coverage_t_ge_b1')}` "
        f"(T={_fmt(decision.get('coverage_t_mean'))}, B1={_fmt(decision.get('coverage_b1_mean'))})\n"
        f"- Note: {decision.get('note')}\n",
    )

    # Thesis section — only what numbers support
    if decision.get("recommend_scale"):
        thesis = (
            "On the frozen proxy diagnostics, T was at least as good as B1 on both "
            "redundancy and requirement-coverage. This is an advisory signal only; "
            "proxies are not human quality ratings, and the fit-label control already "
            "showed that gap scores do not separate human fit labels."
        )
    elif decision.get("case") == "incomplete":
        thesis = (
            "Paid pilot outputs were incomplete at report time; no scaling recommendation "
            "is supported until B1 and T diagnostics exist."
        )
    else:
        thesis = (
            "On the frozen proxy diagnostics, T was **not** at least as good as B1 on both "
            "redundancy and requirement-coverage. The advisory rule therefore recommends "
            "diagnosing components (pool prompt, weight mapping, weighted MMR) before scaling "
            "to 150 pairs. This is consistent with not assuming gap-targeting improves quality "
            "after the null fit-label control."
        )
    replace_section("What this means for the thesis", thesis)

    report.write_text(text, encoding="utf-8")
