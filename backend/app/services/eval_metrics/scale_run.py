"""Part B: statistics, frozen decision rule, diagnostics, cost projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.services.embeddings.service import build_embedding_service
from app.services.eval_metrics.constants import (
    COMBINED_DIR_NAME,
    DEFAULT_SCALE_BUDGET_CNY,
    EXTRACTION_PRIOR_PER_PAIR_USD,
    PRIMARY_METRICS,
    USD_CNY_RATE,
)
from app.services.eval_metrics.decision import (
    apply_holm_across,
    compare_metric,
    decision_rule,
)
from app.services.eval_metrics.diagnostics import (
    d1_pool_vs_selection,
    d2_weight_audit,
    d3_lambda_sensitivity,
    d4_pool_size_sensitivity,
    d5_failures,
    d6_robustness,
    d7_strata,
    d8_attribution,
)
from app.services.eval_metrics.freeze import FreezeError, verify_freeze
from app.services.eval_metrics.loaders import (
    index_gap_by_pair,
    load_condition_records,
    load_json,
    pair_stem,
    primary_pair_ids,
    requirements_from_gap,
    resolve_pilot_dirs,
)
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    project_root,
    utc_experiment_id,
    write_json,
)
from app.services.treatment_pilot.currency import usd_to_cny
from app.services.treatment_pilot.constants import (
    EXTRACTION_COST_PER_PAIR_USD,
    EXTRACTION_LATENCY_PER_PAIR_S,
)


def _find_latest_stage(stage: str) -> Path | None:
    root = project_root() / "experiments" / "results"
    best = None
    for d in sorted(root.iterdir()):
        s = d / "summary.json"
        if not s.is_file():
            continue
        try:
            data = load_json(s)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("stage") == stage and data.get("status") == "completed":
            best = d
    return best


def run_scale_decision_stage(
    *,
    settings: Settings,
    allow_post_hoc: bool = False,
    scale_budget_cny: float = DEFAULT_SCALE_BUDGET_CNY,
    pilot_experiment: str | None = None,
) -> dict[str, Any]:
    try:
        freeze = verify_freeze(allow_post_hoc=allow_post_hoc)
    except FreezeError as exc:
        raise SystemExit(str(exc)) from exc

    post_hoc = bool(freeze.get("post_hoc"))
    tau_map = dict(freeze.get("tau_p") or {})
    root = project_root()

    if pilot_experiment:
        pilot_dir = root / "experiments" / "results" / pilot_experiment
    else:
        pilot_dir = _find_latest_stage("metrics_pilot")
    if pilot_dir is None or not (pilot_dir / "conditions_data.json").is_file():
        raise SystemExit(
            "metrics_pilot results not found. Run --stage metrics_pilot after freeze."
        )

    conditions_data = load_json(pilot_dir / "conditions_data.json")
    # Also load noise floor from freeze baseline summary if present
    noise_floor = {}
    freeze_exp = freeze.get("validity_experiment_id")
    if freeze_exp:
        base_sum = root / "experiments" / "results" / freeze_exp / "summary.json"
        if base_sum.is_file():
            noise_floor = (load_json(base_sum).get("primary30_noise_floor") or {})

    def series(cond: str, metric: str) -> np.ndarray:
        rows = conditions_data.get(cond) or []
        # align by pair_id
        return rows

    prim_ids = primary_pair_ids()

    def aligned(cond: str, metric: str) -> np.ndarray:
        by_pid = {r["pair_id"]: r for r in (conditions_data.get(cond) or [])}
        vals = []
        for pid in prim_ids:
            r = by_pid.get(pid)
            vals.append(r.get(metric) if r else None)
        return np.asarray(
            [float(v) if v is not None else np.nan for v in vals], dtype=np.float64
        )

    comparisons_12: list[dict[str, Any]] = []
    outcomes = {"B0": {}, "B1": {}, "B2": {}}
    for baseline in ("B1", "B2", "B0"):
        for metric in PRIMARY_METRICS:
            row = compare_metric(
                aligned("T", metric),
                aligned(baseline, metric),
                metric,
            )
            row["baseline"] = baseline
            comparisons_12.append(row)
            outcomes[baseline][metric] = row["outcome"]

    comparisons_12 = apply_holm_across(comparisons_12)

    # Cost projection from pilot summaries
    dirs = resolve_pilot_dirs()
    b1_cost = pool_cost = None
    b1_lat = pool_lat = None
    if dirs.get("B1"):
        s = load_json(dirs["B1"] / "summary.json")
        n = float(s.get("combinations_ok") or 30)
        b1_cost = float(s.get("total_estimated_cost_usd") or 0) / n
        # latency if present
        b1_lat = s.get("average_latency_ms")
        if b1_lat is not None:
            b1_lat = float(b1_lat) / 1000.0
    if dirs.get("T"):
        s = load_json(dirs["T"] / "summary.json")
        n = float(s.get("combinations_ok") or 30)
        pool_cost = float(s.get("total_estimated_cost_usd") or 0) / n
        pool_lat = s.get("average_latency_ms")
        if pool_lat is not None:
            pool_lat = float(pool_lat) / 1000.0

    n_missing_extract = 90  # of 150 lacking extraction
    extract_per = EXTRACTION_COST_PER_PAIR_USD
    proj = {
        "label": "linear_extrapolation",
        "extraction_90_usd": extract_per * n_missing_extract,
        "b1_150_usd": (b1_cost * 150) if b1_cost is not None else None,
        "t_pool_150_usd": (pool_cost * 150) if pool_cost is not None else None,
        "per_pair_latency_s": {
            "extraction": EXTRACTION_LATENCY_PER_PAIR_S,
            "B1": b1_lat,
            "T_pool": pool_lat,
        },
    }
    total_usd = 0.0
    known = True
    for key in ("extraction_90_usd", "b1_150_usd", "t_pool_150_usd"):
        if proj[key] is None:
            known = False
        else:
            total_usd += float(proj[key])
    proj["total_usd"] = round(total_usd, 6) if known else None
    proj["total_cny"] = usd_to_cny(total_usd, USD_CNY_RATE) if known else None
    for key in ("extraction_90_usd", "b1_150_usd", "t_pool_150_usd"):
        if proj[key] is not None:
            proj[key.replace("_usd", "_cny")] = usd_to_cny(proj[key], USD_CNY_RATE)

    decision = decision_rule(
        outcomes_vs_b1=outcomes["B1"],
        outcomes_vs_b2=outcomes["B2"],
        outcomes_vs_b0=outcomes["B0"],
        projected_cost_cny=proj["total_cny"],
        scale_budget_cny=scale_budget_cny,
    )

    # Diagnostics
    embeddings = build_embedding_service(
        settings.embedding_model, settings.embedding_device
    )
    gap_index = index_gap_by_pair()
    pool_recs = load_condition_records(dirs["T"], "pool") if dirs.get("T") else {}
    t_recs = load_condition_records(dirs["T"], "treatment") if dirs.get("T") else {}
    b2_recs = (
        load_condition_records(dirs["ablation"], "B2") if dirs.get("ablation") else {}
    )

    d1_list = []
    d2_list = []
    d3_list = []
    d4_list = []
    for pid in prim_ids:
        gap = gap_index.get(pid)
        if not gap:
            continue
        reqs = requirements_from_gap(gap)
        tau = tau_map.get(pid)
        if tau is None:
            continue
        stem = pair_stem(gap.get("profile_id", ""), gap.get("jd_id", ""), "mid")
        pool = pool_recs.get(pid) or pool_recs.get(stem)
        sel = t_recs.get(pid) or t_recs.get(stem)
        if not pool or not sel:
            continue
        d1_list.append(
            d1_pool_vs_selection(
                embeddings=embeddings,
                pool_record=pool,
                selected_record=sel,
                requirements=reqs,
                tau=tau,
            )
        )
        d2_list.append(d2_weight_audit(pool, sel, reqs))
        d2_list[-1]["B2"] = d2_weight_audit(
            pool, b2_recs.get(pid) or b2_recs.get(stem) or {}, reqs
        )
        d3_list.append(
            d3_lambda_sensitivity(
                embeddings=embeddings,
                pool_record=pool,
                requirements=reqs,
                tau=tau,
            )
        )
        d4_list.append(
            d4_pool_size_sensitivity(
                embeddings=embeddings,
                pool_record=pool,
                requirements=reqs,
                tau=tau,
            )
        )

    # Aggregate D1
    def _mean_key(items, path):
        vals = []
        for it in items:
            cur = it
            for p in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(p)
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else None

    d1_agg = {
        "n": len(d1_list),
        "pool_M3.P": _mean_key(d1_list, ["pool", "M3.P"]),
        "T_M3.P": _mean_key(d1_list, ["selected_T", "M3.P"]),
        "oracle_M3.P": _mean_key(d1_list, ["oracle_M3.P", "M3.P"]),
        "oracle_vs_T_M3": None,
        "pool_M2.P": _mean_key(d1_list, ["pool", "M2.P"]),
        "T_M2.P": _mean_key(d1_list, ["selected_T", "M2.P"]),
        "oracle_M2.P": _mean_key(d1_list, ["oracle_M2.P", "M2.P"]),
    }
    if d1_agg["oracle_M3.P"] is not None and d1_agg["T_M3.P"] is not None:
        d1_agg["oracle_vs_T_M3"] = d1_agg["oracle_M3.P"] - d1_agg["T_M3.P"]

    d5 = d5_failures(list(pool_recs.values()))

    # D6: sensitivity from stored pilot per-pair sensitivity blocks
    treatment = {m: aligned("T", m) for m in PRIMARY_METRICS}
    baselines = {
        "B0": {m: aligned("B0", m) for m in PRIMARY_METRICS},
        "B1": {m: aligned("B1", m) for m in PRIMARY_METRICS},
        "B2": {m: aligned("B2", m) for m in PRIMARY_METRICS},
    }
    # Build threshold-free series if present on T/B1 rows
    sens_series: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for variant in ("threshold_free", "fixed_0.45"):
        sens_series[variant] = {}
        for cond in ("T", "B1", "B2", "B0"):
            key = "threshold_free" if variant == "threshold_free" else "fixed_0.45"
            arrs = {"M2.P": [], "M3.P": []}
            by_pid = {r["pair_id"]: r for r in (conditions_data.get(cond) or [])}
            for pid in prim_ids:
                r = by_pid.get(pid) or {}
                sens = (r.get("sensitivity") or {}).get(key) or {}
                arrs["M2.P"].append(sens.get("M2.P"))
                arrs["M3.P"].append(sens.get("M3.P"))
            sens_series[variant][cond] = {
                m: np.asarray(
                    [float(v) if v is not None else np.nan for v in arrs[m]],
                    dtype=np.float64,
                )
                for m in arrs
            }
    d6 = d6_robustness(
        treatment=treatment, baselines=baselines, sensitivity_series=sens_series
    )
    d7 = d7_strata(conditions_data)
    d8 = d8_attribution(d1_agg, {}, d6, fit_label_null=True)

    # Lambda figure aggregate
    lambda_agg: dict[str, dict[str, list[float | None]]] = {
        "T": {"M3.P": [], "M2.P": []},
        "B2": {"M3.P": [], "M2.P": []},
    }
    lambdas = [0.3, 0.5, 0.7, 0.9, 1.0]
    for lam in lambdas:
        for metric in ("M3.P", "M2.P"):
            tvals = [
                (d.get("T_weighted") or {}).get(str(lam), {}).get(metric)
                for d in d3_list
                if d.get("T_weighted")
            ]
            bvals = [
                (d.get("B2_priority") or {}).get(str(lam), {}).get(metric)
                for d in d3_list
                if d.get("B2_priority")
            ]
            lambda_agg["T"][metric].append(
                float(np.mean([v for v in tvals if v is not None])) if any(v is not None for v in tvals) else None
            )
            lambda_agg["B2"][metric].append(
                float(np.mean([v for v in bvals if v is not None])) if any(v is not None for v in bvals) else None
            )

    exp_id = utc_experiment_id()
    out_dir = root / "experiments" / "results" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": exp_id,
        "stage": "scale_decision",
        "status": "completed",
        "post_hoc": post_hoc,
        "post_hoc_tag": freeze.get("post_hoc_tag") if post_hoc else None,
        "spec_sha256": freeze.get("spec_sha256"),
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "pilot_metrics_experiment": pilot_dir.name,
        "comparisons": comparisons_12,
        "win_tie_loss": outcomes,
        "decision": decision,
        "noise_floor_V4": noise_floor,
        "cost_projection": proj,
        "scale_budget_cny": scale_budget_cny,
        "diagnostics": {
            "D1": d1_agg,
            "D2_n": len(d2_list),
            "D2_sample": d2_list[:3],
            "D3_lambda": {"lambdas": lambdas, "means": lambda_agg},
            "D4_n": len(d4_list),
            "D4_sample": d4_list[:3],
            "D5": d5,
            "D6": d6,
            "D7": d7,
            "D8": d8,
        },
        "validity_caveat": (
            "T is conditioned on the gap matrix while B1 is only told in words to cover gaps; "
            "some advantage on M2/M3 is partly by construction. Wins on gap-targeting and "
            "coverage are necessary but not sufficient; parity or better on M1/M4 at equal "
            "or higher coverage is the more informative result."
        ),
        "advisory": True,
        "freeze_not_blind_to_prior_pilot": bool(
            freeze.get("note_pilot_may_have_been_inspected_previously")
        ),
    }
    if post_hoc:
        summary["tag"] = "POST-HOC, not primary"

    write_json(out_dir / "summary.json", summary)
    combined = root / "experiments" / "results" / COMBINED_DIR_NAME
    combined.mkdir(parents=True, exist_ok=True)
    write_json(combined / "scale_decision.json", summary)

    # Figures 53–56
    try:
        from app.services.eval_metrics.figures import (
            figure_53_win_matrix,
            figure_54_pool_oracle,
            figure_55_lambda_sensitivity_labeled,
            figure_56_scale_cost,
        )

        figs = root / "docs" / "figures"
        holm = {"B0": {}, "B1": {}, "B2": {}}
        for c in comparisons_12:
            holm[c["baseline"]][c["metric"]] = c.get("wilcoxon_p_holm")
        figure_53_win_matrix(figs / "fig53_win_tie_loss.png", outcomes, holm)
        figure_54_pool_oracle(
            figs / "fig54_pool_oracle.png",
            [
                {"label": "pool", "M3.P": d1_agg["pool_M3.P"], "M2.P": d1_agg["pool_M2.P"]},
                {"label": "T selected", "M3.P": d1_agg["T_M3.P"], "M2.P": d1_agg["T_M2.P"]},
                {"label": "oracle M3", "M3.P": d1_agg["oracle_M3.P"], "M2.P": None},
                {"label": "oracle M2", "M3.P": None, "M2.P": d1_agg["oracle_M2.P"]},
            ],
        )
        figure_55_lambda_sensitivity_labeled(
            figs / "fig55_lambda_sensitivity.png",
            lambdas,
            lambda_agg,
            metric="M3.P",
        )
        cost_cny = {
            k.replace("_cny", ""): v
            for k, v in proj.items()
            if k.endswith("_cny") and isinstance(v, (int, float))
        }
        if proj.get("total_cny") is not None:
            cost_cny["total"] = proj["total_cny"]
        figure_56_scale_cost(figs / "fig56_scale_cost_cny.png", cost_cny)
    except Exception as exc:  # noqa: BLE001
        print(f"[scale_decision] figure warning: {exc}")

    try:
        from app.services.eval_metrics.report import write_scale_decision_report

        write_scale_decision_report(summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[scale_decision] report warning: {exc}")

    print(
        f"[scale_decision] verdict={decision['verdict']} "
        f"clauses={[c['holds'] for c in decision['clauses']]}"
    )
    print(f"[scale_decision] summary -> {out_dir / 'summary.json'}")
    return summary
