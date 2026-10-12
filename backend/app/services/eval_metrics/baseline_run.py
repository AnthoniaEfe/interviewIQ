"""Part A: validity checks, baseline metrics on stored B0, write freeze."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.services.embeddings.service import build_embedding_service
from app.services.eval_metrics.assignment import (
    calibrate_tau_p,
    hash_tau_map,
    sample_assignment_audit_rows,
)
from app.services.eval_metrics.compute import compute_full_metrics, embed_texts
from app.services.eval_metrics.constants import (
    ASSIGNMENT_AUDIT_REL,
    AUDIT_SAMPLE_N,
    B0_150_EXPERIMENT,
    B0_PRIMARY_REPLICATES,
    BOOTSTRAP_RESAMPLES,
    COMBINED_DIR_NAME,
    SEED,
    USD_CNY_RATE,
)
from app.services.eval_metrics.freeze import (
    build_spec_payload,
    write_freeze,
    write_spec,
)
from app.services.eval_metrics.loaders import (
    all_150_pairs,
    b0_150_index,
    b0_replicate_records,
    detect_pilot_summaries,
    index_gap_by_pair,
    pair_stem,
    primary_pair_ids,
    question_texts,
    requirements_from_gap,
)
from app.services.eval_metrics.validity import (
    v1_known_answer_checks,
    v2_null_control,
    v3_duplicate_injection,
    v4_noise_floor,
    write_assignment_audit_csv,
)
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    project_root,
    utc_experiment_id,
    write_json,
)
from app.services.gap_sensitivity.metrics import bootstrap_mean_ci


def _summarise_metric(values: list[float | None]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return bootstrap_mean_ci(arr, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED)


def run_baseline_and_freeze(
    *,
    settings: Settings,
    extract_missing: bool = False,
    confirm_spend: bool = False,
    max_cost: float = 2.0,
    dry_run: bool = False,
    with_dimension_labels: bool = False,
) -> dict[str, Any]:
    """Compute baseline metrics + validity, write spec and freeze file.

    Never opens pilot question files.
    """
    root = project_root()
    exp_id = utc_experiment_id()
    out_dir = root / "experiments" / "results" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_pair").mkdir(parents=True, exist_ok=True)

    pilot_presence = detect_pilot_summaries()
    print(
        f"[metrics_freeze] pilot stage dirs detected (stage/status only): "
        f"{pilot_presence or 'none'}"
    )
    print("[metrics_freeze] not opening any pilot question files before freeze")

    embeddings = build_embedding_service(
        settings.embedding_model, settings.embedding_device
    )
    device = getattr(embeddings, "device", settings.embedding_device)

    # --- Load B0 150 ---
    b0_index = b0_150_index()
    pairs_meta = {str(p["pair_id"]): p for p in all_150_pairs()}
    gap_index = index_gap_by_pair()
    print(
        f"[metrics_freeze] B0 sets={len(b0_index)} gap_matrices={len(gap_index)} "
        f"embedding={settings.embedding_model} device={device}"
    )

    # Embed all questions once
    pair_ids_150 = sorted(b0_index.keys())
    q_texts: dict[str, list[str]] = {}
    q_emb: dict[str, np.ndarray] = {}
    all_other_emb_list: list[np.ndarray] = []
    all_other_pids: list[str] = []
    for pid in pair_ids_150:
        texts = question_texts(b0_index[pid])
        q_texts[pid] = texts
        E = embed_texts(embeddings, texts)
        q_emb[pid] = E
        for row in E:
            all_other_emb_list.append(row)
            all_other_pids.append(pid)

    # Stack for leave-one-out tau
    if all_other_emb_list:
        all_q_stack = np.stack(all_other_emb_list, axis=0)
    else:
        all_q_stack = np.zeros((0, 1))

    # Requirements + tau for pairs with gap
    req_by_pair: dict[str, list[dict]] = {}
    req_emb: dict[str, np.ndarray] = {}
    tau_map: dict[str, float] = {}
    for pid, gap_rec in gap_index.items():
        if pid not in q_emb:
            continue
        reqs = requirements_from_gap(gap_rec)
        if not reqs:
            continue
        req_by_pair[pid] = reqs
        R = embed_texts(
            embeddings, [str(r.get("job_competency") or "") for r in reqs]
        )
        req_emb[pid] = R
        # leave-one-out: exclude this pair's own questions
        mask = np.asarray([p != pid for p in all_other_pids], dtype=bool)
        other = all_q_stack[mask] if mask.size else all_q_stack
        tau_map[pid] = calibrate_tau_p(
            req_embeddings=R, other_question_embeddings=other
        )

    # Optional extract_missing (gated)
    extraction_info: dict[str, Any] = {
        "requested": extract_missing,
        "ran": False,
        "cost_usd": 0.0,
        "cost_cny": 0.0,
        "n_pairs": 0,
        "skipped_reason": None,
    }
    if extract_missing:
        from app.services.eval_metrics.extract_missing import maybe_extract_missing

        extraction_info = maybe_extract_missing(
            settings=settings,
            missing_pair_ids=[p for p in pair_ids_150 if p not in gap_index],
            confirm_spend=confirm_spend,
            max_cost=max_cost,
            dry_run=dry_run,
        )
        if extraction_info.get("ran") and extraction_info.get("gap_experiment"):
            # Reload gap index including new
            from app.services.eval_metrics.loaders import load_gap_record

            gap_exp = extraction_info["gap_experiment"]
            for pid in extraction_info.get("extracted_pair_ids") or []:
                meta = pairs_meta.get(pid) or {}
                stem = pair_stem(
                    meta.get("profile_id", ""), meta.get("jd_id", ""), "mid"
                )
                # try from new gap dir
                gpath = (
                    root
                    / "experiments"
                    / "results"
                    / gap_exp
                    / "gap_analysis"
                    / f"{stem}.json"
                )
                if gpath.is_file():
                    import json

                    gap_rec = json.loads(gpath.read_text(encoding="utf-8"))
                    gap_index[pid] = gap_rec
                    reqs = requirements_from_gap(gap_rec)
                    req_by_pair[pid] = reqs
                    R = embed_texts(
                        embeddings,
                        [str(r.get("job_competency") or "") for r in reqs],
                    )
                    req_emb[pid] = R
                    mask = np.asarray([p != pid for p in all_other_pids], dtype=bool)
                    other = all_q_stack[mask]
                    tau_map[pid] = calibrate_tau_p(
                        req_embeddings=R, other_question_embeddings=other
                    )

    # Dimension labels skipped by default
    if with_dimension_labels:
        extraction_info["dimension_labels"] = "requested_but_deferred_to_pilot_stage"
    else:
        extraction_info["dimension_labels"] = "not_requested"

    # --- Per-pair metrics on 150 ---
    per_pair_rows: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    for pid in pair_ids_150:
        rec = b0_index[pid]
        texts = q_texts[pid]
        fit = rec.get("fit_label")
        tau = tau_map.get(pid)
        reqs = req_by_pair.get(pid)
        metrics = compute_full_metrics(
            texts,
            embeddings,
            requirements=reqs,
            tau=tau,
            precomputed_q=q_emb[pid],
            precomputed_r=req_emb.get(pid),
        )
        asg = metrics.pop("primary_assignment", None)
        sens = metrics.pop("sensitivity", None)
        row = {
            "pair_id": pid,
            "profile_id": rec.get("profile_id"),
            "jd_id": rec.get("jd_id"),
            "fit_label": fit,
            "source_experiment": B0_150_EXPERIMENT,
            "condition": "B0",
            **{k: metrics[k] for k in metrics if not k.startswith("_")},
            "sensitivity": sens,
        }
        write_json(out_dir / "per_pair" / f"{pid}.json", row)
        per_pair_rows.append(row)
        if asg and reqs:
            for qi, (linked, j, cos) in enumerate(
                zip(
                    asg["linked"],
                    asg["r_star_indices"],
                    asg["cosine_to_r_star"],
                    strict=False,
                )
            ):
                if j is None:
                    continue
                audit_candidates.append(
                    {
                        "pair_id": pid,
                        "question_text": texts[qi] if qi < len(texts) else "",
                        "requirement_text": str(
                            reqs[j].get("job_competency") or ""
                        ),
                        "rank": 1,
                        "cosine": cos,
                        "linked": bool(linked),
                    }
                )

    # Primary-30 replicates
    prim_ids = primary_pair_ids()
    replicate_metrics: dict[str, list[dict[str, float | None]]] = {}
    replicate_rows: list[dict[str, Any]] = []
    for pid in prim_ids:
        reps = b0_replicate_records(pid)
        replicate_metrics[pid] = []
        per_rep_vals: list[dict[str, Any]] = []
        for ri, rep in enumerate(reps):
            texts = question_texts(rep)
            E = embed_texts(embeddings, texts)
            m = compute_full_metrics(
                texts,
                embeddings,
                requirements=req_by_pair.get(pid),
                tau=tau_map.get(pid),
                precomputed_q=E,
                precomputed_r=req_emb.get(pid),
            )
            m.pop("primary_assignment", None)
            m.pop("sensitivity", None)
            vals = {
                "M1.P": m.get("M1.P"),
                "M2.P": m.get("M2.P"),
                "M3.P": m.get("M3.P"),
                "M4.P": m.get("M4.P"),
            }
            replicate_metrics[pid].append(vals)
            per_rep_vals.append(
                {
                    "replicate_index": ri,
                    "source_experiment": rep.get("experiment_id"),
                    **vals,
                }
            )
        # means
        means = {}
        for metric in ("M1.P", "M2.P", "M3.P", "M4.P"):
            xs = [v[metric] for v in replicate_metrics[pid] if v.get(metric) is not None]
            means[metric] = float(np.mean(xs)) if xs else None
        replicate_rows.append(
            {
                "pair_id": pid,
                "fit_label": (b0_index.get(pid) or {}).get("fit_label"),
                "replicates": per_rep_vals,
                "per_pair_mean": means,
            }
        )
    write_json(out_dir / "primary30_replicates.json", replicate_rows)

    # Validity
    print("[metrics_freeze] running validity checks V1–V5")
    v1 = v1_known_answer_checks()
    v2 = v2_null_control(
        pair_ids=sorted(req_by_pair.keys()),
        question_emb_by_pair=q_emb,
        req_emb_by_pair=req_emb,
        requirements_by_pair=req_by_pair,
        tau_by_pair=tau_map,
    )
    v3 = v3_duplicate_injection(question_emb_by_pair=q_emb)
    v4 = v4_noise_floor(replicate_metrics=replicate_metrics)

    audit_rows = sample_assignment_audit_rows(
        audit_candidates, n=AUDIT_SAMPLE_N, seed=SEED
    )
    audit_path = root / ASSIGNMENT_AUDIT_REL
    write_assignment_audit_csv(audit_path, audit_rows)

    # Distributions
    def _dist(rows: list[dict], metric: str) -> dict[str, Any]:
        overall = _summarise_metric([r.get(metric) for r in rows])
        by_fit = {}
        for label in ("Good Fit", "Potential Fit"):
            by_fit[label] = _summarise_metric(
                [r.get(metric) for r in rows if r.get("fit_label") == label]
            )
        return {"overall": overall, "by_fit_label": by_fit}

    distributions = {
        m: _dist(per_pair_rows, m)
        for m in ("M1.P", "M1.S1", "M4.P", "M2.P", "M3.P")
    }
    n_m1 = sum(1 for r in per_pair_rows if r.get("M1.P") is not None)
    n_m4 = sum(1 for r in per_pair_rows if r.get("M4.P") is not None)
    n_m2 = sum(1 for r in per_pair_rows if r.get("M2.P") is not None)
    n_m3 = sum(1 for r in per_pair_rows if r.get("M3.P") is not None)

    # Combined CSV
    combined = root / "experiments" / "results" / COMBINED_DIR_NAME
    combined.mkdir(parents=True, exist_ok=True)
    csv_path = combined / "baseline_150_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "pair_id",
            "fit_label",
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
            "tau",
            "n_requirements",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_pair_rows:
            w.writerow(r)

    # Spec + freeze
    spec_path, spec_hash = write_spec()
    tau_hash = hash_tau_map(tau_map)
    freeze_path = write_freeze(
        tau_map=tau_map,
        baseline_result_dirs=[
            B0_150_EXPERIMENT,
            *B0_PRIMARY_REPLICATES,
            exp_id,
        ],
        calibrated_thresholds_hash=tau_hash,
        extra={
            "validity_experiment_id": exp_id,
            "n_tau_calibrated": len(tau_map),
            "pilot_dirs_present_at_freeze": pilot_presence,
            "freeze_blind_to_pilot_questions": True,
            "note_pilot_may_have_been_inspected_previously": bool(pilot_presence),
        },
    )

    # Human-readable metrics_spec.md
    _write_metrics_spec_md(root, spec_hash)

    summary = {
        "experiment_id": exp_id,
        "stage": "metrics_freeze",
        "status": "completed",
        "spec_version": "v1",
        "spec_sha256": spec_hash,
        "freeze_path": str(freeze_path),
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "embedding_model": settings.embedding_model,
        "embedding_device": device,
        "parameters": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
            "usd_cny_rate": USD_CNY_RATE,
            "extract_missing": extract_missing,
            "with_dimension_labels": with_dimension_labels,
        },
        "counts": {
            "baseline_sets_M1_M4": n_m1,
            "baseline_sets_M4": n_m4,
            "baseline_sets_M2": n_m2,
            "baseline_sets_M3": n_m3,
            "gap_matrices_used": len(req_by_pair),
            "tau_calibrated": len(tau_map),
        },
        "distributions": distributions,
        "validity": {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5_audit": str(audit_path)},
        "primary30_noise_floor": v4,
        "extraction_missing": extraction_info,
        "pilot_summaries_stage_status_only": pilot_presence,
        "result_files": [
            f"per_pair/{pid}.json" for pid in pair_ids_150
        ],
    }
    write_json(out_dir / "summary.json", summary)
    write_json(combined / "baseline_summary.json", summary)

    # Figures 50–51 (baseline / validity) — no pilot data
    try:
        from app.services.eval_metrics.figures import (
            figure_50_baseline_distributions,
            figure_51_validity,
        )

        figs = root / "docs" / "figures"
        figure_50_baseline_distributions(figs / "fig50_baseline_distributions.png", per_pair_rows)
        figure_51_validity(figs / "fig51_validity_checks.png", v2, v3, v4)
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics_freeze] figure warning: {exc}")

    # Baseline report
    try:
        from app.services.eval_metrics.report import write_baseline_report

        write_baseline_report(summary, freeze_path=freeze_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics_freeze] report warning: {exc}")

    print(f"[metrics_freeze] freeze -> {freeze_path}")
    print(f"[metrics_freeze] summary -> {out_dir / 'summary.json'}")
    return summary


def _write_metrics_spec_md(root: Path, spec_hash: str) -> None:
    spec = build_spec_payload()
    lines = [
        "# Frozen evaluation metrics (v1)",
        "",
        f"**Spec SHA-256:** `{spec_hash}`",
        "",
        "These definitions are frozen before pilot question files are opened. "
        "Metrics measure alignment with the gap matrix and surface diversity, "
        "**not** question quality. The fit-label control was null (gap labels are noisy).",
        "",
        "## Primary metrics",
        "",
        "| ID | Name | Direction |",
        "|----|------|-----------|",
        "| M1.P | Mean pairwise cosine similarity | lower better |",
        "| M2.P | Gap-targeting rate | higher better |",
        "| M3.P | Capped requirement coverage | higher better |",
        "| M4.P | Vendi score (cosine kernel) | higher better |",
        "",
        "## Assignment",
        "",
        "- `r*(q) = argmax_r cos(q, r)` over requirement `job_competency` strings.",
        "- Linked iff `cos(q, r*) >= tau_p`.",
        "- `tau_p` = 95th percentile of `max_r cos(q', r)` over B0 questions from the other 149 pairs.",
        "- Sensitivity (not primary): threshold-free; fixed 0.35 / 0.45 / 0.55.",
        "",
        "## Decision rule (advisory)",
        "",
    ]
    for rule in spec["decision_rule"]["SCALE_requires"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "Otherwise: **DIAGNOSE**.",
            "",
            "## Limitations",
            "",
        ]
    )
    for lim in spec["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    (root / "docs" / "metrics_spec.md").write_text("\n".join(lines), encoding="utf-8")
