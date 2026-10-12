"""Part B: compute frozen metrics on pilot outputs (requires verified freeze)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.services.embeddings.service import build_embedding_service
from app.services.eval_metrics.compute import compute_full_metrics, embed_texts
from app.services.eval_metrics.constants import (
    B0_PRIMARY_REPLICATES,
    BOOTSTRAP_RESAMPLES,
    COMBINED_DIR_NAME,
    GAP_PRIMARY,
    SEED,
)
from app.services.eval_metrics.freeze import FreezeError, verify_freeze
from app.services.eval_metrics.loaders import (
    b0_replicate_records,
    index_gap_by_pair,
    load_condition_records,
    load_s0_repeat_records,
    pair_stem,
    primary_pair_ids,
    question_texts,
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
from app.services.gap_sensitivity.metrics import bootstrap_mean_ci


def run_metrics_pilot_stage(
    *,
    settings: Settings,
    allow_post_hoc: bool = False,
) -> dict[str, Any]:
    try:
        freeze = verify_freeze(allow_post_hoc=allow_post_hoc)
    except FreezeError as exc:
        raise SystemExit(str(exc)) from exc

    post_hoc = bool(freeze.get("post_hoc"))
    tau_map: dict[str, float] = dict(freeze.get("tau_p") or {})
    root = project_root()
    dirs = resolve_pilot_dirs()
    if not dirs.get("B1") or not dirs.get("T") or not dirs.get("ablation"):
        cmds = [
            "PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --dry-run",
            "PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --confirm-spend --max-cost 4.00",
        ]
        raise SystemExit(
            "Pilot outputs missing (need completed baseline_profile_aware, "
            "treatment_pipeline, selection_ablation). Run the pilot task first:\n  "
            + "\n  ".join(cmds)
        )

    embeddings = build_embedding_service(
        settings.embedding_model, settings.embedding_device
    )
    gap_index = index_gap_by_pair()
    # Prefer primary gap for primary-30
    prim_ids = primary_pair_ids()

    exp_id = utc_experiment_id()
    out_dir = root / "experiments" / "results" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_pair").mkdir(parents=True, exist_ok=True)

    b1_recs = load_condition_records(dirs["B1"], "baseline_profile_aware")
    t_recs = load_condition_records(dirs["T"], "treatment")
    pool_recs = load_condition_records(dirs["T"], "pool")
    b2_recs = load_condition_records(dirs["ablation"], "B2")
    s1_recs = load_condition_records(dirs["ablation"], "S1")
    s2_recs = load_condition_records(dirs["ablation"], "S2")

    req_cache: dict[str, list[dict]] = {}
    req_emb_cache: dict[str, np.ndarray] = {}
    for pid in prim_ids:
        gap = gap_index.get(pid)
        if not gap:
            continue
        reqs = requirements_from_gap(gap)
        req_cache[pid] = reqs
        req_emb_cache[pid] = embed_texts(
            embeddings, [str(r.get("job_competency") or "") for r in reqs]
        )

    conditions_data: dict[str, dict[str, dict[str, Any]]] = {
        c: {} for c in ("B0", "B1", "T", "B2", "S0", "S1", "S2")
    }

    def _score_texts(pid: str, texts: list[str]) -> dict[str, Any]:
        return compute_full_metrics(
            texts,
            embeddings,
            requirements=req_cache.get(pid),
            tau=tau_map.get(pid),
            precomputed_r=req_emb_cache.get(pid),
        )

    for pid in prim_ids:
        meta_gap = gap_index.get(pid) or {}
        stem = pair_stem(
            meta_gap.get("profile_id")
            or (b1_recs.get(pid) or {}).get("profile_id")
            or "",
            meta_gap.get("jd_id") or (b1_recs.get(pid) or {}).get("jd_id") or "",
            "mid",
        )
        fit = meta_gap.get("fit_label") or (b1_recs.get(pid) or {}).get("fit_label")

        # B0 mean of replicates
        reps = b0_replicate_records(pid)
        rep_metrics = []
        for rep in reps:
            m = _score_texts(pid, question_texts(rep))
            m.pop("primary_assignment", None)
            m.pop("sensitivity", None)
            rep_metrics.append(m)
        b0_mean = _mean_metrics(rep_metrics)
        conditions_data["B0"][pid] = {
            "pair_id": pid,
            "fit_label": fit,
            "condition": "B0",
            **b0_mean,
            "n_replicates": len(rep_metrics),
        }

        for cond, store in (
            ("B1", b1_recs),
            ("T", t_recs),
            ("B2", b2_recs),
            ("S1", s1_recs),
            ("S2", s2_recs),
        ):
            rec = store.get(pid)
            if not rec:
                # try stem key
                rec = store.get(stem)
            if not rec:
                continue
            m = _score_texts(pid, question_texts(rec))
            m.pop("primary_assignment", None)
            sens = m.pop("sensitivity", None)
            conditions_data[cond][pid] = {
                "pair_id": pid,
                "fit_label": fit,
                "condition": cond,
                **m,
                "sensitivity": sens,
            }

        # S0: mean metrics across repeats
        s0_reps = load_s0_repeat_records(dirs["ablation"], stem)
        if not s0_reps:
            # fallback: any file containing pair_id
            s0_dir = dirs["ablation"] / "S0"
            if s0_dir.is_dir():
                import json

                for p in sorted(s0_dir.glob("*.json")):
                    rec = json.loads(p.read_text(encoding="utf-8"))
                    if str(rec.get("pair_id")) == pid:
                        s0_reps.append(rec)
        s0_ms = []
        for rec in s0_reps:
            m = _score_texts(pid, question_texts(rec))
            m.pop("primary_assignment", None)
            m.pop("sensitivity", None)
            s0_ms.append(m)
        if s0_ms:
            conditions_data["S0"][pid] = {
                "pair_id": pid,
                "fit_label": fit,
                "condition": "S0",
                **_mean_metrics(s0_ms),
                "n_repeats": len(s0_ms),
            }

        # Persist per-pair bundle
        bundle = {
            "pair_id": pid,
            "fit_label": fit,
            "conditions": {c: conditions_data[c].get(pid) for c in conditions_data},
            "pool_n_questions": len(question_texts(pool_recs.get(pid) or pool_recs.get(stem) or {})),
        }
        write_json(out_dir / "per_pair" / f"{pid}.json", bundle)

    # Summaries per condition
    condition_summaries: dict[str, dict[str, Any]] = {}
    for cond, by_pid in conditions_data.items():
        rows = list(by_pid.values())
        condition_summaries[cond] = {}
        for metric in ("M1.P", "M2.P", "M3.P", "M4.P", "M1.S1"):
            vals = [r.get(metric) for r in rows]
            condition_summaries[cond][metric] = bootstrap_mean_ci(
                np.asarray([v for v in vals if v is not None], dtype=np.float64),
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            )
        condition_summaries[cond]["n_pairs"] = len(rows)

    # CSV
    combined = root / "experiments" / "results" / COMBINED_DIR_NAME
    combined.mkdir(parents=True, exist_ok=True)
    csv_path = combined / "pilot_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["pair_id", "fit_label", "condition", "M1.P", "M2.P", "M3.P", "M4.P", "M1.S1", "M3.S2"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for cond, by_pid in conditions_data.items():
            for row in by_pid.values():
                w.writerow(row)

    summary = {
        "experiment_id": exp_id,
        "stage": "metrics_pilot",
        "status": "completed",
        "post_hoc": post_hoc,
        "post_hoc_tag": freeze.get("post_hoc_tag") if post_hoc else None,
        "spec_sha256": freeze.get("spec_sha256"),
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "embedding_model": settings.embedding_model,
        "embedding_device": getattr(embeddings, "device", settings.embedding_device),
        "pilot_dirs": {k: (str(v) if v else None) for k, v in dirs.items()},
        "gap_source_primary": GAP_PRIMARY,
        "b0_replicates": list(B0_PRIMARY_REPLICATES),
        "condition_summaries": condition_summaries,
        "n_primary_pairs": len(prim_ids),
        "freeze_note": (
            "Pilot outputs existed before this freeze; decision is advisory "
            "(not blind to prior pilot inspection)."
            if freeze.get("note_pilot_may_have_been_inspected_previously")
            else "Freeze verified before opening pilot question files in this stage."
        ),
    }
    if post_hoc:
        summary["tag"] = "POST-HOC, not primary"

    write_json(out_dir / "summary.json", summary)
    write_json(combined / "pilot_summary.json", summary)
    write_json(out_dir / "conditions_data.json", {c: list(v.values()) for c, v in conditions_data.items()})

    try:
        from app.services.eval_metrics.figures import figure_52_pilot_by_condition

        figure_52_pilot_by_condition(
            root / "docs" / "figures" / "fig52_pilot_primary_metrics.png",
            condition_summaries,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics_pilot] figure warning: {exc}")

    print(f"[metrics_pilot] summary -> {out_dir / 'summary.json'}")
    return summary


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    for k in keys:
        if k in {"sensitivity", "primary_assignment", "linked_category_shares"}:
            continue
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = float(np.mean(vals)) if vals else None
    return out
