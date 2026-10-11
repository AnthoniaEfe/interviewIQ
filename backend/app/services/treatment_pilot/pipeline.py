"""End-to-end treatment pilot orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.services.embeddings.service import build_embedding_service
from app.services.fit_label_control.extract import api_key_present
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    load_extraction_records,
    load_gap_records,
    project_root,
    request_id,
    utc_experiment_id,
    write_json,
)
from app.services.question_generation.pool import POOL_PROMPT, prompt_sha256
from app.services.question_generation.profile_aware import PROFILE_AWARE_PROMPT
from app.services.treatment_pilot.ablation import run_selection_ablation
from app.services.treatment_pilot.compare import run_pilot_compare
from app.services.treatment_pilot.constants import (
    CONFIG_REL,
    DEFAULT_MAX_COST_USD,
    EXTRACTION_EXPERIMENT,
    GAP_EXPERIMENT,
    GENERATION_MODEL,
    MANIFEST_REL,
    MAX_OUTPUT_TOKENS,
    PAIRS_DIR_REL,
    REASONING_EFFORT,
    SEED,
    TEMPERATURE_SETTING,
    USD_CNY_RATE,
)
from app.services.treatment_pilot.generate import (
    estimate_stage_costs,
    generation_parameters,
    run_baseline_profile_aware,
    run_treatment_pipeline,
)
from app.services.treatment_pilot.regression import reproduce_gap_matrices, validate_b0_records
from app.services.treatment_pilot.spend import SpendAborted, TreatmentSpendGuard


def _primary_stems(root: Path) -> set[str]:
    manifest = json.loads((root / MANIFEST_REL).read_text(encoding="utf-8"))
    return {f"{m['profile_id']}__{m['jd_id']}__mid" for m in manifest}


def _update_report_section(root: Path, title: str, body: str) -> None:
    report = root / "docs" / "treatment_pilot_report.md"
    if not report.is_file():
        return
    text = report.read_text(encoding="utf-8")
    marker = f"## {title}"
    start = text.find(marker)
    import re

    if start < 0:
        text = text.rstrip() + f"\n\n{marker}\n\n{body}\n"
    else:
        rest = text[start + len(marker) :]
        m = re.search(r"\n## ", rest)
        end = start + len(marker) + (m.start() if m else len(rest))
        text = text[:start] + f"## {title}\n\n{body}\n" + text[end:]
    report.write_text(text, encoding="utf-8")


async def run_treatment_pilot(
    *,
    stage: str = "all",
    dry_run: bool = False,
    confirm_spend: bool = False,
    max_cost: float = DEFAULT_MAX_COST_USD,
    resume: bool = True,
    replicates: int = 1,
    smoke: bool = False,
    source_experiment: str | None = None,
    gap_experiment: str | None = None,
    selection: str = "competency_weighted_mmr",
    b1_experiment: str | None = None,
    treatment_experiment: str | None = None,
    ablation_experiment: str | None = None,
    skip_regression: bool = False,
) -> dict[str, Any]:
    """Run one stage or the full pilot (`stage='all'`)."""
    root = project_root()
    get_settings.cache_clear()
    settings = Settings()
    extraction_id = source_experiment or EXTRACTION_EXPERIMENT
    gap_id = gap_experiment or GAP_EXPERIMENT
    pairs_dir = root / PAIRS_DIR_REL
    primary_stems = _primary_stems(root)

    out: dict[str, Any] = {
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "generation_parameters": generation_parameters(),
        "usd_cny_rate": USD_CNY_RATE,
        "prompt_meta": {
            "profile_aware_baseline": {
                "file": PROFILE_AWARE_PROMPT.relative_path,
                "version": PROFILE_AWARE_PROMPT.version,
                "sha256": prompt_sha256(PROFILE_AWARE_PROMPT.load()),
            },
            "interviewiq_pool": {
                "file": POOL_PROMPT.relative_path,
                "version": POOL_PROMPT.version,
                "sha256": prompt_sha256(POOL_PROMPT.load()),
            },
        },
        "model": GENERATION_MODEL,
        "seed": SEED,
    }

    # --- regression (before paid calls) ---
    if stage in {"all", "regression", "treatment_pipeline", "baseline_profile_aware"} and not skip_regression:
        print("[treatment_pilot] loading embeddings for regression / generation")
        embeddings = build_embedding_service(
            settings.embedding_model,
            settings.embedding_device,
            use_hashing_fallback=settings.embedding_model == "hashing-test",
        )
        gap_reg = reproduce_gap_matrices(
            root=root,
            settings=settings,
            embeddings=embeddings,
            extraction_experiment=extraction_id,
            gap_experiment=gap_id,
        )
        b0_reg = validate_b0_records(root, primary_stems)
        out["regression"] = {"gap": gap_reg, "b0": b0_reg}
        _update_report_section(
            root,
            "Regression checks",
            (
                f"- Gap reproduction passed: `{gap_reg.get('passed')}`\n"
                f"  - counts: `{gap_reg.get('counts')}` (expected `{gap_reg.get('expected_counts')}`)\n"
                f"  - counts_match_expected: `{gap_reg.get('counts_match_expected')}`\n"
                f"  - n_label_mismatch: {gap_reg.get('n_label_mismatch')}, "
                f"n_score_mismatch: {gap_reg.get('n_score_mismatch')}, "
                f"n_weight_mismatch: {gap_reg.get('n_weight_mismatch')}, "
                f"max_abs_diff: {gap_reg.get('max_abs_diff')}\n"
                f"- B0 validity passed: `{b0_reg.get('passed')}` "
                f"(exceptions: {len(b0_reg.get('exceptions') or [])})\n"
            ),
        )
        if not gap_reg.get("passed") or not b0_reg.get("passed"):
            out["status"] = "failed_regression"
            print("[treatment_pilot] STOP: regression checks failed")
            return out
        if stage == "regression":
            out["status"] = "completed"
            return out
    else:
        embeddings = None

    ext_records = load_extraction_records(
        root / "experiments" / "results" / extraction_id
    )
    if smoke:
        ext_records = ext_records[:2]
    gap_records = load_gap_records(
        root / "experiments" / "results" / gap_id / "gap_analysis"
    )

    est = estimate_stage_costs(len(ext_records))
    out["dry_run_estimate"] = est

    if dry_run or stage == "dry_run":
        print("[treatment_pilot] DRY RUN — no API calls")
        print(json.dumps(est, indent=2))
        print(f"Pairs ({len(ext_records)}):")
        for r in ext_records:
            print(f"  - {request_id(r)}")
        print("Paid stages: baseline_profile_aware (B1), treatment_pipeline pool (T)")
        print("Local stages: selection_ablation, pilot_compare")
        _update_report_section(
            root,
            "Dry-run and spend",
            (
                f"- n_pairs: {est['n_pairs']}\n"
                f"- B1 est total: {est['b1']['est_total_usd']} USD / {est['b1']['est_total_cny']} CNY\n"
                f"- T pool est total: {est['t_pool']['est_total_usd']} USD / {est['t_pool']['est_total_cny']} CNY\n"
                f"- Combined paid est: {est['total_paid_usd']} USD / {est['total_paid_cny']} CNY\n"
                f"- Assumptions: `{json.dumps(est['assumptions'])}`\n"
                f"- API key present: `{api_key_present(settings)}`\n"
                f"- Next command when key is set:\n"
                f"  `PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py "
                f"--confirm-spend --max-cost {max_cost}`\n"
            ),
        )
        out["status"] = "dry_run"
        if not api_key_present(settings):
            cmd = (
                "PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py "
                f"--confirm-spend --max-cost {max_cost}"
            )
            print(
                "\nOPENAI_API_KEY is not set. Stopping after dry-run.\n"
                f"Once the key is available, run:\n  {cmd}\n"
            )
            out["stopped"] = "missing_api_key"
            out["command"] = cmd
        return out

    if embeddings is None:
        embeddings = build_embedding_service(
            settings.embedding_model,
            settings.embedding_device,
            use_hashing_fallback=settings.embedding_model == "hashing-test",
        )

    if not api_key_present(settings) and stage in {
        "all",
        "baseline_profile_aware",
        "treatment_pipeline",
    }:
        cmd = (
            "PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py "
            f"--confirm-spend --max-cost {max_cost}"
        )
        print(
            "OPENAI_API_KEY is not set. Stopping before paid calls.\n"
            f"Run dry-run first, then:\n  {cmd}\n"
        )
        out["status"] = "stopped_missing_api_key"
        out["command"] = cmd
        return out

    # Shared spend guard across B1 + T for `all`
    # Prior: prefer smoke-measured if present in out, else dry-run per-pair sum
    prior = float(est["b1"]["est_cost_per_pair_usd"] + est["t_pool"]["est_cost_per_pair_usd"])
    guard = TreatmentSpendGuard(
        max_cost=max_cost, confirm_spend=confirm_spend, prior_per_pair=prior
    )

    b1_dir: Path | None = Path(b1_experiment) if b1_experiment else None
    t_dir: Path | None = Path(treatment_experiment) if treatment_experiment else None
    abl_dir: Path | None = Path(ablation_experiment) if ablation_experiment else None

    if b1_experiment and not str(b1_experiment).startswith("/"):
        b1_dir = root / "experiments" / "results" / b1_experiment
    if treatment_experiment and not str(treatment_experiment).startswith("/"):
        t_dir = root / "experiments" / "results" / treatment_experiment
    if ablation_experiment and not str(ablation_experiment).startswith("/"):
        abl_dir = root / "experiments" / "results" / ablation_experiment

    try:
        if stage in {"all", "baseline_profile_aware"}:
            exp_id = utc_experiment_id()
            out_dir = root / "experiments" / "results" / exp_id
            out_dir.mkdir(parents=True, exist_ok=True)
            # For smoke: set prior from estimate; after smoke, caller re-estimates
            guard.set_prior(float(est["b1"]["est_cost_per_pair_usd"]))
            # Write provenance summary scaffold
            write_json(
                out_dir / "_provenance.json",
                {
                    "git_commit": out["git_commit"],
                    "library_versions": out["library_versions"],
                    "generation_parameters": out["generation_parameters"],
                    "prompt_meta": out["prompt_meta"],
                    "device": settings.embedding_device,
                    "embedding_model": settings.embedding_model,
                    "usd_cny_rate": USD_CNY_RATE,
                    "temperature_setting": TEMPERATURE_SETTING,
                    "reasoning_effort": REASONING_EFFORT,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "config": CONFIG_REL,
                },
            )
            b1_summary = await run_baseline_profile_aware(
                root=root,
                settings=settings,
                extraction_records=ext_records,
                experiment_id=exp_id,
                out_dir=out_dir,
                guard=guard,
                resume=resume,
                replicates=replicates,
                dry_run=False,
            )
            out["b1"] = b1_summary
            b1_dir = out_dir
            if smoke and b1_summary.get("combinations_ok"):
                # re-estimate 30-pair from measured
                measured = float(b1_summary.get("total_estimated_cost_usd") or 0)
                n_ok = int(b1_summary.get("combinations_ok") or 1)
                per = measured / n_ok
                est30 = per * 30
                out["smoke_b1_reestimate_30"] = {
                    "per_pair_usd": per,
                    "total_30_usd": est30,
                    "exceeds_max_cost": est30 > max_cost,
                }
                print(f"[smoke] B1 re-estimate 30-pair: {est30:.4f} USD")
                if est30 > max_cost:
                    out["status"] = "aborted_smoke_reestimate"
                    return out

        if stage in {"all", "treatment_pipeline"}:
            exp_id = utc_experiment_id()
            out_dir = root / "experiments" / "results" / exp_id
            out_dir.mkdir(parents=True, exist_ok=True)
            guard.set_prior(float(est["t_pool"]["est_cost_per_pair_usd"]))
            write_json(
                out_dir / "_provenance.json",
                {
                    "git_commit": out["git_commit"],
                    "library_versions": out["library_versions"],
                    "generation_parameters": out["generation_parameters"],
                    "prompt_meta": out["prompt_meta"],
                    "selection": selection,
                    "usd_cny_rate": USD_CNY_RATE,
                },
            )
            t_summary = await run_treatment_pipeline(
                root=root,
                settings=settings,
                extraction_records=ext_records,
                gap_records=gap_records,
                embeddings=embeddings,
                experiment_id=exp_id,
                out_dir=out_dir,
                guard=guard,
                selection=selection,
                resume=resume,
                replicates=replicates,
                dry_run=False,
                pairs_dir=pairs_dir,
            )
            out["treatment"] = t_summary
            t_dir = out_dir
            if smoke and t_summary.get("combinations_ok"):
                measured = float(t_summary.get("total_estimated_cost_usd") or 0)
                n_ok = int(t_summary.get("combinations_ok") or 1)
                per = measured / n_ok
                est30 = per * 30
                out["smoke_pool_reestimate_30"] = {
                    "per_pair_usd": per,
                    "total_30_usd": est30,
                    "exceeds_max_cost": est30 > max_cost,
                }
                print(f"[smoke] T pool re-estimate 30-pair: {est30:.4f} USD")
                if est30 > max_cost:
                    out["status"] = "aborted_smoke_reestimate"
                    return out

        if stage in {"all", "selection_ablation"}:
            if t_dir is None:
                raise SystemExit(
                    "selection_ablation requires a treatment experiment with pool/ "
                    "(pass --treatment-experiment)"
                )
            pool_dir = t_dir / "pool"
            exp_id = utc_experiment_id()
            out_dir = root / "experiments" / "results" / exp_id
            abl_summary = run_selection_ablation(
                pool_dir=pool_dir,
                gap_records=gap_records,
                embeddings=embeddings,
                experiment_id=exp_id,
                out_dir=out_dir,
                pairs_dir=pairs_dir,
            )
            out["selection_ablation"] = abl_summary
            abl_dir = out_dir

        if stage in {"all", "pilot_compare"}:
            compare_summary = run_pilot_compare(
                root=root,
                settings=settings,
                b1_dir=b1_dir,
                treatment_dir=t_dir,
                ablation_dir=abl_dir,
                extraction_experiment=extraction_id,
                gap_experiment=gap_id,
                embeddings=embeddings,
            )
            out["pilot_compare"] = compare_summary

    except SpendAborted as exc:
        out["status"] = "aborted_spend"
        out["abort_message"] = str(exc)
        print(f"[treatment_pilot] {exc}")
        return out

    out["status"] = out.get("status") or "completed"
    out["b1_dir"] = str(b1_dir) if b1_dir else None
    out["treatment_dir"] = str(t_dir) if t_dir else None
    out["ablation_dir"] = str(abl_dir) if abl_dir else None
    out["running_cost_usd"] = guard.running_cost
    return out
