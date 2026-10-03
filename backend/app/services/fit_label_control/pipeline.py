"""End-to-end fit-label control pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings, get_settings
from app.services.datasets.resume_fit import export_pair_files, load_split
from app.services.embeddings.service import build_embedding_service
from app.services.extraction.profiles import CV_PROMPT, JD_PROMPT
from app.services.fit_label_control.constants import (
    DEFAULT_MAX_COST_USD,
    EXTRACTION_MODEL,
    FITPOS_ASSETS,
    HOLDOUT_SEED,
    LABEL_ORDER,
    NF_SUBSET_DIRS,
    N_NO_FIT,
    PRIOR_PER_PAIR_USD,
    PROMPT_VERSIONS,
    SEED,
    SMOKE_PAIRS,
    SOURCE_SPLIT,
)
from app.services.fit_label_control.extract import (
    api_key_present,
    dry_run_payload,
    index_extraction_cache,
    load_combos_from_manifest,
    run_extraction_guarded,
)
from app.services.fit_label_control.figures import (
    figure_22_gap_rate_by_label,
    figure_23_roc,
    figure_24_stacked_shares,
    figure_25_confound,
)
from app.services.fit_label_control.report import write_generated_report
from app.services.fit_label_control.sampling import persist_nf_subsets
from app.services.fit_label_control.score import (
    regression_fitpos,
    score_records,
    write_production_gap,
)
from app.services.fit_label_control.spend import SpendAborted, SpendGuard
from app.services.fit_label_control.tables import apply_decision_rule, bundle_condition
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    load_extraction_records,
    load_gap_records,
    project_root,
    resolve_cv_text,
    utc_experiment_id,
    write_json,
)
from app.services.gap_sensitivity.metrics import wilcoxon_paired


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _jd_hash(text: str) -> str:
    return hashlib_sha(text)


def hashlib_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fitpos_records(root: Path, split_key: str) -> tuple[list[dict], Path]:
    meta = FITPOS_ASSETS[split_key]
    source = root / "experiments/results" / meta["extraction"]
    records = load_extraction_records(source)
    return records, root / meta["pairs_dir"]


def _texts_for(records: list[dict], pairs_dir: Path, train_lookup: dict[str, dict] | None = None) -> tuple[list[str | None], list[str | None]]:
    cvs, jds = [], []
    for rec in records:
        cv = resolve_cv_text(rec, pairs_dir)
        jd = None
        pid = rec.get("profile_id")
        candidates = [
            pairs_dir / "jobs" / f"{pid}.txt",
            pairs_dir / f"{pid}.txt",
        ]
        for path in candidates:
            if path.is_file() and "jobs" in str(path) or path.name.endswith(".txt") and path.parent.name == "jobs":
                jd = path.read_text(encoding="utf-8")
                break
        job_path = pairs_dir / "jobs" / f"{pid}.txt"
        if job_path.is_file():
            jd = job_path.read_text(encoding="utf-8")
        cvs.append(cv)
        jds.append(jd)
    return cvs, jds


def _matched_job_candidates(
    train,
    analysis_rows_by_split: dict[str, list[dict]],
    texts_by_pair: dict[str, str],
    used_ids: set[str],
    *,
    seed: int = SEED,
) -> tuple[list[dict], str]:
    """Jobs that appear with both a fit-positive and a No Fit resume.

    Extra extraction: at most one unused No Fit row per overlapping job, seed 42.
    """
    from collections import defaultdict

    jobs_pos: dict[str, list[str]] = defaultdict(list)
    jobs_nf: dict[str, list[str]] = defaultdict(list)
    for split, rows in analysis_rows_by_split.items():
        del split
        for row in rows:
            pid = row.get("pair_id")
            lab = row.get("fit_label")
            text = texts_by_pair.get(str(pid))
            if not text or not lab:
                continue
            h = _jd_hash(text)
            if lab == "No Fit":
                jobs_nf[h].append(str(pid))
            else:
                jobs_pos[h].append(str(pid))

    # Map hash -> job text via train
    hash_to_nf_rows = defaultdict(list)
    for _, r in train.iterrows():
        h = _jd_hash(str(r["job_description_text"]))
        if r["label"] == "No Fit":
            hash_to_nf_rows[h].append(r)

    extra_rows = []
    n_jobs_overlap_in_sample = 0
    n_jobs_need_extra = 0
    for h, pos_ids in jobs_pos.items():
        if h not in hash_to_nf_rows:
            continue
        if jobs_nf.get(h):
            n_jobs_overlap_in_sample += 1
            continue
        n_jobs_need_extra += 1
        candidates = [r for r in hash_to_nf_rows[h] if str(r["pair_id"]) not in used_ids]
        if not candidates:
            continue
        import pandas as pd

        cdf = pd.DataFrame(candidates)
        take = cdf.sample(n=1, random_state=seed).iloc[0]
        extra_rows.append(take)

    note = (
        f"Train jobs with both No Fit and Good/Potential exist "
        f"(see code inspection). In the analysis samples, "
        f"{n_jobs_overlap_in_sample} job hashes already have both sides; "
        f"{n_jobs_need_extra} fit-positive jobs lack a sampled No Fit resume "
        f"({len(extra_rows)} extra No Fit pairs proposed, seed {seed})."
    )
    return extra_rows, note


def _attach_jd_hash(rows: list[dict], jd_texts: list[str | None]) -> None:
    for row, jd in zip(rows, jd_texts, strict=True):
        row["jd_text_hash"] = _jd_hash(jd) if jd else None


def _roc_curves_payload(primary: dict, holdout: dict) -> dict[str, Any]:
    out = {}
    for split_name, block in (("primary", primary), ("holdout", holdout)):
        for cond in ("A", "D"):
            disc = (block.get(cond) or {}).get("discrimination") or {}
            gap = disc.get("gap_rate") or {}
            for key, suffix in (
                ("no_fit_vs_good", "nf_good"),
                ("no_fit_vs_fit_positive", "nf_pos"),
            ):
                item = gap.get(key) or {}
                roc = item.get("roc") or {}
                out[f"{split_name}_{cond}_{suffix}"] = {
                    "fpr": roc.get("fpr") or [],
                    "tpr": roc.get("tpr") or [],
                    "auc": item.get("auc"),
                    "ci_low": item.get("ci_low"),
                    "ci_high": item.get("ci_high"),
                }
    return out


def _thesis_paragraph(decision: dict, primary_a: dict, holdout_a: dict) -> str:
    outcome = decision.get("outcome")
    h1 = decision.get("h1_monotonic_allowed_both_splits")
    p_auc = (decision.get("primary") or {}).get("auc")
    h_auc = (decision.get("holdout") or {}).get("auc")
    mm = ((primary_a.get("confound") or {}).get("kruskal") or {}).get("mismatch_gap_rate") or {}
    mm_p = (mm.get("kruskal_wallis") or {}).get("p_value")
    parts = [
        f"Pre-registered decision-rule outcome: {outcome}.",
        f"No Fit vs Good Fit AUC (gap rate, Condition A) is {_fmt(p_auc)} on primary "
        f"and {_fmt(h_auc)} on holdout.",
    ]
    if h1:
        parts.append(
            "H1 monotonicity is allowed by the pre-registered JT-plus-adjacent-direction rule on both splits."
        )
    else:
        parts.append(
            "H1 monotonicity is not claimed: the JT-plus-adjacent-direction rule is not met on both splits."
        )
    if mm_p is not None and mm_p < 0.05:
        parts.append(
            "The random-candidate control differs by label, so job-side confounds remain possible."
        )
    else:
        parts.append(
            "The random-candidate control did not show a significant label difference on the pooled "
            "mismatch rates reported in the confound section (see numbers there)."
        )
    parts.append("No claim is made beyond what those tests support.")
    return " ".join(parts)


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _strip_results(bundle: dict) -> dict:
    """Drop PairScoreResult objects before JSON serialisation."""
    return {k: v for k, v in bundle.items() if k != "raw_score"}


async def run_fit_label_control(
    *,
    dry_run: bool = False,
    confirm_spend: bool = False,
    max_cost: float = DEFAULT_MAX_COST_USD,
    smoke: bool = False,
    skip_extraction: bool = False,
    skip_figures: bool = False,
    settings: Settings | None = None,
    embeddings=None,
    router=None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or project_root()
    get_settings.cache_clear()
    settings = settings or Settings()
    deviations: list[str] = [
        "Holdout-30 fit-positive pair ids could not be reconstructed from "
        "`sample_experiment_pairs` with seed 2026 on the 150-pool; NF-holdout uses "
        "seed 2026 on remaining No Fit train rows (the documented holdout seed), "
        "not a reverse-engineered undocumented holdout filter.",
        "Priority-weighted gap share uses requirement priority base weights "
        "(required=1.0, preferred=0.6, unspecified=0.8), not stored match-dependent "
        "`ProfileGap.weight` (that weight includes severity and evidence and would "
        "mix the outcome into the weight).",
        "Mismatched control uses the existing `mismatched_partner_indices` per "
        "fit-label group so partners stay in the same human label.",
    ]

    control = persist_nf_subsets(root=root, n=2 if smoke else N_NO_FIT)
    nf_primary_dir = root / NF_SUBSET_DIRS["primary"]
    nf_holdout_dir = root / NF_SUBSET_DIRS["holdout"]
    combos_p = load_combos_from_manifest(nf_primary_dir / "manifest.json", nf_primary_dir)
    combos_h = load_combos_from_manifest(nf_holdout_dir / "manifest.json", nf_holdout_dir)
    combos = combos_p + combos_h
    cache = index_extraction_cache(root / "experiments/results")

    train = load_split(
        SOURCE_SPLIT, cache_dir=root / "experiments/datasets/resume_job_fit/cache"
    )
    used_ids = set(control["primary"]["pair_ids"]) | set(control["holdout"]["pair_ids"])
    for rel in (
        FITPOS_ASSETS["primary"]["manifest"],
        FITPOS_ASSETS["holdout"]["manifest"],
        "experiments/datasets/resume_job_fit/experiment_pairs_150/manifest.json",
    ):
        for item in _load_json(root / rel):
            used_ids.add(item["pair_id"])

    # Matched-job extras estimated from train job overlap with fit-positive samples.
    pos_job_hashes = set()
    for split_key in ("primary", "holdout"):
        recs, pdir = _fitpos_records(root, split_key)
        _, jds = _texts_for(recs, pdir)
        for jd in jds:
            if jd:
                pos_job_hashes.add(_jd_hash(jd))
    nf_hashes = set()
    for combo in combos:
        jd_path: Path = combo["jd_path"]
        if jd_path.is_file():
            nf_hashes.add(_jd_hash(jd_path.read_text(encoding="utf-8")))
    need_extra_hashes = pos_job_hashes - nf_hashes
    extra_df_rows = []
    if need_extra_hashes:
        import pandas as pd

        pool = []
        for _, r in train.iterrows():
            if r["label"] != "No Fit":
                continue
            if str(r["pair_id"]) in used_ids:
                continue
            if _jd_hash(str(r["job_description_text"])) in need_extra_hashes:
                pool.append(r)
        if pool:
            pdf = pd.DataFrame(pool)
            # one per job hash
            extra_df_rows = [
                g.sample(n=1, random_state=SEED).iloc[0]
                for _, g in pdf.groupby(pdf["job_description_text"].map(_jd_hash))
            ]

    extra_pair_ids = {str(r["pair_id"]) for r in extra_df_rows}
    extra_pairs_n = len(extra_df_rows)
    dry = dry_run_payload(
        combos, cache, prior_per_pair=PRIOR_PER_PAIR_USD, extra_uncached=0
    )
    extra_uncached = 0
    extra_ids = [str(r["pair_id"]) for r in extra_df_rows]
    extra_uncached = sum(1 for pid in extra_ids if pid not in cache)
    extra_est = round(PRIOR_PER_PAIR_USD * extra_uncached, 8)
    include_matched_extra = (
        extra_pairs_n > 0
        and (dry["estimated_cost_usd"] + extra_est) <= max_cost + 1e-12
    )
    matched_note = (
        f"Matched-job extra pairs proposed={extra_pairs_n}, uncached={extra_uncached}, "
        f"estimated extra USD={extra_est}. "
    )
    if include_matched_extra:
        matched_note += "Included under --max-cost."
        extra_dir = root / "experiments/datasets/resume_job_fit/experiment_pairs_nf_matched"
        import pandas as pd

        extra_df = pd.DataFrame(extra_df_rows).reset_index(drop=True)
        export_pair_files(extra_df, extra_dir)
        extra_combos = load_combos_from_manifest(extra_dir / "manifest.json", extra_dir)
        combos = combos + extra_combos
        dry = dry_run_payload(combos, cache, prior_per_pair=PRIOR_PER_PAIR_USD)
    else:
        matched_note += (
            "Not extracted: "
            + (
                "would exceed --max-cost."
                if extra_pairs_n
                else "no extra pairs required or none available."
            )
        )
        if extra_pairs_n:
            deviations.append(
                "Matched-job extra extraction skipped because the estimated extra "
                f"cost ({extra_est} USD) plus core estimate "
                f"({dry['estimated_cost_usd']} USD) exceeds --max-cost {max_cost}."
            )

    print("[fit_label_control] dry-run")
    print(json.dumps({**dry, "matched_job": matched_note}, indent=2))
    print("Pairs to extract (uncached):")
    for pid in dry["uncached_pair_ids"]:
        print(f"  {pid}")

    dry_payload = {
        "dry_run": True,
        "pairs": dry,
        "matched_job": matched_note,
        "nf_primary_ids": control["primary"]["pair_ids"],
        "nf_holdout_ids": control["holdout"]["pair_ids"],
        "prompt_versions": {
            "candidate_profile": CV_PROMPT.version,
            "job_description": JD_PROMPT.version,
        },
        "extraction_model": settings.llm_fast_model,
        "git_commit": git_commit_hash(),
        "command_if_key_missing": (
            "PYTHONPATH=backend python experiments/scripts/run_fit_label_control.py "
            f"--confirm-spend --max-cost {max_cost}"
        ),
    }
    if dry_run:
        return dry_payload

    uncached_n = dry["n_uncached"]
    if uncached_n and not skip_extraction and not api_key_present(settings):
        print(
            "OPENAI_API_KEY is missing. Dry-run is complete; no API calls were made.\n"
            "After setting the key (do not paste it into chat), run:\n"
            f"  {dry_payload['command_if_key_missing']}"
        )
        return {**dry_payload, "stopped": "missing_api_key"}

    experiment_id = utc_experiment_id()
    out_dir = root / "experiments/results" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(max_cost=max_cost, confirm_spend=confirm_spend or skip_extraction)

    extraction_summary: dict[str, Any] = {
        "n_reused_from_disk": 0,
        "n_paid_calls": 0,
        "total_estimated_cost_usd": 0.0,
        "errors": [],
        "combinations_ok": 0,
    }
    if skip_extraction:
        deviations.append("Extraction skipped (skip_extraction=True); using disk cache only.")
    else:
        try:
            smoke_n = SMOKE_PAIRS if not smoke else min(SMOKE_PAIRS, len(combos))
            if uncached_n:
                print(f"[fit_label_control] smoke extraction n={smoke_n}")
            extraction_summary = await run_extraction_guarded(
                combos=combos,
                settings=settings,
                guard=guard,
                out_dir=out_dir,
                experiment_id=experiment_id,
                cache=cache,
                smoke_n=smoke_n,
                router=router,
            )
        except SpendAborted as exc:
            write_json(
                out_dir / "summary.json",
                {
                    "experiment_id": experiment_id,
                    "status": "failed_spend",
                    "stage": "fit_label_control",
                    "error": str(exc),
                    "git_commit": git_commit_hash(),
                },
            )
            raise

    nf_extract_dir = out_dir
    if not (out_dir / "extraction").is_dir() or not list((out_dir / "extraction").glob("*.json")):
        # Reuse whatever cache exists for NF pair ids
        nf_extract_dir = out_dir

    if embeddings is None:
        print(
            f"[fit_label_control] loading embeddings {settings.embedding_model} "
            f"device={settings.embedding_device}"
        )
        embeddings = build_embedding_service(
            settings.embedding_model, settings.embedding_device
        )

    nf_records = []
    if (out_dir / "extraction").is_dir() and list((out_dir / "extraction").glob("*.json")):
        nf_records = load_extraction_records(out_dir)
    else:
        # Pull from global cache by pair id
        cache = index_extraction_cache(root / "experiments/results")
        for combo in combos_p + combos_h:
            pid = combo.get("pair_id")
            if pid and pid in cache:
                nf_records.append(_load_json(cache[pid]))

    gap_dir = out_dir / "gap_analysis"
    gap_dir.mkdir(parents=True, exist_ok=True)
    for rec in nf_records:
        write_production_gap(
            rec,
            settings=settings,
            embeddings=embeddings,
            experiment_id=experiment_id,
            source_experiment=experiment_id,
            out_dir=gap_dir,
        )

    split_bundles: dict[str, dict] = {}
    regression = {}
    all_rows_a: list[dict] = []
    all_rows_d: list[dict] = []
    all_rows_s: list[dict] = []

    nf_by_split = {
        "primary": [
            r
            for r in nf_records
            if r.get("pair_id") in set(control["primary"]["pair_ids"])
        ],
        "holdout": [
            r
            for r in nf_records
            if r.get("pair_id") in set(control["holdout"]["pair_ids"])
        ],
    }
    nf_pairs_dir = {
        "primary": nf_primary_dir,
        "holdout": nf_holdout_dir,
    }

    for split_key in ("primary", "holdout"):
        fit_recs, fit_dir = _fitpos_records(root, split_key)
        nf_recs = nf_by_split[split_key]
        records = fit_recs + nf_recs
        fit_cv, fit_jd = _texts_for(fit_recs, fit_dir)
        nf_cv, nf_jd = _texts_for(nf_recs, nf_pairs_dir[split_key])
        cvs = fit_cv + nf_cv
        jds = fit_jd + nf_jd
        stored = load_gap_records(
            root / "experiments/results" / FITPOS_ASSETS[split_key]["stored_gap"] / "gap_analysis"
        )
        reg = regression_fitpos(
            fit_recs, stored, settings, embeddings, FITPOS_ASSETS[split_key]["split"]
        )
        regression[split_key] = reg
        print(
            f"[fit_label_control] regression {split_key} passed={reg.get('passed')} "
            f"counts_match={reg.get('counts_match_expected')} counts={reg.get('counts')}"
        )
        scored = score_records(records, cvs, jds, settings, embeddings)
        _attach_jd_hash(scored["A"]["rows"], jds)
        _attach_jd_hash(scored["D"]["rows"], jds)
        _attach_jd_hash(scored["semantic"]["rows"], jds)
        bundle = {
            "A": bundle_condition(scored["A"]["rows"]),
            "D": bundle_condition(scored["D"]["rows"]),
            "semantic": bundle_condition(scored["semantic"]["rows"]),
            "rows_A": scored["A"]["rows"],
            "rows_D": scored["D"]["rows"],
            "rows_S": scored["semantic"]["rows"],
        }
        split_bundles[split_key] = bundle
        all_rows_a.extend(scored["A"]["rows"])
        all_rows_d.extend(scored["D"]["rows"])
        all_rows_s.extend(scored["semantic"]["rows"])
        for rec, row in zip(records, scored["A"]["rows"], strict=True):
            write_json(
                out_dir / "pairs" / f"{rec.get('pair_id')}.json",
                {"record_meta": {"pair_id": rec.get("pair_id"), "fit_label": rec.get("fit_label")}, **row},
            )

    extra_recs = [r for r in nf_records if r.get("pair_id") in extra_pair_ids]
    extra_rows_a: list[dict] = []
    extra_dir = root / "experiments/datasets/resume_job_fit/experiment_pairs_nf_matched"
    if extra_recs and extra_dir.is_dir():
        extra_cv, extra_jd = _texts_for(extra_recs, extra_dir)
        extra_scored = score_records(extra_recs, extra_cv, extra_jd, settings, embeddings)
        _attach_jd_hash(extra_scored["A"]["rows"], extra_jd)
        extra_rows_a = extra_scored["A"]["rows"]

    pooled = {
        "A": bundle_condition(all_rows_a),
        "D": bundle_condition(all_rows_d),
        "semantic": bundle_condition(all_rows_s),
    }

    decision = apply_decision_rule(
        split_bundles["primary"]["A"]["gradient_gap_rate"],
        split_bundles["holdout"]["A"]["gradient_gap_rate"],
        split_bundles["primary"]["A"]["discrimination"]["gap_rate"],
        split_bundles["holdout"]["A"]["discrimination"]["gap_rate"],
    )

    # Matched-job Wilcoxon on pooled Condition A rows that share a JD hash across labels
    matched_text = matched_note
    by_hash: dict[str, dict[str, list[float]]] = {}
    matched_pool = all_rows_a + extra_rows_a
    for row in matched_pool:
        h = row.get("jd_text_hash")
        lab = row.get("fit_label")
        if not h or not lab or row.get("gap_rate") is None:
            continue
        by_hash.setdefault(h, {k: [] for k in LABEL_ORDER})
        by_hash[h][str(lab)].append(float(row["gap_rate"]))
    pos_rates, nf_rates = [], []
    for h, labs in by_hash.items():
        pos = labs["Good Fit"] + labs["Potential Fit"]
        nf = labs["No Fit"]
        if pos and nf:
            pos_rates.append(float(np.mean(pos)))
            nf_rates.append(float(np.mean(nf)))
    if len(pos_rates) >= 2:
        w = wilcoxon_paired(np.asarray(nf_rates), np.asarray(pos_rates))
        matched_text += (
            f" Paired Wilcoxon on {len(pos_rates)} jobs with both sides in the scored set: "
            f"p={w.get('p_value')}, rank_biserial={w.get('rank_biserial')} "
            f"(No Fit gap vs fit-positive gap, within job)."
        )
    else:
        matched_text += (
            f" Paired Wilcoxon skipped: only {len(pos_rates)} jobs had both sides after scoring."
        )

    combined_dir = root / "experiments/results/fit_label_control_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    desc_rows = []
    for split_name, bundle in (
        ("primary", split_bundles["primary"]),
        ("holdout", split_bundles["holdout"]),
        ("pooled", pooled),
    ):
        for cond in ("A", "D", "semantic"):
            for d in bundle[cond]["descriptives"]:
                desc_rows.append({"split": split_name, "condition": cond, **d})
    _write_csv(combined_dir / "descriptives.csv", desc_rows)
    _write_csv(combined_dir / "pair_rows_A.csv", all_rows_a)
    _write_csv(combined_dir / "pair_rows_D.csv", all_rows_d)
    _write_csv(combined_dir / "pair_rows_semantic.csv", all_rows_s)

    figures_dir = root / "docs/figures"
    if not skip_figures:
        figure_22_gap_rate_by_label(
            figures_dir / "fig22_gap_rate_by_fit_label.png",
            split_bundles["primary"]["rows_A"],
            split_bundles["holdout"]["rows_A"],
            all_rows_a,
        )
        figure_23_roc(
            figures_dir / "fig23_roc_fit_labels.png",
            _roc_curves_payload(split_bundles["primary"], split_bundles["holdout"]),
        )
        figure_24_stacked_shares(
            figures_dir / "fig24_stacked_shares_by_fit_label.png",
            split_bundles["primary"]["A"]["descriptives"],
            split_bundles["holdout"]["A"]["descriptives"],
            pooled["A"]["descriptives"],
        )
        figure_25_confound(figures_dir / "fig25_confound_panel.png", all_rows_a)

    lib = library_versions()
    try:
        import statsmodels

        lib["statsmodels"] = statsmodels.__version__
    except Exception:
        lib["statsmodels"] = None

    payload = {
        "experiment_id": experiment_id,
        "git_commit": git_commit_hash(),
        "library_versions": lib,
        "extraction_model": settings.llm_fast_model,
        "prompt_versions": {
            "candidate_profile": CV_PROMPT.version,
            "job_description": JD_PROMPT.version,
        },
        "embedding_model": settings.embedding_model,
        "embedding_device": getattr(embeddings, "device", settings.embedding_device),
        "regression": regression,
        "actual_cost_usd": extraction_summary.get("total_estimated_cost_usd"),
        "n_reused": extraction_summary.get("n_reused_from_disk"),
        "n_paid": extraction_summary.get("n_paid_calls"),
        "n_failed": extraction_summary.get("combinations_failed"),
        "failures": extraction_summary.get("errors"),
        "primary": split_bundles["primary"],
        "holdout": split_bundles["holdout"],
        "pooled": pooled,
        "decision": decision,
        "matched_job_text": matched_text,
        "deviations": deviations,
        "files": [
            str(out_dir),
            str(combined_dir),
            "docs/figures/fig22_gap_rate_by_fit_label.png",
            "docs/figures/fig23_roc_fit_labels.png",
            "docs/figures/fig24_stacked_shares_by_fit_label.png",
            "docs/figures/fig25_confound_panel.png",
            "docs/fit_label_control_report.md",
            NF_SUBSET_DIRS["primary"],
            NF_SUBSET_DIRS["holdout"],
        ],
    }
    payload["thesis_paragraph"] = _thesis_paragraph(
        decision, split_bundles["primary"]["A"], split_bundles["holdout"]["A"]
    )

    # JSON-safe summary (drop bulky roc points duplication is ok)
    summary = {
        "experiment_id": experiment_id,
        "status": extraction_summary.get("status", "completed"),
        "stage": "fit_label_control",
        "prompt_versions": payload["prompt_versions"],
        "model": settings.llm_fast_model,
        "git_commit": payload["git_commit"],
        "library_versions": lib,
        "embedding_model": payload["embedding_model"],
        "embedding_device": payload["embedding_device"],
        "seed": SEED,
        "holdout_seed": HOLDOUT_SEED,
        "regression": regression,
        "decision": decision,
        "actual_cost_usd": payload["actual_cost_usd"],
        "extraction": {
            k: extraction_summary.get(k)
            for k in (
                "n_reused_from_disk",
                "n_paid_calls",
                "combinations_ok",
                "combinations_failed",
                "errors",
                "total_estimated_cost_usd",
            )
        },
        "primary": {
            "A": split_bundles["primary"]["A"],
            "D": split_bundles["primary"]["D"],
            "semantic": split_bundles["primary"]["semantic"],
        },
        "holdout": {
            "A": split_bundles["holdout"]["A"],
            "D": split_bundles["holdout"]["D"],
            "semantic": split_bundles["holdout"]["semantic"],
        },
        "pooled": pooled,
        "matched_job_text": matched_text,
        "deviations": deviations,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(combined_dir / "summary.json", summary)
    write_generated_report(root / "docs/fit_label_control_report.md", payload)
    print(f"[fit_label_control] summary -> {out_dir / 'summary.json'}")
    return {
        "experiment_id": experiment_id,
        "out_dir": out_dir,
        "decision": decision,
        "regression": regression,
        "actual_cost_usd": payload["actual_cost_usd"],
        "extraction": extraction_summary,
        "primary": split_bundles["primary"],
        "holdout": split_bundles["holdout"],
        "pooled": pooled,
        "matched_job_text": matched_text,
        "deviations": deviations,
        "files": payload["files"],
    }
