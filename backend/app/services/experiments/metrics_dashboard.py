"""Aggregate frozen metrics + scale decision for the research dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.services.experiments.reader import project_root, results_root


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_stage(stage: str) -> tuple[str | None, dict[str, Any] | None]:
    """Prefer non-POST-HOC completed runs; fall back to newest."""
    root = results_root()
    if not root.is_dir():
        return None, None
    primary: tuple[str, dict] | None = None
    fallback: tuple[str, dict] | None = None
    for path in sorted(root.iterdir(), reverse=True):
        summary = _load_json(path / "summary.json")
        if not summary or summary.get("stage") != stage:
            continue
        if summary.get("status") != "completed":
            continue
        item = (path.name, summary)
        fallback = fallback or item
        if not summary.get("post_hoc"):
            primary = item
            break
    chosen = primary or fallback
    if not chosen:
        return None, None
    return chosen


def load_freeze() -> dict[str, Any] | None:
    path = project_root() / "experiments" / "metrics" / "metrics_freeze.json"
    data = _load_json(path)
    if data is None:
        return None
    return {
        "path": str(path.relative_to(project_root())),
        "frozen_at_utc": data.get("frozen_at_utc"),
        "git_commit": data.get("git_commit"),
        "spec_sha256": data.get("spec_sha256"),
        "n_tau": len(data.get("tau_p") or {}),
        "baseline_result_dirs": data.get("baseline_result_dirs"),
        "freeze_blind_to_pilot_questions": data.get("freeze_blind_to_pilot_questions"),
        "note_pilot_may_have_been_inspected_previously": data.get(
            "note_pilot_may_have_been_inspected_previously"
        ),
        "post_hoc": data.get("post_hoc", False),
    }


def list_metric_figures() -> list[dict[str, str]]:
    figs = project_root() / "docs" / "figures"
    out = []
    for name in (
        "fig50_baseline_distributions.png",
        "fig51_validity_checks.png",
        "fig52_pilot_primary_metrics.png",
        "fig53_win_tie_loss.png",
        "fig54_pool_oracle.png",
        "fig55_lambda_sensitivity.png",
        "fig56_scale_cost_cny.png",
    ):
        path = figs / name
        if path.is_file():
            out.append(
                {
                    "id": name.replace(".png", ""),
                    "filename": name,
                    "url": f"/api/v1/experiments/metrics/figures/{name}",
                    "title": name.replace("fig", "Figure ").replace("_", " "),
                }
            )
    return out


def build_metrics_dashboard() -> dict[str, Any]:
    freeze = load_freeze()
    combined = results_root() / "metrics_combined"
    baseline = _load_json(combined / "baseline_summary.json")
    pilot = _load_json(combined / "pilot_summary.json")
    scale = _load_json(combined / "scale_decision.json")

    # Prefer primary (non post-hoc) scale decision from results dirs
    scale_id, scale_summary = _latest_stage("scale_decision")
    if scale_summary and not scale_summary.get("post_hoc"):
        scale = scale_summary
    elif scale is None:
        scale = scale_summary

    freeze_id, freeze_summary = _latest_stage("metrics_freeze")
    if baseline is None:
        baseline = freeze_summary
    pilot_id, pilot_summary = _latest_stage("metrics_pilot")
    if pilot is None:
        pilot = pilot_summary

    decision = (scale or {}).get("decision") or {}
    return {
        "freeze": freeze,
        "baseline_experiment_id": (baseline or {}).get("experiment_id") or freeze_id,
        "pilot_experiment_id": (pilot or {}).get("experiment_id") or pilot_id,
        "scale_experiment_id": (scale or {}).get("experiment_id") or scale_id,
        "baseline": {
            "counts": (baseline or {}).get("counts"),
            "distributions": (baseline or {}).get("distributions"),
            "validity": (baseline or {}).get("validity"),
            "primary30_noise_floor": (baseline or {}).get("primary30_noise_floor")
            or ((baseline or {}).get("validity") or {}).get("V4"),
            "spec_sha256": (baseline or {}).get("spec_sha256"),
            "embedding_model": (baseline or {}).get("embedding_model"),
        },
        "pilot": {
            "condition_summaries": (pilot or {}).get("condition_summaries"),
            "n_primary_pairs": (pilot or {}).get("n_primary_pairs"),
            "pilot_dirs": (pilot or {}).get("pilot_dirs"),
            "freeze_note": (pilot or {}).get("freeze_note"),
        },
        "scale": {
            "verdict": decision.get("verdict"),
            "advisory": decision.get("advisory", True),
            "clauses": decision.get("clauses"),
            "win_tie_loss": (scale or {}).get("win_tie_loss"),
            "comparisons": (scale or {}).get("comparisons"),
            "cost_projection": (scale or {}).get("cost_projection"),
            "diagnostics": (scale or {}).get("diagnostics"),
            "noise_floor_V4": (scale or {}).get("noise_floor_V4"),
            "validity_caveat": (scale or {}).get("validity_caveat"),
            "post_hoc": (scale or {}).get("post_hoc", False),
            "scale_budget_cny": (scale or {}).get("scale_budget_cny"),
        },
        "figures": list_metric_figures(),
        "limitations": [
            "Metrics measure alignment with the gap matrix and surface diversity, not question quality.",
            "Fit-label control was null; gap labels are noisy.",
            "T is conditioned on the gap matrix; M2/M3 vs B1 is partly by construction.",
        ],
    }


def assignment_audit_path() -> Path:
    return project_root() / "experiments" / "metrics" / "assignment_audit.csv"


def load_assignment_audit() -> list[dict[str, str]]:
    path = assignment_audit_path()
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_assignment_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = assignment_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "question_text",
        "requirement_text",
        "rank",
        "cosine",
        "linked",
        "human_judges_related",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: ("" if row.get(k) is None else str(row.get(k))) for k in fields}
            writer.writerow(out)
    filled = sum(
        1
        for r in rows
        if str(r.get("human_judges_related") or "").strip() in {"0", "1"}
    )
    return {"path": str(path.relative_to(project_root())), "n_rows": len(rows), "n_filled": filled}


def question_ratings_path() -> Path:
    return project_root() / "experiments" / "human_eval" / "question_ratings.json"


def load_question_ratings() -> dict[str, Any]:
    path = question_ratings_path()
    data = _load_json(path)
    if data is None:
        return {
            "schema_version": 1,
            "status": "not_started",
            "note": (
                "Scaffold for eventual human evaluation of question quality. "
                "Automatic metrics are alignment/diversity proxies only."
            ),
            "ratings": [],
        }
    return data


def save_question_ratings(payload: dict[str, Any]) -> dict[str, Any]:
    path = question_ratings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_question_ratings()
    ratings = payload.get("ratings")
    if ratings is None:
        ratings = existing.get("ratings") or []
    out = {
        "schema_version": 1,
        "status": payload.get("status") or existing.get("status") or "in_progress",
        "note": payload.get("note")
        or existing.get("note")
        or "Human evaluation of question quality (separate from frozen automatic metrics).",
        "updated_at_utc": payload.get("updated_at_utc"),
        "rater_id": payload.get("rater_id") or existing.get("rater_id"),
        "ratings": ratings,
    }
    from datetime import datetime, timezone

    if not out.get("updated_at_utc"):
        out["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
