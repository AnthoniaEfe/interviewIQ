"""I/O loaders for baseline / gap / (post-freeze) pilot artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.eval_metrics.constants import (
    B0_150_EXPERIMENT,
    B0_PRIMARY_REPLICATES,
    EXTRACTION_HOLDOUT,
    EXTRACTION_PRIMARY,
    GAP_HOLDOUT,
    GAP_PRIMARY,
    MANIFEST_150_REL,
    MANIFEST_HOLDOUT_REL,
    MANIFEST_PRIMARY_REL,
    PILOT_STAGES,
)
from app.services.gap_sensitivity.io_utils import project_root


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(rel: str) -> list[dict]:
    return load_json(project_root() / rel)


def pair_stem(profile_id: str, jd_id: str, seniority: str = "mid") -> str:
    return f"{profile_id}__{jd_id}__{seniority}"


def question_texts(record: dict) -> list[str]:
    """Extract question strings from baseline/treatment/pool record shapes."""
    raw = record.get("questions")
    if raw is None:
        raw = record.get("valid_questions")
    if raw is None:
        raw = record.get("raw_questions")
    out = []
    for q in raw or []:
        if isinstance(q, dict):
            t = str(q.get("question") or "").strip()
        else:
            t = str(q).strip()
        if t:
            out.append(t)
    return out


def load_baseline_record(exp_id: str, stem: str) -> dict | None:
    path = project_root() / "experiments" / "results" / exp_id / "baseline" / f"{stem}.json"
    if not path.is_file():
        return None
    return load_json(path)


def iter_baseline_dir(exp_id: str) -> list[dict]:
    d = project_root() / "experiments" / "results" / exp_id / "baseline"
    records = []
    for path in sorted(d.glob("*.json")):
        if path.name.startswith("_"):
            continue
        records.append(load_json(path))
    return records


def load_gap_record(exp_id: str, stem: str) -> dict | None:
    path = (
        project_root()
        / "experiments"
        / "results"
        / exp_id
        / "gap_analysis"
        / f"{stem}.json"
    )
    if not path.is_file():
        return None
    return load_json(path)


def requirements_from_gap(gap_record: dict) -> list[dict]:
    gm = gap_record.get("gap_matrix") or {}
    return list(gm.get("gaps") or [])


def index_gap_by_pair() -> dict[str, dict]:
    """Map pair_id -> gap record for the 60 stored pairs."""
    out: dict[str, dict] = {}
    for exp_id in (GAP_PRIMARY, GAP_HOLDOUT):
        d = project_root() / "experiments" / "results" / exp_id / "gap_analysis"
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name.startswith("_"):
                continue
            rec = load_json(path)
            pid = rec.get("pair_id")
            if pid:
                out[str(pid)] = rec
    return out


def index_extraction_by_pair() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for exp_id in (EXTRACTION_PRIMARY, EXTRACTION_HOLDOUT):
        d = project_root() / "experiments" / "results" / exp_id / "extraction"
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name.startswith("_"):
                continue
            rec = load_json(path)
            pid = rec.get("pair_id")
            if pid:
                out[str(pid)] = rec
    return out


def primary_pair_ids() -> list[str]:
    return [str(x["pair_id"]) for x in load_manifest(MANIFEST_PRIMARY_REL)]


def holdout_pair_ids() -> list[str]:
    return [str(x["pair_id"]) for x in load_manifest(MANIFEST_HOLDOUT_REL)]


def all_150_pairs() -> list[dict]:
    return load_manifest(MANIFEST_150_REL)


def b0_150_index() -> dict[str, dict]:
    """pair_id -> baseline record from the 150-run."""
    out: dict[str, dict] = {}
    for rec in iter_baseline_dir(B0_150_EXPERIMENT):
        pid = rec.get("pair_id")
        if pid:
            out[str(pid)] = rec
    return out


def b0_replicate_records(pair_id: str) -> list[dict]:
    """Three stored B0 generations for a primary-30 pair (may be fewer if missing)."""
    records = []
    for exp_id in B0_PRIMARY_REPLICATES:
        for rec in iter_baseline_dir(exp_id):
            if str(rec.get("pair_id")) == str(pair_id):
                records.append(rec)
                break
    return records


def detect_pilot_summaries() -> list[dict[str, str | None]]:
    """Read only stage/status from summary.json (allowed before freeze)."""
    results = project_root() / "experiments" / "results"
    found = []
    if not results.is_dir():
        return found
    for d in sorted(results.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        if not summary.is_file():
            continue
        try:
            data = load_json(summary)
        except (OSError, json.JSONDecodeError):
            continue
        stage = data.get("stage")
        if stage in PILOT_STAGES:
            found.append(
                {
                    "experiment_id": d.name,
                    "stage": stage,
                    "status": data.get("status"),
                }
            )
    return found


def resolve_pilot_dirs() -> dict[str, Path | None]:
    """Map condition source dirs from completed pilot experiment summaries."""
    found = detect_pilot_summaries()
    b1 = t = abl = None
    for item in found:
        if item["status"] != "completed":
            continue
        path = project_root() / "experiments" / "results" / str(item["experiment_id"])
        if item["stage"] == "baseline_profile_aware":
            # Prefer the full 30-pair run (later / larger)
            if b1 is None or path.name > b1.name:
                b1 = path
        elif item["stage"] == "treatment_pipeline":
            if t is None or path.name > t.name:
                t = path
        elif item["stage"] == "selection_ablation":
            if abl is None or path.name > abl.name:
                abl = path
    return {"B1": b1, "T": t, "ablation": abl}


def load_condition_records(condition_dir: Path, subdir: str) -> dict[str, dict]:
    """pair_id -> record for a condition subdirectory."""
    d = condition_dir / subdir
    out: dict[str, dict] = {}
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        if path.name.startswith("_"):
            continue
        rec = load_json(path)
        pid = rec.get("pair_id")
        if pid:
            out[str(pid)] = rec
        else:
            out[path.stem] = rec
    return out


def load_s0_mean_questions(ablation_dir: Path, pair_id: str, stem: str) -> list[str] | None:
    """S0 stores repeats; return mean-of-repeats question list (first file's questions
    is wrong — we need the mean metrics later; for metric computation we average
    per-repeat metric values, not question text. This helper lists repeat files.
    """
    d = ablation_dir / "S0"
    if not d.is_dir():
        return None
    # Prefer files matching stem; S0 may have _r0.._r4 suffixes
    matches = sorted(d.glob(f"{stem}*.json"))
    if not matches:
        matches = [p for p in sorted(d.glob("*.json")) if pair_id in p.read_text(encoding="utf-8")[:500]]
    return None  # callers use load_s0_repeat_records


def load_s0_repeat_records(ablation_dir: Path, stem: str) -> list[dict]:
    d = ablation_dir / "S0"
    if not d.is_dir():
        return []
    matches = sorted(d.glob(f"{stem}*.json"))
    if not matches:
        # flat naming without suffix
        single = d / f"{stem}.json"
        if single.is_file():
            return [load_json(single)]
    return [load_json(p) for p in matches]
