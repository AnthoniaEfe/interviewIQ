"""I/O helpers for gap-sensitivity runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def project_root() -> Path:
    return ROOT


def utc_experiment_id() -> str:
    return datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")


def git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def library_versions() -> dict[str, str | None]:
    names = [
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "sentence-transformers",
        "rank-bm25",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_extraction_records(source_dir: Path) -> list[dict]:
    extraction_dir = source_dir / "extraction"
    records = []
    for path in sorted(extraction_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        records.append(load_json(path))
    if not records:
        raise FileNotFoundError(f"No extraction artifacts in {extraction_dir}")
    return records


def load_gap_records(gap_dir: Path) -> dict[str, dict]:
    """Map request filename stem / request_id -> stored gap JSON."""
    out: dict[str, dict] = {}
    for path in sorted(gap_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        out[path.name] = load_json(path)
    return out


def resolve_cv_text(record: dict, pairs_dir: Path) -> str | None:
    profile_id = record.get("profile_id")
    if not profile_id:
        return None
    candidates = [
        pairs_dir / "profiles" / f"{profile_id}.txt",
        pairs_dir / f"{profile_id}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def request_id(record: dict) -> str:
    return (
        f"{record.get('profile_id')}__{record.get('jd_id')}__{record.get('seniority')}"
    )
