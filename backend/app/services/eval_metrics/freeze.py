"""Freeze hashing and verification for metrics v1."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.eval_metrics.constants import (
    METRIC_SPEC_REL,
    METRICS_FREEZE_REL,
    MODULE_HASH_GLOBS,
)
from app.services.gap_sensitivity.io_utils import git_commit_hash, project_root, write_json


class FreezeError(RuntimeError):
    """Raised when freeze is missing or hashed files differ."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(text.encode("utf-8"))


def list_module_files() -> list[Path]:
    root = project_root()
    files: list[Path] = []
    for pattern in MODULE_HASH_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen and p.is_file():
            seen.add(rp)
            out.append(p)
    return out


def file_hashes() -> dict[str, str]:
    root = project_root()
    return {
        str(p.relative_to(root)).replace("\\", "/"): sha256_file(p)
        for p in list_module_files()
    }


def working_tree_diff_hash() -> str | None:
    """Hash of git diff for metrics-related paths; None if clean."""
    import subprocess

    root = project_root()
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "backend/app/services/eval_metrics", "experiments/metrics", "prompts/analysis"],
            cwd=root,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    if not out.strip():
        return None
    return sha256_bytes(out.encode("utf-8"))


def build_spec_payload() -> dict[str, Any]:
    """Machine-readable frozen metric definitions (v1)."""
    return {
        "version": "v1",
        "title": "InterviewIQ automatic evaluation metrics",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_notes": "L2-normalised cosine; questions as plain text; requirements as job_competency",
        "bootstrap": {"n_resamples": 2000, "seed": 42},
        "metrics": {
            "M1.P": {
                "name": "mean_pairwise_cosine_similarity",
                "primary": True,
                "direction": "lower_better",
                "scope": "all_question_sets",
            },
            "M1.S1": {
                "name": "near_duplicate_pair_rate",
                "primary": False,
                "threshold": 0.85,
                "direction": "lower_better",
            },
            "M1.S2": {
                "name": "near_duplicate_pair_rate_sensitivity",
                "primary": False,
                "threshold": 0.75,
                "direction": "lower_better",
            },
            "M4.P": {
                "name": "vendi_score_cosine_kernel",
                "primary": True,
                "direction": "higher_better",
                "scope": "all_question_sets",
            },
            "M4.S": {
                "name": "dimension_entropy",
                "primary": False,
                "requires": "dimension_labels",
                "direction": "higher_better",
            },
            "M2.P": {
                "name": "gap_targeting_rate",
                "primary": True,
                "direction": "higher_better",
                "requires": "gap_matrix_and_assignment",
            },
            "M2.S1": {
                "name": "capped_gap_coverage",
                "primary": False,
                "direction": "higher_better",
            },
            "M2.S2": {
                "name": "gap_weight_captured",
                "primary": False,
                "direction": "higher_better",
            },
            "M3.P": {
                "name": "capped_requirement_coverage",
                "primary": True,
                "direction": "higher_better",
                "requires": "gap_matrix_and_assignment",
            },
            "M3.S1": {
                "name": "required_only_capped_coverage",
                "primary": False,
                "direction": "higher_better",
            },
            "M3.S2": {
                "name": "unlinked_rate",
                "primary": False,
                "direction": "descriptive",
            },
            "M3.S3": {
                "name": "dimension_coverage",
                "primary": False,
                "requires": "dimension_labels",
                "direction": "higher_better",
            },
        },
        "assignment": {
            "rule": "r*(q)=argmax_r cos(q,r); linked iff cos>=tau_p",
            "tau_p": "95th percentile of max_r cos(q',r) over B0 questions from other 149 pairs",
            "sensitivity": ["threshold_free", 0.35, 0.45, 0.55],
            "primary_variant": "calibrated_tau_p",
        },
        "decision_rule": {
            "SCALE_requires": [
                "vs_B1: >=2 wins among {M1.P,M2.P,M3.P,M4.P} and 0 losses",
                "vs_B2: >=1 win among {M2.P,M3.P} and 0 losses among four primaries",
                "vs_B0: >=3 wins and 0 losses",
                "projected_150_cost_cny <= scale_budget_cny (else SCALE_BLOCKED_BY_BUDGET advisory)",
            ],
            "win_definition": (
                "bootstrap 95% CI of favourable paired difference excludes 0 "
                "(favourable = lower M1.P, higher M2.P/M3.P/M4.P)"
            ),
            "otherwise": "DIAGNOSE",
            "advisory": True,
        },
        "limitations": [
            "Metrics measure alignment with the gap matrix and surface diversity, not question quality.",
            "Fit-label control was null (AUC 0.62 primary / 0.50 holdout); gap labels are noisy.",
            "T is conditioned on the gap matrix while B1 is only instructed in words; M2/M3 advantage is partly by construction.",
        ],
        "exploratory_not_frozen": [
            "lexical_specificity_proxy",
            "per_stratum_views",
        ],
    }


def write_spec(path: Path | None = None) -> tuple[Path, str]:
    root = project_root()
    path = path or (root / METRIC_SPEC_REL)
    payload = build_spec_payload()
    write_json(path, payload)
    return path, sha256_file(path)


def write_freeze(
    *,
    tau_map: dict[str, float],
    baseline_result_dirs: list[str],
    calibrated_thresholds_hash: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = project_root()
    spec_path = root / METRIC_SPEC_REL
    if not spec_path.is_file():
        write_spec(spec_path)
    freeze = {
        "version": "v1",
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit_hash(),
        "working_tree_diff_hash": working_tree_diff_hash(),
        "spec_sha256": sha256_file(spec_path),
        "file_hashes": file_hashes(),
        "calibrated_thresholds_hash": calibrated_thresholds_hash,
        "tau_p": tau_map,
        "baseline_result_dirs": baseline_result_dirs,
        "post_hoc": False,
    }
    if extra:
        freeze.update(extra)
    out = root / METRICS_FREEZE_REL
    write_json(out, freeze)
    return out


def load_freeze(path: Path | None = None) -> dict[str, Any]:
    root = project_root()
    path = path or (root / METRICS_FREEZE_REL)
    if not path.is_file():
        raise FreezeError(
            f"Freeze file missing: {path}. Run --stage metrics_freeze first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_freeze(*, allow_post_hoc: bool = False) -> dict[str, Any]:
    """Verify freeze file exists and hashed files match. Returns freeze dict.

    If hashes differ and allow_post_hoc, returns freeze with post_hoc=True tag
    for callers to stamp on outputs; otherwise raises FreezeError.
    """
    freeze = load_freeze()
    mismatches: list[str] = []
    current = file_hashes()
    frozen_hashes = freeze.get("file_hashes") or {}
    for rel, expected in frozen_hashes.items():
        actual = current.get(rel)
        if actual != expected:
            mismatches.append(rel)
    spec_path = project_root() / METRIC_SPEC_REL
    if spec_path.is_file():
        if sha256_file(spec_path) != freeze.get("spec_sha256"):
            mismatches.append(METRIC_SPEC_REL)
    if mismatches:
        if allow_post_hoc:
            freeze = dict(freeze)
            freeze["post_hoc"] = True
            freeze["hash_mismatches"] = mismatches
            freeze["post_hoc_tag"] = "POST-HOC, not primary"
            return freeze
        raise FreezeError(
            "Freeze hash mismatch for: "
            + ", ".join(mismatches)
            + ". Re-freeze as v1.1 or pass --allow-post-hoc (tags POST-HOC)."
        )
    freeze = dict(freeze)
    freeze["post_hoc"] = bool(freeze.get("post_hoc", False))
    return freeze
