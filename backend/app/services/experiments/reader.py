"""Experiment result readers for the research dashboard API."""

from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def results_root() -> Path:
    return project_root() / "experiments" / "results"


def list_experiments() -> list[dict]:
    root = results_root()
    if not root.exists():
        return []
    items: list[dict] = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir() or path.name.startswith("."):
            continue
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        items.append(
            {
                "experiment_id": summary.get("experiment_id", path.name),
                "status": summary.get("status"),
                "generation_mode": summary.get("generation_mode"),
                "primary_dataset": summary.get("primary_dataset"),
                "model": summary.get("model"),
                "provider": summary.get("provider"),
                "combinations_attempted": summary.get("combinations_attempted"),
                "combinations_ok": summary.get("combinations_ok"),
                "combinations_failed": summary.get("combinations_failed"),
                "total_estimated_cost_usd": summary.get("total_estimated_cost_usd"),
                "average_latency_ms": summary.get("average_latency_ms"),
                "human_evaluation": summary.get("human_evaluation"),
                "final_question_count_requested": summary.get(
                    "final_question_count_requested"
                ),
            }
        )
    return items


def _read_under_root(path: Path, root: Path) -> str | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None
    return resolved.read_text(encoding="utf-8")


class PairSourceIndex:
    """Resolve the CV and JD text that a recorded pair was generated from."""

    def __init__(self, summary: dict, *, root: Path | None = None) -> None:
        self.root = (root or project_root()).resolve()
        data = (summary.get("config") or {}).get("data") or {}
        self._by_id: dict[str, tuple[Path, Path]] = {}
        self._profiles_dir: Path | None = None
        self._jobs_dir: Path | None = None
        if data.get("manifest"):
            self._index_manifest(data)
        else:
            self._profiles_dir = self.root / data.get(
                "profiles_dir", "experiments/datasets/profiles"
            )
            self._jobs_dir = self.root / data.get(
                "jobs_dir", "experiments/datasets/jobs"
            )

    def _index_manifest(self, data: dict) -> None:
        manifest_path = (self.root / data["manifest"]).resolve()
        if not manifest_path.is_file() or not manifest_path.is_relative_to(self.root):
            return
        pairs_dir = self.root / data.get(
            "pairs_dir", str(manifest_path.parent.relative_to(self.root))
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest:
            cv_path = pairs_dir / item["profile_path"]
            jd_path = pairs_dir / item["job_path"]
            for key in (item.get("pair_id"), item.get("profile_id")):
                if key:
                    self._by_id[str(key)] = (cv_path, jd_path)

    def texts_for(self, record: dict) -> tuple[str | None, str | None]:
        lookup_id = record.get("pair_id") or record.get("profile_id")
        if lookup_id and str(lookup_id) in self._by_id:
            cv_path, jd_path = self._by_id[str(lookup_id)]
            return (
                _read_under_root(cv_path, self.root),
                _read_under_root(jd_path, self.root),
            )
        profile_id = record.get("profile_id")
        jd_id = record.get("jd_id")
        if self._profiles_dir is None or self._jobs_dir is None:
            return None, None
        if not profile_id or not jd_id:
            return None, None
        return (
            _read_under_root(self._profiles_dir / f"{profile_id}.txt", self.root),
            _read_under_root(self._jobs_dir / f"{jd_id}.txt", self.root),
        )


def load_experiment(experiment_id: str) -> dict:
    path = results_root() / experiment_id
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = PairSourceIndex(summary)
    pairs: list[dict] = []
    baseline_dir = path / "baseline"
    if baseline_dir.exists():
        for file in sorted(baseline_dir.glob("*.json")):
            if file.name.startswith("_"):
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            usage = record.get("usage") or {}
            questions = record.get("questions") or []
            resume_text, job_description_text = sources.texts_for(record)
            pairs.append(
                {
                    "file": file.name,
                    "profile_id": record.get("profile_id"),
                    "jd_id": record.get("jd_id"),
                    "pair_id": record.get("pair_id"),
                    "fit_label": record.get("fit_label"),
                    "seniority": record.get("seniority"),
                    "model": record.get("model"),
                    "n_questions": len(questions),
                    "questions": questions,
                    "resume_text": resume_text,
                    "job_description_text": job_description_text,
                    "latency_ms": usage.get("latency_ms"),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "cached_tokens": usage.get("cached_tokens"),
                    "estimated_cost": usage.get("estimated_cost"),
                    "fallback_used": usage.get("fallback_used", False),
                    "generated_at": record.get("generated_at"),
                }
            )

    by_label: dict[str, dict] = {}
    for pair in pairs:
        label = pair.get("fit_label") or "unlabeled"
        bucket = by_label.setdefault(
            label,
            {
                "label": label,
                "n": 0,
                "total_cost": 0.0,
                "total_latency_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
        )
        bucket["n"] += 1
        bucket["total_cost"] += float(pair.get("estimated_cost") or 0)
        bucket["total_latency_ms"] += float(pair.get("latency_ms") or 0)
        bucket["total_input_tokens"] += int(pair.get("input_tokens") or 0)
        bucket["total_output_tokens"] += int(pair.get("output_tokens") or 0)

    label_stats = []
    for bucket in by_label.values():
        n = bucket["n"] or 1
        label_stats.append(
            {
                **bucket,
                "avg_cost": round(bucket["total_cost"] / n, 8),
                "avg_latency_ms": round(bucket["total_latency_ms"] / n, 2),
                "total_cost": round(bucket["total_cost"], 8),
            }
        )

    return {
        "summary": summary,
        "pairs": pairs,
        "label_stats": sorted(label_stats, key=lambda x: x["label"]),
        "charts": {
            "latency_ms": [
                {
                    "index": i + 1,
                    "pair_id": p.get("pair_id") or p.get("profile_id"),
                    "fit_label": p.get("fit_label") or "unlabeled",
                    "value": p.get("latency_ms") or 0,
                }
                for i, p in enumerate(pairs)
            ],
            "cost_usd": [
                {
                    "index": i + 1,
                    "pair_id": p.get("pair_id") or p.get("profile_id"),
                    "fit_label": p.get("fit_label") or "unlabeled",
                    "value": p.get("estimated_cost") or 0,
                }
                for i, p in enumerate(pairs)
            ],
            "tokens": [
                {
                    "index": i + 1,
                    "pair_id": p.get("pair_id") or p.get("profile_id"),
                    "fit_label": p.get("fit_label") or "unlabeled",
                    "input": p.get("input_tokens") or 0,
                    "output": p.get("output_tokens") or 0,
                }
                for i, p in enumerate(pairs)
            ],
        },
    }
