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


def load_experiment(experiment_id: str) -> dict:
    path = results_root() / experiment_id
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs: list[dict] = []
    baseline_dir = path / "baseline"
    if baseline_dir.exists():
        for file in sorted(baseline_dir.glob("*.json")):
            if file.name.startswith("_"):
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            usage = record.get("usage") or {}
            questions = record.get("questions") or []
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
