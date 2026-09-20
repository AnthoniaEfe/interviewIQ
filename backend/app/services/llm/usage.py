from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.services.llm.provider import LLMResult


@dataclass
class UsageStore:
    records: list[LLMResult] = field(default_factory=list)

    def add(self, result: LLMResult) -> None:
        self.records.append(result)

    def summary(self) -> dict:
        by_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "estimated_cost": 0.0,
                "latency_ms": 0.0,
            }
        )
        for record in self.records:
            bucket = by_model[record.model]
            bucket["calls"] += 1
            bucket["input_tokens"] += record.input_tokens
            bucket["output_tokens"] += record.output_tokens
            bucket["cached_tokens"] += record.cached_tokens
            bucket["estimated_cost"] += record.estimated_cost
            bucket["latency_ms"] += record.latency_ms

        total_cost = sum(r.estimated_cost for r in self.records)
        total_latency = sum(r.latency_ms for r in self.records)
        n = len(self.records)
        return {
            "calls": n,
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "total_cached_tokens": sum(r.cached_tokens for r in self.records),
            "total_estimated_cost": round(total_cost, 8),
            "average_cost": round(total_cost / n, 8) if n else 0.0,
            "average_latency_ms": round(total_latency / n, 2) if n else 0.0,
            "by_model": {k: dict(v) for k, v in by_model.items()},
        }


usage_store = UsageStore()
