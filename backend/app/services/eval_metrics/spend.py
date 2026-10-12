"""Spend guard for optional Luna calls (extraction / dimension labels)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.eval_metrics.constants import (
    DEFAULT_MAX_COST_USD,
    EXTRACTION_PRIOR_PER_PAIR_USD,
)


class SpendAborted(RuntimeError):
    """Raised when a paid step would exceed --max-cost or lacks --confirm-spend."""


@dataclass
class MetricsSpendGuard:
    max_cost: float = DEFAULT_MAX_COST_USD
    confirm_spend: bool = False
    prior_per_pair: float = EXTRACTION_PRIOR_PER_PAIR_USD
    running_cost: float = 0.0
    aborted: bool = False
    abort_message: str | None = None
    events: list[dict] = field(default_factory=list)

    def estimate_total(self, n_uncached: int) -> float:
        return round(self.prior_per_pair * n_uncached, 8)

    def require_confirm(self) -> None:
        if not self.confirm_spend:
            raise SpendAborted(
                "Refusing paid LLM calls without --confirm-spend. "
                "Re-run with --confirm-spend --max-cost <usd> after reviewing the dry-run."
            )

    def check_next(self, prior: float | None = None) -> None:
        step = self.prior_per_pair if prior is None else float(prior)
        projected = self.running_cost + step
        if projected > self.max_cost + 1e-12:
            self.aborted = True
            self.abort_message = (
                f"Aborting: running cost {self.running_cost:.6f} USD plus "
                f"prior {step:.6f} USD would exceed --max-cost {self.max_cost:.2f} USD."
            )
            raise SpendAborted(self.abort_message)

    def add_actual(self, cost: float, *, label: str) -> None:
        self.running_cost += float(cost)
        self.events.append({"label": label, "cost_usd": float(cost)})
        if self.running_cost > self.max_cost + 1e-12:
            self.aborted = True
            self.abort_message = (
                f"Aborting: running cost {self.running_cost:.6f} USD "
                f"exceeded --max-cost {self.max_cost:.2f} USD."
            )
            raise SpendAborted(self.abort_message)
