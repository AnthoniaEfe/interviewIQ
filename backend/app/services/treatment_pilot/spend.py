"""Spend guard for B1 and T pool generation (paid Terra steps)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.treatment_pilot.constants import DEFAULT_MAX_COST_USD


class SpendAborted(RuntimeError):
    """Raised when a paid call would exceed --max-cost or lacks --confirm-spend."""


@dataclass
class TreatmentSpendGuard:
    max_cost: float = DEFAULT_MAX_COST_USD
    confirm_spend: bool = False
    prior_per_pair: float = 0.0
    running_cost: float = 0.0
    aborted: bool = False
    abort_message: str | None = None
    events: list[dict] = field(default_factory=list)

    def estimate_total(self, n_uncached_pairs: int) -> float:
        return round(self.prior_per_pair * n_uncached_pairs, 8)

    def require_confirm(self) -> None:
        if not self.confirm_spend:
            raise SpendAborted(
                "Refusing paid generation without --confirm-spend. "
                "Re-run with --confirm-spend --max-cost <usd> after reviewing the dry-run."
            )

    def check_next_pair(self) -> None:
        projected = self.running_cost + self.prior_per_pair
        if projected > self.max_cost + 1e-12:
            self.aborted = True
            self.abort_message = (
                f"Aborting: running cost {self.running_cost:.6f} USD plus "
                f"prior-per-pair estimate {self.prior_per_pair:.6f} USD would exceed "
                f"--max-cost {self.max_cost:.2f} USD."
            )
            raise SpendAborted(self.abort_message)

    def add_actual(self, cost: float, *, pair_id: str, stage: str) -> None:
        self.running_cost += float(cost)
        self.events.append(
            {"pair_id": pair_id, "stage": stage, "cost_usd": float(cost)}
        )
        if self.running_cost > self.max_cost + 1e-12:
            self.aborted = True
            self.abort_message = (
                f"Aborting: running cost {self.running_cost:.6f} USD "
                f"exceeded --max-cost {self.max_cost:.2f} USD."
            )
            raise SpendAborted(self.abort_message)

    def set_prior(self, prior: float) -> None:
        self.prior_per_pair = float(prior)
