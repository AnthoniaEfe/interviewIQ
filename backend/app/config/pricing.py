"""Centralized model pricing. Override via environment; do not scatter constants."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.settings import Settings


class ModelPricing(BaseModel):
    model: str
    input_cost_per_million: float
    output_cost_per_million: float
    cached_input_cost_per_million: float = 0.0

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        billed_input = max(input_tokens - cached_tokens, 0)
        input_cost = billed_input * self.input_cost_per_million / 1_000_000
        cached_cost = cached_tokens * self.cached_input_cost_per_million / 1_000_000
        output_cost = output_tokens * self.output_cost_per_million / 1_000_000
        return round(input_cost + cached_cost + output_cost, 8)


def pricing_table(settings: Settings) -> dict[str, ModelPricing]:
    return {
        settings.llm_primary_model: ModelPricing(
            model=settings.llm_primary_model,
            input_cost_per_million=settings.llm_primary_input_cost_per_million,
            output_cost_per_million=settings.llm_primary_output_cost_per_million,
            cached_input_cost_per_million=settings.llm_primary_cached_input_cost_per_million,
        ),
        settings.llm_fast_model: ModelPricing(
            model=settings.llm_fast_model,
            input_cost_per_million=settings.llm_fast_input_cost_per_million,
            output_cost_per_million=settings.llm_fast_output_cost_per_million,
            cached_input_cost_per_million=settings.llm_fast_cached_input_cost_per_million,
        ),
    }


def get_pricing(settings: Settings, model: str) -> ModelPricing:
    table = pricing_table(settings)
    if model in table:
        return table[model]
    return ModelPricing(
        model=model,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        cached_input_cost_per_million=0.0,
    )
