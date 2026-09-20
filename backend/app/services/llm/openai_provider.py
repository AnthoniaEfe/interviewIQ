from __future__ import annotations

import logging
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config.pricing import get_pricing
from app.config.settings import Settings
from app.core.exceptions import (
    LLMPermanentError,
    LLMTransientError,
    ModelUnavailableError,
    StructuredOutputError,
)
from app.services.llm.provider import LLMProvider, LLMResult

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise LLMPermanentError(
                "OPENAI_API_KEY is not set.",
                public_message="The language model is not configured.",
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )
        return self._client

    async def list_models(self) -> list[str]:
        client = self._get_client()
        page = await client.models.list()
        return [item.id for item in page.data]

    async def complete_structured(
        self,
        *,
        model: str,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        reasoning_effort: str | None,
        max_output_tokens: int,
        prompt_cache_key: str | None = None,
    ) -> tuple[T, LLMResult]:
        client = self._get_client()
        started = time.perf_counter()
        kwargs: dict = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text_format": schema,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key

        try:
            response = await client.responses.parse(**kwargs)
        except Exception as exc:  # noqa: BLE001 — mapped below
            raise self._map_error(exc, model) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise StructuredOutputError(
                "OpenAI response contained no parsed structured output.",
                public_message="Question generation failed validation.",
            )
        if not isinstance(parsed, schema):
            try:
                parsed = schema.model_validate(parsed)
            except ValidationError as exc:
                raise StructuredOutputError(
                    f"Structured output failed Pydantic validation: {exc}",
                    public_message="Question generation failed validation.",
                ) from exc

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached_tokens = 0
        input_details = getattr(usage, "input_tokens_details", None)
        if input_details is not None:
            cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

        pricing = get_pricing(self.settings, model)
        result = LLMResult(
            content=parsed.model_dump(),
            provider=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            latency_ms=round(latency_ms, 2),
            estimated_cost=pricing.estimate_cost(input_tokens, output_tokens, cached_tokens),
            raw_id=getattr(response, "id", None),
        )
        return parsed, result

    def _map_error(self, exc: Exception, model: str) -> Exception:
        status = getattr(exc, "status_code", None)
        message = str(exc)
        lowered = message.lower()
        if status == 401 or "invalid api key" in lowered:
            return LLMPermanentError(
                "OpenAI authentication failed.",
                public_message="The language model is not configured.",
            )
        if status == 404 or "model" in lowered and "not found" in lowered:
            return ModelUnavailableError(
                (
                    f"Configured model '{model}' is not available from OpenAI. "
                    "Update LLM_PRIMARY_MODEL / LLM_FAST_MODEL. "
                    "Do not silently substitute another model."
                ),
                public_message="The configured language model is unavailable.",
            )
        if status in TRANSIENT_STATUS or "rate limit" in lowered or "timeout" in lowered:
            return LLMTransientError(
                f"Transient OpenAI error (status={status}): {type(exc).__name__}",
                public_message="The language model is temporarily unavailable.",
            )
        return LLMPermanentError(
            f"OpenAI request failed: {type(exc).__name__}",
            public_message="Question generation failed.",
        )


async def verify_configured_models(provider: LLMProvider, settings: Settings) -> None:
    """Fail closed if configured IDs are missing. Never substitute."""
    catalog = set(await provider.list_models())
    missing = [
        model
        for model in (settings.llm_fast_model, settings.llm_primary_model)
        if model not in catalog
    ]
    if settings.llm_fallback_model and settings.llm_fallback_model not in catalog:
        missing.append(settings.llm_fallback_model)
    if missing:
        raise ModelUnavailableError(
            (
                "Configured model ID(s) not found in the provider catalog: "
                f"{', '.join(missing)}. Update environment configuration. "
                "InterviewIQ will not substitute another model."
            ),
            public_message="The configured language model is unavailable.",
        )
