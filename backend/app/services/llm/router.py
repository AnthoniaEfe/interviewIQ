"""Central model router. Business logic selects LLMTask, never a model ID."""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from app.config.settings import Settings
from app.core.exceptions import ConfigurationError, LLMPermanentError, ModelUnavailableError
from app.services.llm.provider import LLMProvider, LLMResult
from app.services.llm.retry import with_retries
from app.services.llm.tasks import FAST_TASKS, LLMTask, PRIMARY_TASKS
from app.services.llm.usage import usage_store

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self.settings = settings
        self.provider = provider

    def model_for(self, task: LLMTask) -> str:
        if task in FAST_TASKS:
            return self.settings.llm_fast_model
        if task in PRIMARY_TASKS:
            return self.settings.llm_primary_model
        raise ConfigurationError(f"No routing policy for task {task}.")

    def is_research_critical(self, task: LLMTask) -> bool:
        return task in {
            LLMTask.QUESTION_GENERATION,
            LLMTask.QUESTION_REFINEMENT,
            LLMTask.PROFILE_ANALYSIS,
            LLMTask.GAP_ANALYSIS,
        }

    async def complete_structured(
        self,
        *,
        task: LLMTask,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        allow_configured_fallback: bool = False,
        request_id: str | None = None,
        experiment_id: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> tuple[T, LLMResult]:
        model = self.model_for(task)
        temp = self.settings.llm_temperature if temperature is None else temperature
        effort = reasoning_effort if reasoning_effort is not None else self.settings.llm_reasoning_effort

        async def _call(target_model: str):
            return await self.provider.complete_structured(
                model=target_model,
                schema=schema,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temp,
                reasoning_effort=effort,
                max_output_tokens=self.settings.llm_max_output_tokens,
                prompt_cache_key=prompt_cache_key,
            )

        try:
            (parsed, result), retries = await with_retries(lambda: _call(model))
        except (ModelUnavailableError, LLMPermanentError, ConfigurationError):
            raise
        except Exception:
            if (
                allow_configured_fallback
                and self.settings.llm_fallback_model
                and self.settings.llm_fallback_model != model
            ):
                logger.warning(
                    "Escalating to configured fallback model",
                    extra={
                        "task_type": task.value,
                        "model": self.settings.llm_fallback_model,
                        "provider": self.provider.name,
                        "request_id": request_id,
                        "experiment_id": experiment_id,
                        "success": False,
                    },
                )
                (parsed, result), retries = await with_retries(
                    lambda: _call(self.settings.llm_fallback_model)
                )
                result.fallback_used = True
                result.fallback_reason = f"primary_model_failed:{model}"
                result.retry_count = retries
                usage_store.add(result)
                self._log(task, result, request_id, experiment_id, success=True)
                return parsed, result
            raise

        result.retry_count = retries
        usage_store.add(result)
        self._log(task, result, request_id, experiment_id, success=True)
        return parsed, result

    def _log(
        self,
        task: LLMTask,
        result: LLMResult,
        request_id: str | None,
        experiment_id: str | None,
        *,
        success: bool,
    ) -> None:
        logger.info(
            "LLM call completed",
            extra={
                "request_id": request_id,
                "experiment_id": experiment_id,
                "task_type": task.value,
                "provider": result.provider,
                "model": result.model,
                "latency": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost": result.estimated_cost,
                "retry_count": result.retry_count,
                "fallback_used": result.fallback_used,
                "success": success,
            },
        )
