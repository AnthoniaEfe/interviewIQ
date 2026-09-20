from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from app.services.llm.tasks import LLMTask

T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel):
    content: Any
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0
    retry_count: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None
    raw_id: str | None = None


class LLMProvider(ABC):
    name: str

    @abstractmethod
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
        """Return a validated Pydantic object. Never parse JSON with regex."""

    @abstractmethod
    async def list_models(self) -> Sequence[str]:
        """Provider catalog used to fail closed on unknown model IDs."""
