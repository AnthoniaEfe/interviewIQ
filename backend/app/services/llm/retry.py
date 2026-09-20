from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.exceptions import LLMTransientError, StructuredOutputError

T = TypeVar("T")
logger = logging.getLogger(__name__)


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    retry_validation: bool = True,
) -> tuple[T, int]:
    """Retry transient provider failures and optional structured-output errors.

    Does not retry configuration, auth, or invalid-model errors.
    """
    last_error: Exception | None = None
    retries = 0
    for attempt in range(1, max_attempts + 1):
        try:
            result = await operation()
            return result, retries
        except StructuredOutputError as exc:
            last_error = exc
            if not retry_validation or attempt == max_attempts:
                raise
        except LLMTransientError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
        else:
            continue
        retries += 1
        delay = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
        logger.warning(
            "Retrying LLM call",
            extra={"retry_count": retries, "success": False},
        )
        await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error
