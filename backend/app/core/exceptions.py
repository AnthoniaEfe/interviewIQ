"""Application errors that must not leak secrets or provider internals to clients."""


class InterviewIQError(Exception):
    """Base application error."""

    def __init__(self, message: str, *, public_message: str | None = None) -> None:
        super().__init__(message)
        self.public_message = public_message or "An unexpected error occurred."


class ConfigurationError(InterviewIQError):
    """Invalid or missing configuration. Do not retry."""


class ModelUnavailableError(ConfigurationError):
    """Configured model ID is not available from the provider. Never substitute."""


class LLMTransientError(InterviewIQError):
    """Timeout, rate limit, or temporary provider failure. Retryable."""


class LLMPermanentError(InterviewIQError):
    """Auth, invalid request, or non-retryable provider error."""


class StructuredOutputError(InterviewIQError):
    """LLM response failed schema validation. May trigger controlled regeneration."""


class ValidationInputError(InterviewIQError):
    """Empty CV/JD, unsupported file, or insufficient structured data."""
