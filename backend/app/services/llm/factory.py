from app.services.llm.openai_provider import OpenAIProvider
from app.services.llm.provider import LLMProvider
from app.services.llm.router import LLMRouter
from app.config.settings import Settings


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider}. "
        "Only OpenAI is implemented in this phase."
    )


def build_router(settings: Settings) -> LLMRouter:
    return LLMRouter(settings, build_provider(settings))
