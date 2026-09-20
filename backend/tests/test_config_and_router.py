from app.config.pricing import ModelPricing, get_pricing
from app.config.settings import Settings, get_settings
from app.services.llm.router import LLMRouter
from app.services.llm.tasks import LLMTask


class FakeProvider:
    name = "openai"

    async def complete_structured(self, **kwargs):
        raise AssertionError("not used")

    async def list_models(self):
        return ["gpt-5.6-luna", "gpt-5.6-terra"]


def test_settings_model_ids_from_env(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_FAST_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("LLM_PRIMARY_MODEL", "gpt-5.6-terra")
    settings = Settings()
    assert settings.llm_fast_model == "gpt-5.6-luna"
    assert settings.llm_primary_model == "gpt-5.6-terra"


def test_router_maps_luna_and_terra():
    settings = Settings()
    router = LLMRouter(settings, FakeProvider())
    assert router.model_for(LLMTask.CV_EXTRACTION) == settings.llm_fast_model
    assert router.model_for(LLMTask.JD_EXTRACTION) == settings.llm_fast_model
    assert router.model_for(LLMTask.NORMALIZATION) == settings.llm_fast_model
    assert router.model_for(LLMTask.QUESTION_VALIDATION) == settings.llm_fast_model
    assert router.model_for(LLMTask.QUESTION_GENERATION) == settings.llm_primary_model
    assert router.model_for(LLMTask.PROFILE_ANALYSIS) == settings.llm_primary_model
    assert router.model_for(LLMTask.GAP_ANALYSIS) == settings.llm_primary_model
    assert router.model_for(LLMTask.QUESTION_REFINEMENT) == settings.llm_primary_model


def test_cost_calculation_terra_defaults():
    pricing = ModelPricing(
        model="gpt-5.6-terra",
        input_cost_per_million=2.0,
        output_cost_per_million=12.0,
        cached_input_cost_per_million=0.2,
    )
    cost = pricing.estimate_cost(input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=0)
    assert cost == 14.0
    cached = pricing.estimate_cost(input_tokens=1_000_000, output_tokens=0, cached_tokens=1_000_000)
    assert cached == 0.2


def test_unknown_model_pricing_is_zero():
    settings = Settings()
    pricing = get_pricing(settings, "unknown-model-id")
    assert pricing.estimate_cost(100, 100) == 0.0
