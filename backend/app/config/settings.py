"""Typed application configuration. Model IDs are never hard-coded in business logic."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["openai"] = "openai"
    llm_fast_model: str = "gpt-5.6-luna"
    llm_primary_model: str = "gpt-5.6-terra"
    llm_fallback_model: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    google_api_key: str = ""

    llm_temperature: float = 0.7
    llm_reasoning_effort: str = "medium"
    llm_max_output_tokens: int = 4096

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "auto"

    question_pool_size: int = 30
    final_question_count: int = 10
    mmr_lambda: float = 0.7
    question_similarity_threshold: float = 0.85

    tfidf_min_similarity: float = 0.0
    bm25_min_score: float = 0.0
    semantic_min_similarity: float = 0.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    required_competency_weight: float = 1.0
    preferred_competency_weight: float = 0.6

    enable_llm_cache: bool = True
    enable_experiment_logging: bool = True
    disable_cache_for_experiments: bool = False
    log_level: str = "INFO"
    log_prompts: bool = False

    llm_primary_input_cost_per_million: float = 2.0
    llm_primary_cached_input_cost_per_million: float = 0.2
    llm_primary_output_cost_per_million: float = 12.0
    llm_fast_input_cost_per_million: float = 0.2
    llm_fast_cached_input_cost_per_million: float = 0.02
    llm_fast_output_cost_per_million: float = 1.2

    validate_models_on_startup: bool = False

    enable_tfidf: bool = True
    enable_bm25: bool = True
    enable_semantic: bool = True
    enable_gap_analysis: bool = True
    enable_competency_weighting: bool = True
    enable_mmr: bool = True
    enable_redundancy_filter: bool = True

    mongodb_uri: str = ""

    @field_validator("llm_fast_model", "llm_primary_model")
    @classmethod
    def model_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Configured model ID must not be empty.")
        return value.strip()

    @property
    def llm_cache_enabled(self) -> bool:
        if self.disable_cache_for_experiments:
            return False
        return self.enable_llm_cache


@lru_cache
def get_settings() -> Settings:
    return Settings()
