# LLM routing

Luna is used for high-volume structured work. Terra is used for research-critical reasoning and generation.

```
LLMTask.CV_EXTRACTION            → LLM_FAST_MODEL     (gpt-5.6-luna)
LLMTask.JD_EXTRACTION            → LLM_FAST_MODEL
LLMTask.NORMALIZATION            → LLM_FAST_MODEL
LLMTask.CLASSIFICATION           → LLM_FAST_MODEL
LLMTask.QUESTION_VALIDATION      → LLM_FAST_MODEL
LLMTask.SIMPLE_TRANSFORMATION    → LLM_FAST_MODEL
LLMTask.PROFILE_ANALYSIS         → LLM_PRIMARY_MODEL  (gpt-5.6-terra)
LLMTask.GAP_ANALYSIS             → LLM_PRIMARY_MODEL
LLMTask.QUESTION_GENERATION      → LLM_PRIMARY_MODEL
LLMTask.QUESTION_REFINEMENT      → LLM_PRIMARY_MODEL
```

Business logic never selects a model ID. It selects an `LLMTask`.

## Failure policy

- Invalid API key, invalid model ID, invalid configuration: no retry, no substitution.
- Timeout / rate limit / 5xx: exponential backoff on the **same** model.
- Structured-output validation failure: controlled regeneration on the same model.
- Optional `LLM_FALLBACK_MODEL` only when explicitly configured and requested; results must set `fallback_used=true` and `fallback_reason`.

## Provider

OpenAI is the only implemented provider. Gemini remains a future benchmarking option and is not integrated yet.
