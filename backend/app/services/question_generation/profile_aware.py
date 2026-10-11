"""Profile-aware baseline (B1): extraction JSON only → Terra → N questions."""

from __future__ import annotations

import hashlib
import json

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription
from app.schemas.generation import BaselineGeneratedSet
from app.services.llm.prompts import PromptTemplate
from app.services.llm.provider import LLMResult
from app.services.llm.router import LLMRouter
from app.services.llm.tasks import LLMTask
from app.services.question_generation.pool import POOL_MAX_OUTPUT_TOKENS

PROFILE_AWARE_PROMPT = PromptTemplate(
    "generation/profile_aware_baseline.md", version="1.0.0"
)

USER_TEMPLATE = """\
SENIORITY LEVEL: {seniority}

Generate exactly {n} interview questions appropriate for this candidate and this role.

Five-dimension taxonomy (vary across the set): Technical, Behavioural, Situational, Cultural/Motivational, Role-Specific.

CANDIDATE PROFILE (JSON):
{candidate_json}

JOB PROFILE (JSON):
{job_json}

This is the PROFILE-AWARE BASELINE experimental condition:
- Analyse the candidate against the job from these profiles alone.
- Cover gaps and partial matches as well as probing strengths.
- Avoid redundant questions; reference concrete profile specifics.
- Do not invent candidate experience.
- You are NOT given a gap matrix, scores, weights, embeddings, or a question pool.

Return structured output only.
"""

# Tokens that must not appear in the rendered B1 prompt (gap/pool leakage).
FORBIDDEN_B1_MARKERS = (
    "gap_matrix",
    "match_label",
    "strong_match",
    "partial_match",
    "weak_match",
    "competency_weighted",
    "mmr_lambda",
    "question_pool",
    "profile_probe",
    "tfidf_score",
    "bm25_score",
    "semantic_score",
)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_profile_aware_user_prompt(
    *,
    candidate: CandidateProfile,
    job: JobDescription,
    seniority: str,
    n: int,
) -> str:
    return USER_TEMPLATE.format(
        seniority=seniority,
        n=n,
        candidate_json=json.dumps(candidate.model_dump(), ensure_ascii=False),
        job_json=json.dumps(job.model_dump(), ensure_ascii=False),
    )


def assert_no_gap_leakage(system_prompt: str, user_prompt: str) -> None:
    blob = (system_prompt + "\n" + user_prompt).lower()
    # Allow the instructional phrase "gap matrix" in the system prompt (negated).
    # Forbid numeric/score/pool artifacts from appearing as supplied inputs.
    for marker in (
        "tfidf_score",
        "bm25_score",
        "semantic_score",
        "mmr_lambda",
        "question_pool",
        '"weight":',
        "competency_weighted",
    ):
        if marker in blob:
            raise ValueError(f"B1 prompt leakage: found {marker!r}")


async def generate_profile_aware(
    *,
    router: LLMRouter,
    settings: Settings,
    candidate: CandidateProfile,
    job: JobDescription,
    seniority: str,
    n_questions: int | None = None,
    experiment_id: str | None = None,
    request_id: str | None = None,
    max_output_tokens: int = POOL_MAX_OUTPUT_TOKENS,
    reasoning_effort: str | None = None,
) -> tuple[BaselineGeneratedSet, LLMResult, str, str]:
    n = n_questions if n_questions is not None else settings.final_question_count
    system_prompt = PROFILE_AWARE_PROMPT.load()
    user_prompt = render_profile_aware_user_prompt(
        candidate=candidate, job=job, seniority=seniority, n=n
    )
    assert_no_gap_leakage(system_prompt, user_prompt)
    effort = reasoning_effort if reasoning_effort is not None else settings.llm_reasoning_effort
    parsed, usage = await router.complete_structured(
        task=LLMTask.QUESTION_GENERATION,
        schema=BaselineGeneratedSet,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=None,
        reasoning_effort=effort,
        max_output_tokens=max_output_tokens,
        experiment_id=experiment_id,
        request_id=request_id,
        prompt_cache_key=(
            "interviewiq-profile-aware-v1" if settings.llm_cache_enabled else None
        ),
    )
    return parsed, usage, system_prompt, user_prompt
