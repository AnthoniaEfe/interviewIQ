"""InterviewIQ pool generation (Terra) with strict post-validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, GapMatrix, InterviewQuestion, JobDescription
from app.schemas.pool import (
    PROFILE_PROBE,
    POOL_DIMENSIONS,
    PoolGeneratedSet,
    PoolQuestionItem,
    pool_item_to_interview_question,
)
from app.services.llm.prompts import PromptTemplate
from app.services.llm.provider import LLMResult
from app.services.llm.router import LLMRouter
from app.services.llm.tasks import LLMTask

POOL_PROMPT = PromptTemplate("generation/interviewiq_pool.md", version="1.1.0")
POOL_MAX_OUTPUT_TOKENS = 12000

USER_TEMPLATE = """\
SENIORITY LEVEL: {seniority}

Produce exactly {n} interview questions for the pool.

CANDIDATE PROFILE (JSON):
{candidate_json}

JOB PROFILE (JSON):
{job_json}

GAP MATRIX (JSON; use labels/weights/evidence as given; do not recompute scores):
{gap_json}

Allowed dimension values: {dimensions}
For target_gap use an exact job_competency from the gap matrix, or the literal {profile_probe}.
Each priority must be a float in (0, 1].
Return structured output only.
"""


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_pool_user_prompt(
    *,
    candidate: CandidateProfile,
    job: JobDescription,
    gap_matrix: GapMatrix,
    seniority: str,
    n: int,
) -> str:
    return USER_TEMPLATE.format(
        seniority=seniority,
        n=n,
        candidate_json=json.dumps(candidate.model_dump(), ensure_ascii=False),
        job_json=json.dumps(job.model_dump(), ensure_ascii=False),
        gap_json=json.dumps(gap_matrix.model_dump(), ensure_ascii=False),
        dimensions=", ".join(POOL_DIMENSIONS),
        profile_probe=PROFILE_PROBE,
    )


@dataclass
class PoolValidationResult:
    valid_items: list[PoolQuestionItem] = field(default_factory=list)
    interview_questions: list[InterviewQuestion] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    n_raw: int = 0
    n_valid: int = 0
    n_failed: int = 0


def validate_pool_items(
    items: list[PoolQuestionItem],
    gap_matrix: GapMatrix,
) -> PoolValidationResult:
    allowed = {g.job_competency for g in gap_matrix.gaps} | {PROFILE_PROBE}
    result = PoolValidationResult(n_raw=len(items))
    for idx, item in enumerate(items):
        reasons: list[str] = []
        try:
            # Re-validate priority bounds even if pydantic already did.
            if not (0.0 < float(item.priority) <= 1.0):
                reasons.append("priority_out_of_range")
        except (TypeError, ValueError):
            reasons.append("priority_invalid")
        if item.dimension not in POOL_DIMENSIONS:
            reasons.append("unknown_dimension")
        if item.target_gap not in allowed:
            reasons.append("unknown_target_gap")
        if not item.question.strip():
            reasons.append("empty_question")
        if reasons:
            result.failures.append(
                {
                    "index": idx,
                    "reasons": reasons,
                    "target_gap": item.target_gap,
                    "dimension": getattr(item, "dimension", None),
                    "priority": getattr(item, "priority", None),
                }
            )
            continue
        result.valid_items.append(item)
        result.interview_questions.append(pool_item_to_interview_question(item))
    result.n_valid = len(result.valid_items)
    result.n_failed = len(result.failures)
    return result


async def generate_pool(
    *,
    router: LLMRouter,
    settings: Settings,
    candidate: CandidateProfile,
    job: JobDescription,
    gap_matrix: GapMatrix,
    seniority: str,
    n_questions: int | None = None,
    experiment_id: str | None = None,
    request_id: str | None = None,
    max_output_tokens: int = POOL_MAX_OUTPUT_TOKENS,
    reasoning_effort: str | None = None,
) -> tuple[PoolGeneratedSet, LLMResult, str, str]:
    """Return (raw_parsed, usage, system_prompt, user_prompt)."""
    n = n_questions if n_questions is not None else settings.question_pool_size
    system_prompt = POOL_PROMPT.load()
    user_prompt = render_pool_user_prompt(
        candidate=candidate,
        job=job,
        gap_matrix=gap_matrix,
        seniority=seniority,
        n=n,
    )
    effort = reasoning_effort if reasoning_effort is not None else settings.llm_reasoning_effort
    parsed, usage = await router.complete_structured(
        task=LLMTask.QUESTION_GENERATION,
        schema=PoolGeneratedSet,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=None,
        reasoning_effort=effort,
        max_output_tokens=max_output_tokens,
        experiment_id=experiment_id,
        request_id=request_id,
        prompt_cache_key="interviewiq-pool-v1.1" if settings.llm_cache_enabled else None,
    )
    return parsed, usage, system_prompt, user_prompt
