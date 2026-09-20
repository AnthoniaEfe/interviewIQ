"""Baseline generation: raw CV + JD → Terra → N questions. No gap/MMR pipeline."""

from __future__ import annotations

from app.config.settings import Settings
from app.schemas.generation import BaselineGeneratedSet
from app.services.llm.prompts import PromptTemplate
from app.services.llm.provider import LLMResult
from app.services.llm.router import LLMRouter
from app.services.llm.tasks import LLMTask

PROMPT = PromptTemplate("generation/baseline.md", version="1.0.0")

USER_TEMPLATE = """\
JOB DESCRIPTION:
{jd}

CANDIDATE CV:
{cv}

SENIORITY LEVEL: {seniority}

Generate exactly {n} interview questions appropriate for this candidate and this role.

This is the BASELINE experimental condition:
- Do not perform profile-gap analysis.
- Do not apply competency budgets.
- Do not apply redundancy control or MMR.
- Do not invent candidate experience that is not in the CV.

Avoid generic questions such as "Tell me about yourself" unless explicitly requested.
Return structured output only.
"""


async def generate_baseline(
    *,
    router: LLMRouter,
    settings: Settings,
    cv_text: str,
    jd_text: str,
    seniority: str,
    n_questions: int | None = None,
    experiment_id: str | None = None,
    request_id: str | None = None,
) -> tuple[BaselineGeneratedSet, LLMResult]:
    if not cv_text.strip():
        raise ValueError("Candidate CV is empty.")
    if not jd_text.strip():
        raise ValueError("Job description is empty.")
    n = n_questions if n_questions is not None else settings.final_question_count
    parsed, usage = await router.complete_structured(
        task=LLMTask.QUESTION_GENERATION,
        schema=BaselineGeneratedSet,
        system_prompt=PROMPT.load(),
        user_prompt=USER_TEMPLATE.format(
            jd=jd_text.strip(),
            cv=cv_text.strip(),
            seniority=seniority,
            n=n,
        ),
        experiment_id=experiment_id,
        request_id=request_id,
        prompt_cache_key="interviewiq-baseline-v1" if settings.llm_cache_enabled else None,
    )
    return parsed, usage
