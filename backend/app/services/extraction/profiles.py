"""Stage-1 hybrid profile extraction: Luna structured NER over CV and JD text."""

from __future__ import annotations

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription
from app.services.llm.prompts import PromptTemplate
from app.services.llm.provider import LLMResult
from app.services.llm.router import LLMRouter
from app.services.llm.tasks import LLMTask

CV_PROMPT = PromptTemplate("extraction/candidate_profile.md", version="1.0.0")
JD_PROMPT = PromptTemplate("extraction/job_description.md", version="1.0.0")

# Structured profiles can exceed the default generation token cap.
EXTRACTION_MAX_OUTPUT_TOKENS = 16384

CV_USER_TEMPLATE = """\
CANDIDATE CV:
{cv}

Extract a structured candidate profile from this CV.
Keep source_text and original_mentions brief; do not copy the full document into any field.
Return structured output only.
"""

JD_USER_TEMPLATE = """\
JOB DESCRIPTION:
{jd}

Extract structured requirements from this job description.
Keep source_text and original_mentions brief; do not copy the full document into any field.
Return structured output only.
"""


async def extract_candidate(
    *,
    router: LLMRouter,
    settings: Settings,
    cv_text: str,
    experiment_id: str | None = None,
    request_id: str | None = None,
) -> tuple[CandidateProfile, LLMResult]:
    if not cv_text.strip():
        raise ValueError("Candidate CV is empty.")
    parsed, usage = await router.complete_structured(
        task=LLMTask.CV_EXTRACTION,
        schema=CandidateProfile,
        system_prompt=CV_PROMPT.load(),
        user_prompt=CV_USER_TEMPLATE.format(cv=cv_text.strip()),
        experiment_id=experiment_id,
        request_id=request_id,
        prompt_cache_key="interviewiq-cv-extraction-v1" if settings.llm_cache_enabled else None,
        max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS,
    )
    return parsed, usage


async def extract_job(
    *,
    router: LLMRouter,
    settings: Settings,
    jd_text: str,
    experiment_id: str | None = None,
    request_id: str | None = None,
) -> tuple[JobDescription, LLMResult]:
    if not jd_text.strip():
        raise ValueError("Job description is empty.")
    parsed, usage = await router.complete_structured(
        task=LLMTask.JD_EXTRACTION,
        schema=JobDescription,
        system_prompt=JD_PROMPT.load(),
        user_prompt=JD_USER_TEMPLATE.format(jd=jd_text.strip()),
        experiment_id=experiment_id,
        request_id=request_id,
        prompt_cache_key="interviewiq-jd-extraction-v1" if settings.llm_cache_enabled else None,
        max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS,
    )
    return parsed, usage
