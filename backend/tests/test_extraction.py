import pytest

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, JobDescription
from app.services.extraction.profiles import extract_candidate, extract_job
from app.services.llm.provider import LLMResult
from app.services.llm.tasks import LLMTask


class FakeRouter:
    def __init__(self) -> None:
        self.calls = []

    async def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["task"] == LLMTask.CV_EXTRACTION:
            parsed = CandidateProfile(name="Ada", technical_skills=[{"normalized_label": "Python"}])
        else:
            parsed = JobDescription(title="Engineer", required_skills=[{"normalized_label": "Python"}])
        usage = LLMResult(content=parsed.model_dump(), provider="openai", model="gpt-5.6-luna")
        return parsed, usage


@pytest.mark.asyncio
async def test_extract_candidate_routes_to_cv_extraction():
    router = FakeRouter()
    parsed, _ = await extract_candidate(
        router=router,
        settings=Settings(),
        cv_text="Ada is a Python developer.",
    )
    assert parsed.technical_skills[0].normalized_label == "Python"
    assert router.calls[0]["task"] == LLMTask.CV_EXTRACTION
    assert "supported by the CV" in router.calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_extract_job_routes_to_jd_extraction():
    router = FakeRouter()
    parsed, _ = await extract_job(
        router=router,
        settings=Settings(),
        jd_text="Software engineer requiring Python.",
    )
    assert parsed.required_skills[0].normalized_label == "Python"
    assert router.calls[0]["task"] == LLMTask.JD_EXTRACTION
    assert "stated in the job description" in router.calls[0]["system_prompt"]
