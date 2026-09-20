import pytest

from app.schemas.generation import BaselineGeneratedSet
from app.services.llm.provider import LLMResult
from app.services.llm.tasks import LLMTask
from app.services.question_generation.baseline import generate_baseline
from app.config.settings import Settings


class FakeRouter:
    def __init__(self) -> None:
        self.calls = []

    async def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        parsed = BaselineGeneratedSet(questions=[{"question": "How did you design the FastAPI service?"}])
        usage = LLMResult(content=parsed.model_dump(), provider="openai", model="gpt-5.6-terra")
        return parsed, usage


@pytest.mark.asyncio
async def test_baseline_routes_to_question_generation_task():
    router = FakeRouter()
    parsed, _ = await generate_baseline(
        router=router,
        settings=Settings(),
        cv_text="Python FastAPI developer",
        jd_text="Software Engineer requiring Python",
        seniority="junior",
        n_questions=1,
    )
    assert parsed.questions[0].question
    assert router.calls[0]["task"] == LLMTask.QUESTION_GENERATION
    assert "BASELINE" in router.calls[0]["system_prompt"]
