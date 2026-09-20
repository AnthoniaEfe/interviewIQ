from __future__ import annotations

from pydantic import BaseModel, Field


class BaselineGeneratedQuestion(BaseModel):
    question: str


class BaselineGeneratedSet(BaseModel):
    questions: list[BaselineGeneratedQuestion] = Field(min_length=1)
