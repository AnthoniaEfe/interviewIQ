"""LLM judge over the same blinded presentation as human raters.

The judge task uses LLMTask.SIMPLE_TRANSFORMATION so the existing router sends
it to the fast model (Luna), not the generator (Terra). Paid calls happen only
when confirm_spend is true and the approval flag is set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.llm.tasks import LLMTask

JUDGE_PROMPT = Path(__file__).resolve().parents[4] / "prompts" / "evaluation" / "judge_v1.md"
JUDGE_TASK = LLMTask.SIMPLE_TRANSFORMATION


class JudgeOutput(BaseModel):
    relevance_x: int = Field(ge=1, le=5)
    coverage_x: int = Field(ge=1, le=5)
    non_redundancy_x: int = Field(ge=1, le=5)
    relevance_y: int = Field(ge=1, le=5)
    coverage_y: int = Field(ge=1, le=5)
    non_redundancy_y: int = Field(ge=1, le=5)
    preference: Literal["X", "Y", "no_preference"]


class JudgeParseError(ValueError):
    pass


def prompt_sha256(path: Path | None = None) -> str:
    target = path or JUDGE_PROMPT
    return hashlib.sha256(target.read_bytes()).hexdigest()


def render_user_prompt(item: dict[str, Any], *, order: str) -> str:
    if order == "xy":
        first, second = item["set_x"], item["set_y"]
        first_name, second_name = "Set X", "Set Y"
    elif order == "yx":
        first, second = item["set_y"], item["set_x"]
        first_name, second_name = "Set Y", "Set X"
    else:
        raise ValueError("order must be 'xy' or 'yx'")
    return (
        "JOB SUMMARY:\n"
        f"{item.get('job_summary', '')}\n\n"
        "CANDIDATE SUMMARY:\n"
        f"{item.get('candidate_summary', '')}\n\n"
        f"{first_name}:\n"
        + "\n".join(f"- {question}" for question in first)
        + f"\n\n{second_name}:\n"
        + "\n".join(f"- {question}" for question in second)
        + "\n\nReturn the JSON scores for Set X and Set Y as labeled above."
    )


def parse_judge_output(payload: Any) -> JudgeOutput:
    if isinstance(payload, JudgeOutput):
        return payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise JudgeParseError("judge output is not JSON") from exc
    if not isinstance(payload, dict):
        raise JudgeParseError("judge output is not an object")
    try:
        return JudgeOutput.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise JudgeParseError(str(exc)) from exc


def position_bias(first: JudgeOutput, second: JudgeOutput) -> bool:
    """True when both presentations prefer the slot that was shown first.

    `first` is the xy order (X shown before Y). `second` is the yx order
    (Y shown before X). Always choosing the first-shown set means preference
    X on the first call and preference Y on the second call.
    """
    return first.preference == "X" and second.preference == "Y"


async def judge_item(
    router: Any,
    item: dict[str, Any],
    *,
    confirm_spend: bool,
    approved: bool,
    cache: dict[str, Any],
    max_cost_usd: float,
    system_prompt: str,
) -> dict[str, Any]:
    if not approved:
        raise RuntimeError("judge thresholds are not approved; refusing to call the model")
    orders = ("xy", "yx")
    if not confirm_spend:
        return {"dry_run": True, "calls": 0, "orders": list(orders), "item_id": item.get("item_id")}
    outputs = {}
    calls = 0
    spent = float(cache.get("_spent_usd") or 0.0)
    for order in orders:
        key = hashlib.sha256(
            json.dumps(
                {
                    "item_id": item.get("item_id"),
                    "order": order,
                    "prompt_sha256": prompt_sha256(),
                    "set_x": item.get("set_x"),
                    "set_y": item.get("set_y"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if key in cache:
            outputs[order] = parse_judge_output(cache[key]["parsed"])
            continue
        if spent >= max_cost_usd:
            raise RuntimeError(f"judge spend {spent:.6f} reached max_cost {max_cost_usd:.6f}")
        parsed, usage = await router.complete_structured(
            task=JUDGE_TASK,
            schema=JudgeOutput,
            system_prompt=system_prompt,
            user_prompt=render_user_prompt(item, order=order),
            temperature=None,
            request_id=key,
            prompt_cache_key=key,
        )
        output = parse_judge_output(parsed)
        cost = float(getattr(usage, "estimated_cost", 0.0) or 0.0)
        spent += cost
        calls += 1
        cache[key] = {"parsed": output.model_dump(), "cost_usd": cost}
        outputs[order] = output
    cache["_spent_usd"] = spent
    return {
        "dry_run": False,
        "calls": calls,
        "item_id": item.get("item_id"),
        "xy": outputs["xy"].model_dump(),
        "yx": outputs["yx"].model_dump(),
        "position_bias": position_bias(outputs["xy"], outputs["yx"]),
        "spent_usd": spent,
    }
