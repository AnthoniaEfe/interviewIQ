"""Round-3 independent dual coding under the worked rule.

Uses two LLM annotators (P1/P2) with different models and shuffled item
orders. Never loads the answer key or system labels for main items.
Calibration may be scored after the fact against adjudicated_calibration.csv.

Writes:
  experiments/gold_set/round3/annotations/annotator_P1.csv
  experiments/gold_set/round3/annotations/annotator_P2.csv
  experiments/gold_set/round3/annotations/_run_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import BaseModel, Field  # noqa: E402

from app.config.settings import Settings, get_settings  # noqa: E402
from app.services.gap_sensitivity.io_utils import git_commit_hash, write_json  # noqa: E402
from app.services.llm.factory import build_router  # noqa: E402
from app.services.llm.tasks import LLMTask  # noqa: E402

ROUND3 = ROOT / "experiments/gold_set/round3"
BRIEF = (ROUND3 / "training/worked_rule_brief.md").read_text(encoding="utf-8")
BLIND = ROOT / "experiments/gold_set/items_blind.json"
ASSIGN = ROUND3 / "assignment.json"
OUT = ROUND3 / "annotations"


class GoldLabel(BaseModel):
    label: int | None = Field(
        default=None,
        description="0 not evidenced, 1 partial, 2 met; null when cannot_judge",
    )
    cannot_judge: bool = False
    snippet: str = ""
    notes: str = ""


# Distinct task→model routes and seeds for independence.
ANNOTATORS = {
    "annotator_P1": {
        "model_task": LLMTask.CLASSIFICATION,  # luna
        "seed": 20261126,
        "persona": (
            "You are Annotator P1. Be strict: upgrade to met (2) only when the CV "
            "clearly shows the named requirement. Prefer partial or not evidenced "
            "when unsure. Use cannot_judge for vague soft requirements."
        ),
    },
    "annotator_P2": {
        "model_task": LLMTask.PROFILE_ANALYSIS,  # terra
        "seed": 20261127,
        "persona": (
            "You are Annotator P2. Apply the worked rule carefully. Related skills "
            "are partial (1), never met. Absence is 0. Vague attitude/culture "
            "phrases without a clear evidence standard are cannot_judge."
        ),
    },
}

FIELDS = ["item_id", "label", "cannot_judge", "snippet", "notes"]


def _load_items() -> dict[str, dict]:
    blind = json.loads(BLIND.read_text(encoding="utf-8"))
    items = {}
    for row in list(blind.get("calibration") or []) + list(blind.get("items") or []):
        items[str(row["item_id"])] = row
    return items


def _read_existing(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["item_id"]: r for r in csv.DictReader(fh) if r.get("item_id")}


def _write_csv(path: Path, rows_by_id: dict[str, dict], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for item_id in order:
            row = rows_by_id.get(item_id)
            if not row:
                continue
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def _normalize(raw: dict) -> dict:
    cj = raw.get("cannot_judge")
    if isinstance(cj, str):
        cj = cj.strip().lower() in {"1", "true", "yes", "c"}
    else:
        cj = bool(cj)
    label = raw.get("label")
    if cj:
        return {
            "label": "",
            "cannot_judge": "true",
            "snippet": str(raw.get("snippet") or "")[:500],
            "notes": str(raw.get("notes") or "")[:500],
        }
    try:
        lab = int(float(label))
    except (TypeError, ValueError):
        lab = None
    if lab not in {0, 1, 2}:
        # Fail closed to cannot_judge rather than inventing a score.
        return {
            "label": "",
            "cannot_judge": "true",
            "snippet": str(raw.get("snippet") or "")[:500],
            "notes": "parse_fallback_cannot_judge; " + str(raw.get("notes") or "")[:400],
        }
    return {
        "label": str(lab),
        "cannot_judge": "false",
        "snippet": str(raw.get("snippet") or "")[:500],
        "notes": str(raw.get("notes") or "")[:500],
    }


async def _label_one(router, *, annotator: str, item: dict, cfg: dict) -> tuple[dict, object]:
    del annotator
    system = (
        BRIEF
        + "\n\n"
        + cfg["persona"]
        + "\n\nIf cannot_judge is true, set label to null."
    )
    user = (
        f"item_id: {item['item_id']}\n"
        f"requirement: {item.get('requirement_text') or ''}\n\n"
        f"JOB DESCRIPTION (context only):\n{item.get('jd_text') or ''}\n\n"
        f"CV:\n{item.get('cv_text') or ''}\n"
    )
    parsed, result = await router.complete_structured(
        task=cfg["model_task"],
        schema=GoldLabel,
        system_prompt=system,
        user_prompt=user,
        temperature=0.0,
        reasoning_effort="low",
        max_output_tokens=400,
    )
    out = _normalize(parsed.model_dump())
    out["item_id"] = item["item_id"]
    return out, result


async def run_annotator(
    name: str,
    *,
    include_calibration: bool,
    resume: bool,
    limit: int | None,
) -> dict:
    get_settings.cache_clear()
    settings = Settings()
    router = build_router(settings)
    assign = json.loads(ASSIGN.read_text(encoding="utf-8"))
    cfg = ANNOTATORS[name]
    items = _load_items()
    ids = list(assign["annotators"][name]["main_ids"])
    if include_calibration:
        ids = list(assign["annotators"][name]["calibration_ids"]) + ids
    rng = random.Random(cfg["seed"])
    order = list(ids)
    rng.shuffle(order)
    if limit is not None:
        order = order[:limit]

    path = OUT / f"{name}.csv"
    existing = _read_existing(path) if resume else {}
    done = 0
    errors = []
    for item_id in order:
        if item_id in existing and resume:
            done += 1
            continue
        item = items[item_id]
        try:
            row, usage = await _label_one(router, annotator=name, item=item, cfg=cfg)
            existing[item_id] = row
            _write_csv(
                path,
                existing,
                assign["annotators"][name]["calibration_ids"]
                + assign["annotators"][name]["main_ids"]
                if include_calibration
                else assign["annotators"][name]["main_ids"],
            )
            done += 1
            cost = getattr(usage, "estimated_cost", None)
            print(f"[{name}] {done}/{len(order)} {item_id} label={row['label'] or 'C'} cost={cost}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"item_id": item_id, "error": str(exc)})
            print(f"[{name}] ERROR {item_id}: {exc}")
            time.sleep(1.5)
    return {"annotator": name, "n_done": len(existing), "n_target": len(order), "errors": errors, "path": str(path)}


async def main_async(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.annotators:
        summaries.append(
            await run_annotator(
                name,
                include_calibration=args.include_calibration,
                resume=args.resume,
                limit=args.limit,
            )
        )
    meta = {
        "round": 3,
        "git_commit": git_commit_hash(),
        "annotators": ANNOTATORS,
        "include_calibration": args.include_calibration,
        "summaries": summaries,
        "integrity": (
            "LLM dual coding under worked_rule_brief.md. "
            "Does not replace human round 1–2 unless humans replicate."
        ),
    }
    write_json(OUT / "_run_meta.json", meta)
    print(json.dumps(meta, indent=2, default=str))
    return 0 if all(not s["errors"] for s in summaries) else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", nargs="+", default=["annotator_P1", "annotator_P2"])
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug: label only first N after shuffle")
    args = parser.parse_args()
    import asyncio

    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
