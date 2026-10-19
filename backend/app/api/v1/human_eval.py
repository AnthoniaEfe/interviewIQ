"""Rater API. Responses contain blinded payloads only."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.services.human_eval.store import MODES, JsonlRatingStore

router = APIRouter(prefix="/human-eval")


class LoadStudyBody(BaseModel):
    mode: str
    blinded: dict


class IssueTokenBody(BaseModel):
    mode: str
    rater_id: str
    background: dict | None = None


class RatingBody(BaseModel):
    item_id: str
    relevance_x: int | None = None
    coverage_x: int | None = None
    non_redundancy_x: int | None = None
    relevance_y: int | None = None
    coverage_y: int | None = None
    non_redundancy_y: int | None = None
    preference: str | None = None
    comment: str | None = None
    elapsed_ms: int | None = None


class BackgroundBody(BaseModel):
    hr_or_interviewing_experience: bool | None = None
    years: int | None = Field(default=None, ge=0)


def get_store(request: Request) -> JsonlRatingStore:
    existing = getattr(request.app.state, "human_eval_store", None)
    if isinstance(existing, JsonlRatingStore):
        return existing
    configured = os.environ.get("HUMAN_EVAL_DATA_DIR", "").strip()
    if configured:
        root = Path(configured)
    else:
        root = Path(__file__).resolve().parents[4] / "experiments" / "human_eval" / "data"
    store = JsonlRatingStore(root)
    request.app.state.human_eval_store = store
    return store


def _admin(token: str | None) -> None:
    expected = os.environ.get("HUMAN_EVAL_ADMIN_TOKEN", "").strip()
    if not expected or token != expected:
        raise HTTPException(status_code=401, detail="admin token required")


def _mode(mode: str) -> str:
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
    return mode


@router.post("/admin/study")
def load_study(
    body: LoadStudyBody,
    request: Request,
    x_human_eval_admin: str | None = Header(default=None),
) -> dict:
    _admin(x_human_eval_admin)
    mode = _mode(body.mode)
    try:
        get_store(request).save_study(mode, body.blinded)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"mode": mode, "n_items": len(body.blinded.get("items") or [])}


@router.post("/admin/tokens")
def issue_token(
    body: IssueTokenBody,
    request: Request,
    x_human_eval_admin: str | None = Header(default=None),
) -> dict:
    _admin(x_human_eval_admin)
    mode = _mode(body.mode)
    try:
        token = get_store(request).issue_token(mode, body.rater_id, background=body.background)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"mode": mode, "rater_id": body.rater_id, "token": token, "path": f"/study/{token}?mode={mode}"}


@router.post("/admin/revoke")
def revoke_token(
    request: Request,
    token: str = Query(...),
    mode: str = Query(...),
    x_human_eval_admin: str | None = Header(default=None),
) -> dict:
    _admin(x_human_eval_admin)
    mode = _mode(mode)
    try:
        get_store(request).revoke_token(mode, token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {"revoked": True}


@router.get("/admin/export")
def export_ratings(
    request: Request,
    mode: str = Query(...),
    x_human_eval_admin: str | None = Header(default=None),
) -> dict:
    _admin(x_human_eval_admin)
    mode = _mode(mode)
    try:
        rows = get_store(request).export_ratings(mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"mode": mode, "n_ratings": len(rows), "ratings": rows}


@router.get("/session")
def session(
    request: Request,
    token: str = Query(...),
    mode: str = Query("main"),
) -> dict:
    mode = _mode(mode)
    try:
        return get_store(request).session(mode, token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/ratings")
def save_rating(
    body: RatingBody,
    request: Request,
    token: str = Query(...),
    mode: str = Query("main"),
) -> dict:
    mode = _mode(mode)
    try:
        row = get_store(request).upsert_rating(mode, token, body.model_dump())
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"saved": True, "item_id": row["item_id"], "complete": JsonlRatingStore.is_complete(row)}


@router.post("/background")
def save_background(
    body: BackgroundBody,
    request: Request,
    token: str = Query(...),
    mode: str = Query("main"),
) -> dict:
    mode = _mode(mode)
    try:
        get_store(request).set_background(mode, token, body.model_dump())
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"saved": True}


@router.get("/progress")
def progress(
    request: Request,
    token: str = Query(...),
    mode: str = Query("main"),
) -> dict:
    mode = _mode(mode)
    try:
        view = get_store(request).session(mode, token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {
        "mode": mode,
        "rater_id": view["rater_id"],
        "n_items": view["n_items"],
        "n_complete": view["n_complete"],
        "completion": view["completion"],
        "resume_item_id": view["resume_item_id"],
    }
