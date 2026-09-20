from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.experiments.reader import list_experiments, load_experiment, project_root

router = APIRouter(prefix="/experiments")

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")
_MAX_UPLOAD_BYTES = 512_000


class UploadPairResponse(BaseModel):
    upload_id: str
    profile_path: str
    job_path: str
    seniority: str
    note: str


class ManualPairRequest(BaseModel):
    resume_text: str = Field(min_length=20)
    job_description_text: str = Field(min_length=20)
    seniority: str = "mid"
    label: str | None = None


def _uploads_root() -> Path:
    root = project_root() / "experiments" / "datasets" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


async def _read_upload(file: UploadFile | None, text: str | None, *, field: str) -> str:
    if file is not None and file.filename:
        raw = await file.read()
        if len(raw) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"{field} exceeds size limit.")
        try:
            return raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{field} must be UTF-8 text.",
            ) from exc
    if text and text.strip():
        return text.strip()
    raise HTTPException(status_code=400, detail=f"{field} is required.")


@router.get("")
async def get_experiments() -> dict:
    return {"experiments": list_experiments()}


@router.get("/uploads/list")
async def list_uploads() -> dict:
    root = _uploads_root()
    items = []
    for path in sorted(root.iterdir(), reverse=True):
        meta_path = path / "meta.json"
        if meta_path.exists():
            items.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return {"uploads": items}


@router.post("/uploads", response_model=UploadPairResponse)
async def upload_pair(
    resume_file: UploadFile | None = File(default=None),
    job_file: UploadFile | None = File(default=None),
    resume_text: str | None = Form(default=None),
    job_description_text: str | None = Form(default=None),
    seniority: str = Form(default="mid"),
    label: str | None = Form(default=None),
) -> UploadPairResponse:
    """Store a CV/JD pair for ad-hoc testing. Does not call the LLM."""
    if seniority not in {"junior", "mid", "senior"}:
        raise HTTPException(status_code=400, detail="seniority must be junior, mid, or senior.")

    resume = await _read_upload(resume_file, resume_text, field="resume")
    job = await _read_upload(job_file, job_description_text, field="job description")
    if len(resume) < 20 or len(job) < 20:
        raise HTTPException(status_code=400, detail="CV and JD must contain meaningful text.")

    upload_id = datetime.now(timezone.utc).strftime("upload_%Y%m%dT%H%M%SZ")
    dest = _uploads_root() / upload_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "resume.txt").write_text(resume, encoding="utf-8")
    (dest / "job.txt").write_text(job, encoding="utf-8")
    meta = {
        "upload_id": upload_id,
        "seniority": seniority,
        "label": label,
        "resume_chars": len(resume),
        "job_chars": len(job),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stored",
        "note": "Stored for testing. Generation can be triggered from the experiment runner.",
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return UploadPairResponse(
        upload_id=upload_id,
        profile_path=str((dest / "resume.txt").relative_to(project_root())),
        job_path=str((dest / "job.txt").relative_to(project_root())),
        seniority=seniority,
        note=meta["note"],
    )


@router.post("/uploads/json", response_model=UploadPairResponse)
async def upload_pair_json(payload: ManualPairRequest) -> UploadPairResponse:
    if payload.seniority not in {"junior", "mid", "senior"}:
        raise HTTPException(status_code=400, detail="seniority must be junior, mid, or senior.")
    upload_id = datetime.now(timezone.utc).strftime("upload_%Y%m%dT%H%M%SZ")
    dest = _uploads_root() / upload_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "resume.txt").write_text(payload.resume_text.strip(), encoding="utf-8")
    (dest / "job.txt").write_text(payload.job_description_text.strip(), encoding="utf-8")
    meta = {
        "upload_id": upload_id,
        "seniority": payload.seniority,
        "label": payload.label,
        "resume_chars": len(payload.resume_text.strip()),
        "job_chars": len(payload.job_description_text.strip()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stored",
        "note": "Stored for testing. Generation can be triggered from the experiment runner.",
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return UploadPairResponse(
        upload_id=upload_id,
        profile_path=str((dest / "resume.txt").relative_to(project_root())),
        job_path=str((dest / "job.txt").relative_to(project_root())),
        seniority=payload.seniority,
        note=meta["note"],
    )


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    if not _SAFE_ID.match(experiment_id):
        raise HTTPException(status_code=400, detail="Invalid experiment id.")
    try:
        return load_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{experiment_id}/pairs/{pair_file}")
async def get_pair(experiment_id: str, pair_file: str) -> dict:
    if not _SAFE_ID.match(experiment_id):
        raise HTTPException(status_code=400, detail="Invalid experiment id.")
    if "/" in pair_file or "\\" in pair_file or not pair_file.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid pair file.")
    path = project_root() / "experiments" / "results" / experiment_id / "baseline" / pair_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pair result not found.")
    return json.loads(path.read_text(encoding="utf-8"))
