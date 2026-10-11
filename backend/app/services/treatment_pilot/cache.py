"""Disk cache for Terra generation outputs keyed by pair/prompt/model/params."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(
    *,
    pair_id: str,
    prompt_hash: str,
    model_id: str,
    parameters: dict[str, Any],
) -> str:
    payload = {
        "pair_id": pair_id,
        "prompt_hash": prompt_hash,
        "model_id": model_id,
        "parameters": parameters,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def read_cache(cache_dir: Path, key: str) -> dict | None:
    path = cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_cache(cache_dir: Path, key: str, record: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, key)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def index_generation_cache(
    results_dir: Path,
    *,
    stage_subdir: str,
    prompt_hash: str,
    model_id: str,
    parameters: dict[str, Any],
) -> dict[str, Path]:
    """Map pair request_id stem -> newest matching cached per-pair JSON."""
    index: dict[str, Path] = {}
    if not results_dir.is_dir():
        return {}
    for exp_dir in sorted(results_dir.iterdir()):
        stage = exp_dir / stage_subdir
        if not stage.is_dir():
            continue
        for path in stage.glob("*.json"):
            if path.name.startswith("_"):
                continue
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("prompt_hash") != prompt_hash:
                continue
            if rec.get("model") != model_id:
                continue
            params = rec.get("generation_parameters") or {}
            if params.get("reasoning_effort") != parameters.get("reasoning_effort"):
                continue
            if params.get("max_output_tokens") != parameters.get("max_output_tokens"):
                continue
            stem = path.stem
            index[stem] = path
    return index
