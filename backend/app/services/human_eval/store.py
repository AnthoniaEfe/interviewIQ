"""Append-only JSONL rating store. Demo, pilot, and main are separate directories."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODES = ("pilot", "main", "demo")
CRITERIA = ("relevance", "coverage", "non_redundancy")
PREFERENCES = ("X", "Y", "no_preference")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class JsonlRatingStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def mode_dir(self, mode: str) -> Path:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        path = self.root / mode
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_study(self, mode: str) -> dict[str, Any]:
        path = self.mode_dir(mode) / "study.json"
        if not path.is_file():
            raise FileNotFoundError(f"no study loaded for mode {mode}")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_study(self, mode: str, blinded: dict[str, Any]) -> None:
        from app.services.human_eval.study import _assert_blind

        _assert_blind(blinded)
        if "condition_key" in blinded or "condition" in blinded:
            raise ValueError("refusing to store a condition key in the rating store")
        path = self.mode_dir(mode) / "study.json"
        path.write_text(json.dumps(blinded, indent=2), encoding="utf-8")

    def _raters_path(self, mode: str) -> Path:
        return self.mode_dir(mode) / "raters.jsonl"

    def _ratings_path(self, mode: str) -> Path:
        return self.mode_dir(mode) / "ratings.jsonl"

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def issue_token(
        self,
        mode: str,
        rater_id: str,
        *,
        background: dict[str, Any] | None = None,
    ) -> str:
        study = self.load_study(mode)
        known = {row["rater_id"] for row in study.get("assignments") or []}
        if rater_id not in known:
            raise KeyError(f"rater {rater_id} is not in the {mode} study")
        token = secrets.token_urlsafe(32)
        self._append(
            self._raters_path(mode),
            {
                "rater_id": rater_id,
                "token_sha256": hash_token(token),
                "background": background or {},
                "issued_at": _now(),
                "revoked": False,
                "mode": mode,
            },
        )
        return token

    def revoke_token(self, mode: str, token: str) -> None:
        rater = self.rater_for_token(mode, token)
        self._append(
            self._raters_path(mode),
            {
                "rater_id": rater["rater_id"],
                "token_sha256": hash_token(token),
                "revoked": True,
                "issued_at": _now(),
                "mode": mode,
            },
        )

    def rater_for_token(self, mode: str, token: str) -> dict[str, Any]:
        digest = hash_token(token)
        latest: dict[str, Any] | None = None
        for row in self._read_jsonl(self._raters_path(mode)):
            if row.get("token_sha256") == digest:
                latest = row
        if latest is None or latest.get("revoked"):
            raise PermissionError("unknown or revoked rater token")
        return latest

    def _assignment(self, study: dict[str, Any], rater_id: str) -> list[str]:
        for row in study.get("assignments") or []:
            if row.get("rater_id") == rater_id:
                return list(row.get("item_order") or [])
        raise KeyError(rater_id)

    def _items_by_id(self, study: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {item["item_id"]: item for item in study.get("items") or []}

    def latest_ratings(self, mode: str, rater_id: str) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._read_jsonl(self._ratings_path(mode)):
            if row.get("rater_id") == rater_id and row.get("mode") == mode:
                latest[str(row["item_id"])] = row
        return latest

    @staticmethod
    def is_complete(row: dict[str, Any] | None) -> bool:
        if not row:
            return False
        for side in ("x", "y"):
            for criterion in CRITERIA:
                value = row.get(f"{criterion}_{side}")
                if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                    return False
        return row.get("preference") in PREFERENCES

    def upsert_rating(self, mode: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        rater = self.rater_for_token(mode, token)
        study = self.load_study(mode)
        order = self._assignment(study, rater["rater_id"])
        item_id = str(payload.get("item_id") or "")
        if item_id not in order:
            raise KeyError(f"item {item_id} is not assigned to this rater")
        row = self._validate_rating(payload)
        row.update(
            {
                "rater_id": rater["rater_id"],
                "mode": mode,
                "is_demo": mode == "demo",
                "saved_at": _now(),
            }
        )
        self._append(self._ratings_path(mode), row)
        return row

    def _validate_rating(self, payload: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {"item_id": str(payload.get("item_id") or "")}
        if not row["item_id"]:
            raise ValueError("item_id is required")
        for side in ("x", "y"):
            for criterion in CRITERIA:
                key = f"{criterion}_{side}"
                value = payload.get(key)
                if value is None or value == "":
                    row[key] = None
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                    raise ValueError(f"{key} must be an integer from 1 to 5")
                row[key] = value
        preference = payload.get("preference")
        if preference in (None, ""):
            row["preference"] = None
        elif preference not in PREFERENCES:
            raise ValueError("preference must be X, Y, or no_preference")
        else:
            row["preference"] = preference
        comment = payload.get("comment")
        row["comment"] = "" if comment is None else str(comment)[:4000]
        elapsed = payload.get("elapsed_ms")
        if elapsed is None:
            row["elapsed_ms"] = None
        else:
            if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
                raise ValueError("elapsed_ms must be a non-negative int")
            row["elapsed_ms"] = elapsed
        return row

    def session(self, mode: str, token: str) -> dict[str, Any]:
        rater = self.rater_for_token(mode, token)
        study = self.load_study(mode)
        order = self._assignment(study, rater["rater_id"])
        by_id = self._items_by_id(study)
        drafts = self.latest_ratings(mode, rater["rater_id"])
        items = []
        first_open = None
        n_complete = 0
        for item_id in order:
            draft = drafts.get(item_id)
            complete = self.is_complete(draft)
            if complete:
                n_complete += 1
            elif first_open is None:
                first_open = item_id
            items.append({"item": by_id[item_id], "draft": draft, "complete": complete})
        n = len(order)
        return {
            "mode": mode,
            "rater_id": rater["rater_id"],
            "background": rater.get("background") or {},
            "n_items": n,
            "n_complete": n_complete,
            "completion": (n_complete / n) if n else 0.0,
            "resume_item_id": first_open,
            "items": items,
        }

    def set_background(self, mode: str, token: str, background: dict[str, Any]) -> None:
        rater = self.rater_for_token(mode, token)
        allowed = {
            "hr_or_interviewing_experience": background.get("hr_or_interviewing_experience"),
            "years": background.get("years"),
        }
        if allowed["hr_or_interviewing_experience"] not in {None, True, False}:
            raise ValueError("hr_or_interviewing_experience must be true, false, or omitted")
        years = allowed["years"]
        if years is not None and (isinstance(years, bool) or not isinstance(years, int) or years < 0):
            raise ValueError("years must be a non-negative int or omitted")
        self._append(
            self._raters_path(mode),
            {
                "rater_id": rater["rater_id"],
                "token_sha256": hash_token(token),
                "background": {k: v for k, v in allowed.items() if v is not None},
                "issued_at": rater.get("issued_at"),
                "revoked": False,
                "mode": mode,
                "background_updated_at": _now(),
            },
        )

    def export_ratings(self, mode: str) -> list[dict[str, Any]]:
        if mode == "demo":
            raise ValueError("demo ratings are excluded from export")
        study = self.load_study(mode)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        raw = self._read_jsonl(self._ratings_path(mode))
        for row in reversed(raw):
            if row.get("is_demo") or row.get("mode") == "demo":
                continue
            key = (str(row.get("rater_id")), str(row.get("item_id")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        rows.reverse()
        assigned = {
            (assignment["rater_id"], item_id)
            for assignment in study.get("assignments") or []
            for item_id in assignment.get("item_order") or []
        }
        return [row for row in rows if (row.get("rater_id"), row.get("item_id")) in assigned]
