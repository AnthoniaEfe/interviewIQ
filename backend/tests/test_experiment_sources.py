import json
from pathlib import Path

from app.services.experiments.reader import PairSourceIndex


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_manifest_sources(tmp_path: Path):
    pair_id = "rjf_00001_abc"
    _write(tmp_path / "pairs/profiles" / f"{pair_id}.txt", "Resume body for the pair.")
    _write(tmp_path / "pairs/jobs" / f"{pair_id}.txt", "Job description for the pair.")
    manifest = [
        {
            "pair_id": pair_id,
            "profile_id": pair_id,
            "profile_path": f"profiles/{pair_id}.txt",
            "job_path": f"jobs/{pair_id}.txt",
        }
    ]
    _write(tmp_path / "pairs/manifest.json", json.dumps(manifest))
    summary = {
        "config": {
            "data": {
                "manifest": "pairs/manifest.json",
                "pairs_dir": "pairs",
            }
        }
    }
    resume, job = PairSourceIndex(summary, root=tmp_path).texts_for(
        {"pair_id": pair_id, "profile_id": pair_id, "jd_id": pair_id}
    )
    assert resume == "Resume body for the pair."
    assert job == "Job description for the pair."


def test_directory_sources_and_path_escape(tmp_path: Path):
    _write(tmp_path / "profiles/p01.txt", "Legacy resume.")
    _write(tmp_path / "jobs/software_engineer.txt", "Legacy job.")
    secret = tmp_path.parent / "interviewiq-source-escape.txt"
    secret.write_text("do not read", encoding="utf-8")
    summary = {
        "config": {
            "data": {
                "profiles_dir": "profiles",
                "jobs_dir": "jobs",
            }
        }
    }
    index = PairSourceIndex(summary, root=tmp_path)
    resume, job = index.texts_for(
        {"profile_id": "p01", "jd_id": "software_engineer"}
    )
    assert resume == "Legacy resume."
    assert job == "Legacy job."
    escaped, _ = index.texts_for(
        {"profile_id": "../../interviewiq-source-escape", "jd_id": "software_engineer"}
    )
    secret.unlink(missing_ok=True)
    assert escaped is None
