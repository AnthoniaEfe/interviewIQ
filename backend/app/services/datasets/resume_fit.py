"""Primary research dataset: cnamuangtoun/resume-job-description-fit.

Schema:
  resume_text, job_description_text, label ∈ {No Fit, Potential Fit, Good Fit}

Terra/Luna are never trained on this data. Local training covers only the
deterministic / embedding fit model used to calibrate profile-gap analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HF_DATASET = "cnamuangtoun/resume-job-description-fit"
SPLITS = {"train": "train.csv", "test": "test.csv"}
LABELS = ("No Fit", "Potential Fit", "Good Fit")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def hf_url(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {list(SPLITS)}.")
    return f"hf://datasets/{HF_DATASET}/{SPLITS[split]}"


def load_split(split: str, *, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load a split with pandas, optionally caching a local parquet copy."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / f"{split}.parquet"
        meta_path = cache_dir / f"{split}.meta.json"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

    df = pd.read_csv(hf_url(split))
    required = {"resume_text", "job_description_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip()
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    df["pair_id"] = [
        _pair_id(resume, jd, i)
        for i, (resume, jd) in enumerate(
            zip(df["resume_text"], df["job_description_text"], strict=True)
        )
    ]
    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)

    if cache_dir is not None:
        df.to_parquet(parquet_path, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    "dataset": HF_DATASET,
                    "split": split,
                    "source": hf_url(split),
                    "rows": int(len(df)),
                    "labels": df["label"].value_counts().to_dict(),
                    "unique_resumes": int(df["resume_text"].nunique()),
                    "unique_jobs": int(df["job_description_text"].nunique()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return df


def _pair_id(resume: str, jd: str, index: int) -> str:
    digest = hashlib.sha256(f"{resume}\n---\n{jd}".encode("utf-8")).hexdigest()[:12]
    return f"rjf_{index:05d}_{digest}"


def sample_experiment_pairs(
    df: pd.DataFrame,
    *,
    n_per_label: int,
    seed: int = 42,
    labels: tuple[str, ...] = ("Good Fit", "Potential Fit"),
) -> pd.DataFrame:
    """Stratified sample for generation experiments. Dimensions come from config."""
    parts: list[pd.DataFrame] = []
    for label in labels:
        subset = df[df["label"] == label]
        if subset.empty:
            continue
        take = min(n_per_label, len(subset))
        parts.append(subset.sample(n=take, random_state=seed))
    if not parts:
        raise ValueError("No rows available for the requested labels.")
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def export_pair_files(df: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Write CV/JD text files plus a manifest for the experiment runner."""
    profiles_dir = out_dir / "profiles"
    jobs_dir = out_dir / "jobs"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for _, row in df.iterrows():
        pair_id = str(row["pair_id"])
        profile_path = profiles_dir / f"{pair_id}.txt"
        job_path = jobs_dir / f"{pair_id}.txt"
        profile_path.write_text(str(row["resume_text"]), encoding="utf-8")
        job_path.write_text(str(row["job_description_text"]), encoding="utf-8")
        manifest.append(
            {
                "pair_id": pair_id,
                "profile_id": pair_id,
                "jd_id": pair_id,
                "label": row["label"],
                "label_id": int(row["label_id"]),
                "profile_path": str(profile_path.relative_to(out_dir)),
                "job_path": str(job_path.relative_to(out_dir)),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
