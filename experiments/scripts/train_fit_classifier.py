"""Train a local resume–JD fit classifier on the Hugging Face train split.

This does NOT fine-tune gpt-5.6-terra / gpt-5.6-luna.
Those remain API models. Training here calibrates the local embedding-based
fit scorer against human fit labels for research evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.services.datasets.resume_fit import (  # noqa: E402
    HF_DATASET,
    LABELS,
    export_pair_files,
    load_split,
    sample_experiment_pairs,
)
from app.services.embeddings.service import (  # noqa: E402
    HashingEmbeddingService,
    SentenceTransformerEmbeddingService,
    resolve_device,
)


def _pair_features(embeddings, resumes: list[str], jobs: list[str]) -> np.ndarray:
    resume_vecs = embeddings.embed_batch(resumes)
    job_vecs = embeddings.embed_batch(jobs)
    # Concatenate [resume, job, |resume-job|, resume*job] for a simple fit model.
    diff = np.abs(resume_vecs - job_vecs)
    prod = resume_vecs * job_vecs
    return np.concatenate([resume_vecs, job_vecs, diff, prod], axis=1)


def _load_or_build_features(
    cache_path: Path,
    embeddings,
    resumes: list[str],
    jobs: list[str],
) -> np.ndarray:
    if cache_path.exists():
        print(f"[features] loading cache {cache_path}")
        return np.load(cache_path)
    print(f"[features] computing embeddings -> {cache_path}")
    features = _pair_features(embeddings, resumes, jobs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    return features


def train_fit_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    embedding_model: str,
    device: str,
    use_hashing: bool,
    max_train_rows: int | None,
    seed: int,
    feature_cache_dir: Path | None = None,
) -> tuple[Pipeline, dict, object]:
    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    if use_hashing:
        embeddings = HashingEmbeddingService(dim=128, model_name="hashing-test")
    else:
        embeddings = SentenceTransformerEmbeddingService(
            model_name=embedding_model,
            device=resolve_device(device),
        )

    print(f"[train] embedding model={embeddings.model_name} rows_train={len(train_df)}")
    train_cache = None
    test_cache = None
    if feature_cache_dir is not None and not use_hashing and max_train_rows is None:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = embeddings.model_name.replace("/", "_")
        train_cache = feature_cache_dir / f"train_features_{safe_name}.npy"
        test_cache = feature_cache_dir / f"test_features_{safe_name}.npy"

    if train_cache is not None and test_cache is not None:
        x_train = _load_or_build_features(
            train_cache,
            embeddings,
            train_df["resume_text"].astype(str).tolist(),
            train_df["job_description_text"].astype(str).tolist(),
        )
        print(f"[eval] rows_test={len(test_df)}")
        x_test = _load_or_build_features(
            test_cache,
            embeddings,
            test_df["resume_text"].astype(str).tolist(),
            test_df["job_description_text"].astype(str).tolist(),
        )
    else:
        x_train = _pair_features(
            embeddings,
            train_df["resume_text"].astype(str).tolist(),
            train_df["job_description_text"].astype(str).tolist(),
        )
        print(f"[eval] rows_test={len(test_df)}")
        x_test = _pair_features(
            embeddings,
            test_df["resume_text"].astype(str).tolist(),
            test_df["job_description_text"].astype(str).tolist(),
        )
    y_train = train_df["label_id"].to_numpy()
    y_test = test_df["label_id"].to_numpy()

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "dataset": HF_DATASET,
        "embedding_model": embeddings.model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": list(LABELS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return clf, metrics, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local resume–JD fit classifier")
    parser.add_argument(
        "--cache-dir",
        default="experiments/datasets/resume_job_fit/cache",
        help="Local parquet cache for HF splits",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/datasets/resume_job_fit",
        help="Output directory for model, metrics, and experiment pairs",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional subsample for faster iteration",
    )
    parser.add_argument(
        "--n-per-label",
        type=int,
        default=15,
        help="Experiment pairs to export per label (Good Fit / Potential Fit)",
    )
    parser.add_argument(
        "--hashing",
        action="store_true",
        help="Use hashing embeddings (tests / CPU smoke only)",
    )
    args = parser.parse_args()

    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / args.out_dir
    models_dir = out_dir / "models"
    pairs_dir = out_dir / "experiment_pairs"
    models_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train", cache_dir=cache_dir)
    test_df = load_split("test", cache_dir=cache_dir)

    clf, metrics, embeddings = train_fit_model(
        train_df,
        test_df,
        embedding_model=args.embedding_model,
        device=args.device,
        use_hashing=args.hashing,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        feature_cache_dir=cache_dir / "features",
    )

    model_path = models_dir / "fit_classifier.joblib"
    joblib.dump(
        {
            "classifier": clf,
            "labels": list(LABELS),
            "embedding_model": metrics["embedding_model"],
            "dataset": HF_DATASET,
        },
        model_path,
    )
    metrics_path = models_dir / "fit_classifier_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sample = sample_experiment_pairs(train_df, n_per_label=args.n_per_label, seed=args.seed)
    manifest = export_pair_files(sample, pairs_dir)
    (pairs_dir / "sample_meta.json").write_text(
        json.dumps(
            {
                "source_split": "train",
                "n_per_label": args.n_per_label,
                "seed": args.seed,
                "labels": sample["label"].value_counts().to_dict(),
                "pairs": len(manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "trained",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "experiment_pairs": len(manifest),
        "pairs_dir": str(pairs_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
