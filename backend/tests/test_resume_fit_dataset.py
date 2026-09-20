import pandas as pd

from app.services.datasets.resume_fit import (
    LABELS,
    export_pair_files,
    sample_experiment_pairs,
)


def test_sample_and_export(tmp_path):
    df = pd.DataFrame(
        {
            "resume_text": [f"resume {i} python fastapi" for i in range(6)],
            "job_description_text": [f"job {i} software engineer" for i in range(6)],
            "label": [
                "Good Fit",
                "Good Fit",
                "Potential Fit",
                "Potential Fit",
                "No Fit",
                "No Fit",
            ],
            "label_id": [2, 2, 1, 1, 0, 0],
            "pair_id": [f"p{i}" for i in range(6)],
        }
    )
    sample = sample_experiment_pairs(df, n_per_label=1, seed=0)
    assert set(sample["label"]) <= {"Good Fit", "Potential Fit"}
    assert len(sample) == 2
    manifest = export_pair_files(sample, tmp_path)
    assert len(manifest) == 2
    assert (tmp_path / "manifest.json").exists()
    assert LABELS[0] == "No Fit"
