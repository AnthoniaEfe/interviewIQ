"""Sample five primary-30 T′ vs B1 pairs and build the rubric-pilot study.

No ratings are invented. Writes:
  experiments/human_eval/inputs/pilot5_Tprime_vs_B1.json
  experiments/human_eval/data/pilot/study.json  (gitignored)
  experiments/human_eval/keys/condition_key_pilot.json  (gitignored)
  experiments/human_eval/manifest_pilot.json  (hash only; safe to commit)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.human_eval.study import build_blinded_study  # noqa: E402

FULL = ROOT / "experiments/human_eval/inputs/primary30_Tprime_vs_B1.json"
PILOT_INPUTS = ROOT / "experiments/human_eval/inputs/pilot5_Tprime_vs_B1.json"
PILOT_SEED = 20261126
RATERS = ["R1", "R2", "R3"]
N_PILOT = 5


def main() -> int:
    pairs = json.loads(FULL.read_text(encoding="utf-8"))
    if len(pairs) < N_PILOT:
        raise SystemExit(f"need at least {N_PILOT} pairs, found {len(pairs)}")
    rng = random.Random(PILOT_SEED)
    chosen = rng.sample(list(range(len(pairs))), N_PILOT)
    chosen.sort()
    pilot = []
    for index in chosen:
        row = dict(pairs[index])
        row["pilot_sample_index"] = index
        row["pilot_seed"] = PILOT_SEED
        pilot.append(row)
    PILOT_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    PILOT_INPUTS.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")

    built = build_blinded_study(pilot, RATERS, seed=PILOT_SEED)
    out_dir = ROOT / "experiments" / "human_eval" / "data" / "pilot"
    key_dir = ROOT / "experiments" / "human_eval" / "keys"
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study.json").write_text(
        json.dumps(built["blinded"], indent=2) + "\n", encoding="utf-8"
    )
    (key_dir / "condition_key_pilot.json").write_text(
        json.dumps(built["condition_key"], indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        **built["manifest"],
        "mode": "pilot",
        "treatment": "Tprime",
        "baseline": "B1",
        "n_pairs_sampled": N_PILOT,
        "sample_seed": PILOT_SEED,
        "sample_pair_ids": [row["pair_id"] for row in pilot],
        "raters": RATERS,
        "inputs": str(PILOT_INPUTS.relative_to(ROOT)),
        "rubric": "docs/human_eval_rubric.md",
        "protocol": "docs/human_eval_protocol.md",
        "addendum": "docs/human_eval_tprime_addendum.md",
        "note": (
            "Rubric pilot only. Ratings must come from human raters. "
            "Pilot store is excluded from main analysis."
        ),
    }
    (ROOT / "experiments/human_eval/manifest_pilot.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
