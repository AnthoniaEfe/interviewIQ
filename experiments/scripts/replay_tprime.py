"""Replay selector T-prime on stored primary-30 pools.

No LLM calls and no embedding computation. If the similarity matrices used by
weighted MMR are not already on disk, this script writes a stop record and
exits. It does not select T-prime in that case.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

POOL_DIR = ROOT / "experiments/results/exp_20261011T130619Z/pool"
OUT_DIR = ROOT / "experiments/results/selector_tprime_replay"
PREREG = "docs/selector_tprime_prereg.md"
CACHE_DIRS = (
    ROOT / "experiments/results/exp_20261011T130619Z/similarity",
    ROOT / "experiments/results/exp_20261011T130619Z/embeddings",
    ROOT / "experiments/adaptive_selection_replay/similarity",
)


def _prereg_commit() -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", PREREG],
        cwd=ROOT,
        text=True,
    ).strip()


def _cached_similarity_files() -> list[Path]:
    found: list[Path] = []
    for directory in CACHE_DIRS:
        if not directory.is_dir():
            continue
        found.extend(sorted(directory.glob("*.npy")))
        found.extend(sorted(directory.glob("*.npz")))
    return found


def main() -> int:
    commit = _prereg_commit()
    if not commit:
        raise SystemExit(f"Refusing to run: {PREREG} has no commit yet")
    pools = sorted(p for p in POOL_DIR.glob("*.json") if not p.name.startswith("_"))
    cached = _cached_similarity_files()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(cached) < len(pools):
        payload = {
            "status": "stopped",
            "label": "post-hoc",
            "prereg_commit": commit,
            "prereg_path": PREREG,
            "n_pools": len(pools),
            "n_cached_similarity_files": len(cached),
            "cache_dirs_checked": [str(path.relative_to(ROOT)) for path in CACHE_DIRS],
            "llm_calls": 0,
            "embeddings_computed": 0,
            "tprime_metrics": None,
            "reason": (
                "T-prime uses the weighted-MMR score, which needs a stored cosine "
                "similarity matrix per pool. Those matrices are not on disk. "
                "No embedding was computed and no T-prime number was produced."
            ),
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(payload["reason"])
        return 2
    raise SystemExit(
        "Cached similarity files were found, but this runner does not load them yet. "
        "Stop rather than guessing a matrix layout."
    )


if __name__ == "__main__":
    raise SystemExit(main())
