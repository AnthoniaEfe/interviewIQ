"""
run_batch_baseline.py
=====================
Generates baseline question sets for all 45 combinations:
  15 profiles x 3 job roles x 3 seniority levels

Reads from data/profiles/ and data/jds/.
Saves to results/baseline/{profile_id}_{jd_id}_{seniority}.json

Usage:
    python run_batch_baseline.py
    python run_batch_baseline.py --smoke   # 3 combinations only, fast check
"""

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


PROFILES_DIR = Path("data/profiles")
JDS_DIR      = Path("data/jds")
OUT_DIR      = Path("results/baseline")
SENIORITIES  = ["junior", "mid", "senior"]


def get_profiles() -> list[Path]:
    return sorted(PROFILES_DIR.glob("*.txt"))


def get_jds() -> list[Path]:
    return sorted(JDS_DIR.glob("*.txt"))


def run_one(cv: Path, jd: Path, seniority: str, out_dir: Path) -> dict:
    profile_id = cv.stem
    jd_id      = jd.stem
    out_path   = out_dir / f"{profile_id}_{jd_id}_{seniority}.json"

    if out_path.exists():
        print(f"  [skip]  {out_path.name} already exists")
        return {"status": "skipped", "file": str(out_path)}

    cmd = [
        sys.executable, "baseline.py",
        "--cv",        str(cv),
        "--jd",        str(jd),
        "--seniority", seniority,
        "--out",       str(out_path),
        "--profile-id", profile_id,
        "--jd-id",      jd_id,
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [ERROR] {out_path.name}")
        print(result.stderr)
        return {"status": "error", "file": str(out_path), "stderr": result.stderr}

    print(f"  [ok]    {out_path.name}  ({elapsed:.1f}s)")
    return {"status": "ok", "file": str(out_path), "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run 3 combinations only (fast sanity check)")
    args = parser.parse_args()

    profiles = get_profiles()
    jds      = get_jds()

    if not profiles:
        sys.exit(f"No .txt files found in {PROFILES_DIR}")
    if not jds:
        sys.exit(f"No .txt files found in {JDS_DIR}")

    combinations = list(product(profiles, jds, SENIORITIES))

    if args.smoke:
        combinations = combinations[:3]
        print(f"[smoke] Running {len(combinations)} combinations")
    else:
        print(f"[batch] Running {len(combinations)} combinations "
              f"({len(profiles)} profiles x {len(jds)} JDs x {len(SENIORITIES)} seniorities)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (cv, jd, seniority) in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] {cv.stem} x {jd.stem} x {seniority}")
        r = run_one(cv, jd, seniority, OUT_DIR)
        results.append(r)
        time.sleep(0.5)  # rate-limit buffer

    # Summary
    ok      = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors  = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'='*50}")
    print(f"Done: {ok} generated, {skipped} skipped, {errors} errors")

    summary_path = OUT_DIR / "_batch_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
