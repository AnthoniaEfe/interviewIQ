"""
inspect.py
==========
Print generated baseline question sets for manual review.

Usage:
    python inspect.py                          # list all results
    python inspect.py --file results/baseline/p01_swe_junior.json
    python inspect.py --all                    # print all question sets
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results/baseline")


def print_set(path: Path):
    data = json.loads(path.read_text())
    print(f"\n{'='*60}")
    print(f"Profile:   {data['profile_id']}")
    print(f"JD:        {data['jd_id']}")
    print(f"Seniority: {data['seniority']}")
    print(f"Model:     {data['model']}  temp={data['temperature']}  seed={data['seed']}")
    print(f"Generated: {data['generated_at']}")
    print(f"Questions: {len(data['questions'])}")
    print()
    for q in data["questions"]:
        print(f"  {q['id']:>2}. {q['question']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Inspect a single result file")
    parser.add_argument("--all",  action="store_true", help="Print all result files")
    args = parser.parse_args()

    if args.file:
        print_set(Path(args.file))
        return

    files = sorted(f for f in RESULTS_DIR.glob("*.json")
                   if not f.name.startswith("_"))

    if not files:
        print(f"No results found in {RESULTS_DIR}")
        return

    if args.all:
        for f in files:
            print_set(f)
        return

    # Default: just list
    print(f"\nBaseline results in {RESULTS_DIR}:")
    print(f"{'File':<50} {'Questions':>9}")
    print("-" * 62)
    for f in files:
        data = json.loads(f.read_text())
        n = len(data["questions"])
        flag = "  ⚠" if n != 15 else ""
        print(f"  {f.name:<48} {n:>9}{flag}")
    print(f"\nTotal: {len(files)} files  |  use --all to print questions")


if __name__ == "__main__":
    main()
