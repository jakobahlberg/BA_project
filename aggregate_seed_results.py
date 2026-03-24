#!/usr/bin/env python3
import argparse
import glob
import os
import re
import statistics
from typing import Dict, List

METRICS = [
    "Total correct",
    "Overall average turns per round",
    "avg_win_score",
    "avg_efficiency_score",
    "avg_secret_reliability_score",
    "avg_layer1_score",
    "avg_semantic_relevance_score",
    "avg_semantic_diversity_score",
    "avg_information_gain_score",
    "avg_layer2_score",
    "avg_llm_judge_strategy",
    "avg_llm_judge_question_quality",
    "avg_llm_judge_logical_consistency",
    "avg_llm_judge_efficiency",
    "avg_layer3_score",
    "avg_overall_score",
    "num_games",
    "num_wins",
]


def parse_file(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {"file": path}
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    seed_match = re.search(r"experiment_seed:\s*(\d+)", txt)
    if seed_match:
        out["seed"] = float(seed_match.group(1))
    else:
        name_seed = re.search(r"slurm-seed(\d+)-", os.path.basename(path))
        if name_seed:
            out["seed"] = float(name_seed.group(1))

    for metric in METRICS:
        m = re.search(rf"{re.escape(metric)}:\s*([0-9]+(?:\.[0-9]+)?)", txt)
        if m:
            out[metric] = float(m.group(1))

    return out


def mean_std(vals: List[float]):
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (statistics.mean(vals), statistics.pstdev(vals))


def main():
    ap = argparse.ArgumentParser(description="Aggregate seeded slurm outputs")
    ap.add_argument("pattern", nargs="?", default="slurm-seed*.out")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched: {args.pattern}")
        return

    runs = [parse_file(f) for f in files]

    print("Per-run summary:")
    for r in runs:
        seed = int(r["seed"]) if "seed" in r else "?"
        wins = int(r.get("num_wins", -1))
        games = int(r.get("num_games", -1))
        overall = r.get("avg_overall_score", float("nan"))
        print(f"  seed={seed:>3}  wins={wins}/{games}  avg_overall={overall:.3f}  file={r['file']}")

    print("\nAggregate (mean +/- std):")
    keys = ["num_wins", "Total correct", "avg_overall_score", "avg_layer1_score", "avg_layer2_score", "avg_layer3_score", "avg_efficiency_score"]
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        m, s = mean_std(vals)
        if vals:
            print(f"  {k}: {m:.3f} +/- {s:.3f}")


if __name__ == "__main__":
    main()