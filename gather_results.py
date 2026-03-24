import csv
import os
import re
import statistics
from collections import defaultdict

ROOT      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
CSV_PATH  = os.path.join(RESULTS_DIR, "results.csv")

SCORE_FIELDS = [
    "avg_win_score", "avg_efficiency_score", "avg_secret_reliability_score",
    "avg_semantic_relevance_score", "avg_canonical_coverage_score", "avg_information_gain_score",
    "avg_llm_judge_strategy", "avg_llm_judge_question_quality",
    "avg_llm_judge_logical_consistency", "avg_llm_judge_secret_accuracy",
    "avg_llm_judge_guesser_format", "avg_layer3_score", "judge_layer1_agreement",
    "avg_hints_used", "avg_web_searches_used", "avg_tool_calls_used",
]

CSV_COLUMNS = (
    ["run_id", "seed", "mode", "guesser_model", "secret_model", "num_games", "num_wins",
     "judge_win_count"]
    + SCORE_FIELDS
    + ["avg_diversity", "energy_kwh", "energy_per_game_wh", "co2_g", "co2_per_game_g", "adjusted_co2_per_game_g"]
)

CARBON_HEADER = "=== CARBON SUMMARY ==="


def parse_out(path: str) -> dict | None:
    with open(path) as f:
        text = f.read()

    # Must have an evaluation summary to be a valid result file
    if "=== EVALUATION SUMMARY" not in text:
        return None

    def get(pattern, cast=str, flags=0):
        m = re.search(pattern, text, flags)
        return cast(m.group(1).strip()) if m else None

    energy_kwh = get(r"Actual consumption.*?Energy:\s*([\d.]+)\s*kWh", float, re.DOTALL)
    co2_g      = get(r"Actual consumption.*?CO2eq:\s*([\d.]+)\s*g",    float, re.DOTALL)
    num_games  = get(r"num_games\s*:\s*(\d+)", int)

    seed = get(r"experiment_seed:\s*(\d+)", int)
    if seed is None:
        m = re.search(r"slurm-seed(\d+)-", os.path.basename(path))
        seed = int(m.group(1)) if m else None

    row = {
        "run_id":        os.path.basename(path).replace(".out", ""),
        "seed":          seed,
        "mode":          get(r"Mode\s*:\s*(\S+)"),
        "guesser_model": get(r"Guesser model\s*:\s*(.+)"),
        "secret_model":  get(r"Secret model\s*:\s*(.+)"),
        "num_games":     num_games,
        "num_wins":         get(r"num_wins\s*:\s*(\d+)", int),
        "judge_win_count":  get(r"judge_win_count\s*:\s*(\d+)", int),
        "energy_kwh":    energy_kwh,
        "energy_per_game_wh": round(energy_kwh / num_games * 1000, 6) if energy_kwh and num_games else None,
        "co2_g":         co2_g,
        "co2_per_game_g": round(co2_g / num_games, 6) if co2_g and num_games else None,
    }
    for f in SCORE_FIELDS:
        row[f] = get(rf"{f}\s*:\s*([\d.]+)", float)

    # adjusted_co2: reliability * co2_per_game — if reliability=1 carbon is unchanged,
    # if reliability<1 carbon is scaled down proportionally.
    rel = row.get("avg_llm_judge_secret_accuracy")
    co2_pg = row.get("co2_per_game_g")
    row["adjusted_co2_per_game_g"] = round(rel * co2_pg, 6) if (rel is not None and co2_pg is not None and rel > 0) else None

    # avg diversity across all games, excluding Q1 (always 1.0)
    all_scores = []
    for line in re.findall(r"Q1:[\d.]+(.+)", text):
        all_scores += [float(v) for v in re.findall(r"Q\d+:([\d.]+)", line)]
    row["avg_diversity"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else None

    return row


def append_carbon_summary(path: str, row: dict) -> None:
    with open(path) as f:
        text = f.read()
    if CARBON_HEADER in text and "Total energy" in text:
        return  # already appended

    lines = [f"\n{CARBON_HEADER}"]
    if row["energy_kwh"] is not None:
        lines.append(f"  Total energy     : {row['energy_kwh']:.6f} kWh")
    if row["energy_per_game_wh"] is not None:
        lines.append(f"  Energy per game  : {row['energy_per_game_wh']:.4f} Wh")
    if row["co2_g"] is not None:
        lines.append(f"  Total CO2eq      : {row['co2_g']:.4f} g")
    if row["co2_per_game_g"] is not None:
        lines.append(f"  CO2eq per game   : {row['co2_per_game_g']:.4f} g")

    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Scan root, outputs/, and any runs_*/ subdirectories
    scan_dirs = [ROOT, os.path.join(ROOT, "outputs")]
    scan_dirs += sorted(
        os.path.join(ROOT, d) for d in os.listdir(ROOT)
        if re.match(r"runs_", d) and os.path.isdir(os.path.join(ROOT, d))
    )
    out_files = sorted(
        os.path.join(d, f)
        for d in scan_dirs if os.path.isdir(d)
        for f in os.listdir(d)
        if re.match(r"slurm-.+\.out", f)
    )

    rows = []
    for path in out_files:
        filename = os.path.basename(path)
        row = parse_out(path)
        if row is None:
            print(f"  skip  {filename}  (no evaluation summary)")
            continue
        append_carbon_summary(path, row)
        rows.append(row)
        print(f"  ok    {filename}  ({row['mode']}, {row['num_games']} games)")

    if not rows:
        print("No valid result files found.")
        return

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} run(s) → {CSV_PATH}")

    _print_aggregate_summary(rows)


_AGGREGATE_METRICS = [
    # Win / outcome
    "avg_win_score", "avg_verified_win_score",
    # Layer 1
    "avg_efficiency_score", "avg_secret_reliability_score",
    # Layer 2
    "avg_semantic_relevance_score", "avg_canonical_coverage_score",
    "avg_information_gain_score",
    # Layer 3
    "avg_llm_judge_strategy", "avg_llm_judge_question_quality",
    "avg_llm_judge_logical_consistency", "avg_llm_judge_secret_accuracy",
    "avg_llm_judge_guesser_format", "avg_layer3_score",
    # Agreement
    "judge_layer1_agreement",
    # Tool usage (tool mode only; 0 for standard)
    "avg_hints_used", "avg_web_searches_used", "avg_tool_calls_used",
    # Energy (per-game so comparable across run sizes)
    "energy_per_game_wh", "co2_per_game_g", "adjusted_co2_per_game_g",
]


def _mean_std(vals: list) -> tuple:
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def _print_aggregate_summary(rows: list) -> None:
    if not rows:
        return

    # Derive avg_verified_win_score (judge wins / total games) for each row
    for r in rows:
        n = r.get("num_games")
        j = r.get("judge_win_count")
        r["avg_verified_win_score"] = (j / n) if (n and j is not None) else None

    # Group by (mode, guesser_model, secret_model) — each group = one config run N times
    groups: dict = defaultdict(list)
    for r in rows:
        key = (r.get("mode", "?"), r.get("guesser_model", "?"), r.get("secret_model", "?"))
        groups[key].append(r)

    print("\n=== AGGREGATE SUMMARY (mean ± std across seeds) ===")
    for (mode, guesser, secret), group_rows in sorted(groups.items()):
        guesser_label = guesser.split("/")[-1]
        secret_label  = secret.split("/")[-1]
        n_seeds = len(group_rows)
        print(f"\n  Config: {mode} | guesser={guesser_label} | secret={secret_label}  ({n_seeds} seed(s))")
        for metric in _AGGREGATE_METRICS:
            vals = [r.get(metric) for r in group_rows]
            m, s = _mean_std(vals)
            if m == m:  # skip NaN
                print(f"    {metric}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
