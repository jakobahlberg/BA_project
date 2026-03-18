import csv
import os
import re

ROOT      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
CSV_PATH  = os.path.join(RESULTS_DIR, "results.csv")

SCORE_FIELDS = [
    "avg_win_score", "avg_efficiency_score", "avg_secret_reliability_score",
    "avg_semantic_relevance_score", "avg_canonical_coverage_score", "avg_information_gain_score",
    "avg_llm_judge_strategy", "avg_llm_judge_question_quality",
    "avg_llm_judge_logical_consistency", "avg_llm_judge_secret_accuracy",
]

CSV_COLUMNS = (
    ["run_id", "seed", "mode", "guesser_model", "secret_model", "num_games", "num_wins"]
    + SCORE_FIELDS
    + ["avg_diversity", "energy_kwh", "energy_per_game_wh", "co2_g", "co2_per_game_g"]
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
        "num_wins":      get(r"num_wins\s*:\s*(\d+)", int),
        "energy_kwh":    energy_kwh,
        "energy_per_game_wh": round(energy_kwh / num_games * 1000, 6) if energy_kwh and num_games else None,
        "co2_g":         co2_g,
        "co2_per_game_g": round(co2_g / num_games, 6) if co2_g and num_games else None,
    }
    for f in SCORE_FIELDS:
        row[f] = get(rf"{f}\s*:\s*([\d.]+)", float)

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

    scan_dirs = [ROOT, os.path.join(ROOT, "outputs")]
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


if __name__ == "__main__":
    main()
