import re
import sys
from collections import defaultdict
from pathlib import Path


def count_file(filepath):
    text = Path(filepath).read_text(errors="replace")
    rounds      = len(re.findall(r"=== ROUND \d+ START", text))
    guesses     = len(re.findall(r"^Guesser: (?:ACTION: )?GUESS:", text, re.MULTILINE))
    questions   = len(re.findall(r"^Guesser: (?:ACTION: )?QUESTION:", text, re.MULTILINE))
    web_searches = len(re.findall(r"^Guesser: (?:ACTION: )?WEB_SEARCH:", text, re.MULTILINE))
    hints       = len(re.findall(r"^Guesser: (?:ACTION: )?USE_HINT", text, re.MULTILINE))
    return {
        "rounds": rounds,
        "guesses": guesses,
        "questions": questions,
        "web_searches": web_searches,
        "hints": hints,
        "total_turns": guesses + questions,
    }


def aggregate(stats_list):
    total = defaultdict(int)
    for s in stats_list:
        for k, v in s.items():
            if k != "_files":
                total[k] += v
    return dict(total)


def per_round(agg):
    r = agg["rounds"]
    if r == 0:
        return {}
    return {k: v / r for k, v in agg.items() if k not in ("rounds", "_files")}


def extract_guesser(folder_name):
    m = re.search(r"_g([^_]+)_s", folder_name)
    return m.group(1) if m else folder_name


def print_agg(agg, label, indent="  "):
    pr = per_round(agg)
    print(f"{label}  ({agg['rounds']} rounds across {len(agg.get('_files', [0]))} seeds)")
    print(f"{indent}guesses/round:      {pr.get('guesses', 0):.2f}  (total {agg['guesses']})")
    print(f"{indent}questions/round:    {pr.get('questions', 0):.2f}  (total {agg['questions']})")
    print(f"{indent}web searches/round: {pr.get('web_searches', 0):.2f}  (total {agg['web_searches']})")
    print(f"{indent}hints/round:        {pr.get('hints', 0):.2f}  (total {agg['hints']})")
    print(f"{indent}total turns/round:  {pr.get('total_turns', 0):.2f}  (total {agg['total_turns']})")


def process_folder(folder_path):
    folder = Path(folder_path)
    out_files = sorted(folder.glob("*.out"))
    if not out_files:
        print(f"  [warning] no .out files in {folder}", file=sys.stderr)
        return None
    file_stats = [count_file(f) for f in out_files]
    agg = aggregate(file_stats)
    agg["_files"] = out_files
    return agg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_guesses.py <folder1> [folder2 ...]")
        sys.exit(1)

    by_guesser = defaultdict(list)
    folder_results = []

    for folder_path in sys.argv[1:]:
        agg = process_folder(folder_path)
        if agg is None:
            continue
        folder_name = Path(folder_path).name
        guesser = extract_guesser(folder_name)
        folder_results.append((folder_name, guesser, agg))
        by_guesser[guesser].append(agg)

    if not folder_results:
        print("No data found.")
        sys.exit(1)

    print("=== PER FOLDER ===")
    for folder_name, guesser, agg in folder_results:
        print_agg(agg, f"\n[{folder_name}]")

    if len(by_guesser) > 1 or any(len(v) > 1 for v in by_guesser.values()):
        print("\n=== PER GUESSER (avg across folders) ===")
        for guesser, agg_list in sorted(by_guesser.items()):
            combined = aggregate(agg_list)
            combined["_files"] = [f for a in agg_list for f in a.get("_files", [])]
            print_agg(combined, f"\nguesser={guesser}")

    if len(folder_results) > 1:
        all_agg = aggregate([a for _, _, a in folder_results])
        all_agg["_files"] = [f for _, _, a in folder_results for f in a.get("_files", [])]
        print("\n=== GRAND TOTAL ===")
        print_agg(all_agg, "\nAll folders combined")
