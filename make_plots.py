"""Generate publication-ready plots for the two summary tables."""
import csv
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results/results.csv"
OUT = "results/plots"

# Broken guesser to exclude from the guesser-side plot (emits 0 tool calls,
# never produces a valid final guess — plumbing failure, not a real data point).
BROKEN_GUESSERS = {"Qwen3.5-2B"}

# Keeper display order (large → small), guesser display order (best → worst raw win%).
KEEPER_ORDER  = ["Qwen3-8B", "Qwen3.5-4B", "Qwen3.5-2B", "Qwen3-1.7B"]
GUESSER_ORDER = ["Qwen3-1.7B", "Qwen3.5-4B", "Qwen3-8B"]


def short(name: str) -> str:
    return name.split("/")[-1]


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def aggregate_by(rows, key_field):
    groups = defaultdict(list)
    for r in rows:
        groups[short(r[key_field])].append(r)
    out = {}
    for k, rs in groups.items():
        ng = sum(int(r["num_games"]) for r in rs)
        nw = sum(int(r["num_wins"]) for r in rs)
        nv = sum(int(r["num_verified_wins"]) for r in rs)
        nfc = sum(int(r["num_false_correct"]) for r in rs)
        def mean(col):
            vals = [float(r[col]) for r in rs if r[col] not in ("", None)]
            return statistics.mean(vals) if vals else float("nan")
        out[k] = {
            "n_games": ng,
            "raw":     nw / ng * 100 if ng else 0.0,
            "verified": nv / ng * 100 if ng else 0.0,
            "fc":      nfc,
            "fc_pct":  nfc / ng * 100 if ng else 0.0,
            "survival": min(100.0, nv / nw * 100) if nw else 100.0,
            "tools":   mean("avg_tool_calls_used"),
            "eff":     mean("avg_efficiency_score"),
        }
    return out


rows = load_rows()
by_keeper  = aggregate_by(rows, "secret_model")
by_guesser = aggregate_by(
    [r for r in rows if short(r["guesser_model"]) not in BROKEN_GUESSERS],
    "guesser_model",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ------------------------------------------------------------
# PLOT 1: Keeper honesty — raw win% vs verified% + false-corrects
# ------------------------------------------------------------
keepers  = KEEPER_ORDER
raw      = [by_keeper[k]["raw"]      for k in keepers]
verified = [by_keeper[k]["verified"] for k in keepers]
fc       = [by_keeper[k]["fc"]       for k in keepers]
survival = [by_keeper[k]["survival"] for k in keepers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(keepers))
w = 0.38
b1 = ax1.bar(x - w/2, raw,      w, label="Raw win %",     color="#d9534f", edgecolor="black", linewidth=0.6)
b2 = ax1.bar(x + w/2, verified, w, label="Verified win %", color="#5cb85c", edgecolor="black", linewidth=0.6)
ax1.set_xticks(x); ax1.set_xticklabels(keepers, rotation=15)
ax1.set_ylabel("Win rate (%)")
ax1.set_title("Raw vs. verified wins against each keeper")
ax1.set_ylim(0, 75)
ax1.legend(loc="upper left", frameon=False)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
for bars in (b1, b2):
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                 f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

# Right: false-correct rate (per 360 games) as bar + survival as line
fc_pct = [f / 360 * 100 for f in fc]
b3 = ax2.bar(keepers, fc_pct, color="#f0ad4e", edgecolor="black", linewidth=0.6, label="False-correct rate")
ax2.set_ylabel("False-correct rate (%)", color="#b37419")
ax2.tick_params(axis="y", labelcolor="#b37419")
ax2.set_ylim(0, 16)
for b, n in zip(b3, fc):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.25,
             f"n={n}", ha="center", va="bottom", fontsize=9, color="#7a4e0f")

ax2b = ax2.twinx()
ax2b.plot(keepers, survival, "o-", color="#337ab7", linewidth=2, markersize=7, label="Verified-win survival %")
ax2b.set_ylabel("Wins surviving verification (%)", color="#1f4f7a")
ax2b.tick_params(axis="y", labelcolor="#1f4f7a")
ax2b.set_ylim(0, 105)
for k, s in zip(keepers, survival):
    ax2b.text(k, s + 3, f"{s:.0f}%", ha="center", va="bottom", fontsize=9, color="#1f4f7a")

ax2.set_title("Keeper honesty: false-corrects and win survival")
ax2.set_xticks(x); ax2.set_xticklabels(keepers, rotation=15)
ax2.grid(axis="y", linestyle=":", alpha=0.5)

n_per_keeper = by_keeper[keepers[0]]["n_games"]
fig.suptitle(f"Keeper effect on scoring ({n_per_keeper} games per keeper)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/keeper_honesty.png")
plt.savefig(f"{OUT}/keeper_honesty.pdf")
plt.close()


# ------------------------------------------------------------
# PLOT 2: Guesser performance — win%, verified%, tools, efficiency
# ------------------------------------------------------------
guessers   = GUESSER_ORDER
g_win      = [by_guesser[g]["raw"]      for g in guessers]
g_verified = [by_guesser[g]["verified"] for g in guessers]
g_tools    = [by_guesser[g]["tools"]    for g in guessers]
g_eff      = [by_guesser[g]["eff"]      for g in guessers]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(guessers))
w = 0.38
bL1 = axL.bar(x - w/2, g_win,      w, label="Raw win %",      color="#d9534f", edgecolor="black", linewidth=0.6)
bL2 = axL.bar(x + w/2, g_verified, w, label="Verified win %", color="#5cb85c", edgecolor="black", linewidth=0.6)
axL.set_xticks(x); axL.set_xticklabels(guessers)
axL.set_ylabel("Win rate (%)")
n_working_keepers = sum(1 for k in KEEPER_ORDER if k not in BROKEN_GUESSERS) or len(KEEPER_ORDER)
axL.set_title(f"Guesser win rates (avg over {n_working_keepers} keepers)")
axL.set_ylim(0, 90)
axL.legend(loc="upper right", frameon=False)
axL.grid(axis="y", linestyle=":", alpha=0.5)
for bars in (bL1, bL2):
    for b in bars:
        axL.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                 f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

# Right: tool usage (bar) + efficiency (line)
bR = axR.bar(guessers, g_tools, color="#5bc0de", edgecolor="black", linewidth=0.6, label="Tool calls / game")
axR.set_ylabel("Tool calls per game", color="#1b6177")
axR.tick_params(axis="y", labelcolor="#1b6177")
axR.set_ylim(0, 7)
for b, t in zip(bR, g_tools):
    axR.text(b.get_x() + b.get_width()/2, b.get_height() + 0.12,
             f"{t:.2f}", ha="center", va="bottom", fontsize=9, color="#1b6177")

axR2 = axR.twinx()
axR2.plot(guessers, g_eff, "s-", color="#9467bd", linewidth=2, markersize=7, label="Efficiency")
axR2.set_ylabel("Efficiency score", color="#5a3378")
axR2.tick_params(axis="y", labelcolor="#5a3378")
axR2.set_ylim(0, 0.8)
for g, e in zip(guessers, g_eff):
    axR2.text(g, e + 0.03, f"{e:.3f}", ha="center", va="bottom", fontsize=9, color="#5a3378")

axR.set_title("Tool usage and efficiency by guesser")
axR.grid(axis="y", linestyle=":", alpha=0.5)

fig.suptitle("Guesser performance summary", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/guesser_summary.png")
plt.savefig(f"{OUT}/guesser_summary.pdf")
plt.close()

print(f"Saved 4 files to {OUT}/:")
print("  keeper_honesty.png / .pdf")
print("  guesser_summary.png / .pdf")
