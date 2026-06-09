"""
run.py
──────
Single entry point for all game modes.

Configure the run in config.py, then execute:
    python3 run.py

The MODE flag in config.py selects which secrets and game class to use:
    "standard" — no tools (golden retriever, python snake, ...)
    "tool"     — same secrets with USE_HINT and WEB_SEARCH tools enabled
"""

from __future__ import annotations

import os
import random
import shutil

import torch
from carbontracker import parser
from carbontracker.tracker import CarbonTracker

import config


def _read_latest_carbon_log(log_dir: str) -> tuple[float, float]:
    """Parse the carbontracker log in `log_dir` and return (kWh, gCO2eq).

    The caller is responsible for ensuring `log_dir` contains exactly the
    log(s) written by the most recent tracker (we wipe and recreate the dir
    before each game), so `parse_all_logs` returns a single entry.

    Returns (0.0, 0.0) if no valid log is found.
    """
    try:
        logs = parser.parse_all_logs(log_dir=log_dir)
    except Exception as exc:
        print(f"[CARBON] WARNING: parse_all_logs failed: {exc}")
        return 0.0, 0.0
    if not logs:
        print(f"[CARBON] WARNING: no logs found in {log_dir}")
        return 0.0, 0.0
    actual = logs[-1].get("actual") or {}
    return (
        float(actual.get("energy (kWh)") or 0.0),
        float(actual.get("co2eq (g)") or 0.0),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


EXPERIMENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))
_seed_everything(EXPERIMENT_SEED)
print(f"Experiment seed: {EXPERIMENT_SEED}")
from evaluation import evaluate_game, summarise_results
from models import generate_answer, load_model


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _one_line(text: str) -> str:
    return text.replace("\n", "\\n").strip()


def _postgame_secret_audit(game, secret_label: str) -> None:
    """Optionally probe the keeper after the game without touching gameplay state.

    This is intentionally opt-in and runs only after game.play() and carbon
    tracking are complete. It copies the keeper history before adding audit
    prompts, so the game transcript, record, metrics, and future prompts are
    unchanged.
    """
    if not _env_flag_enabled("POSTGAME_SECRET_AUDIT"):
        return

    true_messages = [dict(msg) for msg in game.secret_messages]
    true_messages.append({"role": "user", "content": f"My guess is: {secret_label}"})
    true_raw = generate_answer(
        true_messages,
        game.secret_model,
        game.secret_tokenizer,
        max_new_tokens=20,
    )

    decoy = "__not_the_secret__"
    decoy_messages = [dict(msg) for msg in game.secret_messages]
    decoy_messages.append({"role": "user", "content": f"My guess is: {decoy}"})
    decoy_raw = generate_answer(
        decoy_messages,
        game.secret_model,
        game.secret_tokenizer,
        max_new_tokens=20,
    )

    print(f"[POSTGAME SECRET AUDIT] Round {game.round_number} ({secret_label})")
    print(
        f"[POSTGAME SECRET AUDIT] true_guess={secret_label!r} "
        f"raw={_one_line(true_raw)!r}"
    )
    print(
        f"[POSTGAME SECRET AUDIT] decoy_guess={decoy!r} "
        f"raw={_one_line(decoy_raw)!r}"
    )


def main() -> None:
    """Load models, run all rounds for the configured mode, print evaluation summary."""

    # ── Select mode ──────────────────────────────────────────────────────────
    if config.MODE == "standard":
        from word_bank.standard import SECRETS
        from game.base import BaseGame as GameClass
        from prompts import GUESSER_SYSTEM_PROMPT as guesser_prompt

    elif config.MODE == "tool":
        from word_bank.standard import SECRETS
        from game.tool import ToolGame as GameClass
        from prompts import TOOL_GUESSER_SYSTEM_PROMPT as guesser_prompt

    else:
        raise ValueError(f"Unknown MODE '{config.MODE}'. Choose: standard | tool")

    # ── Load models ──────────────────────────────────────────────────────────
    guesser_model, guesser_tokenizer = load_model(config.GUESSER_MODEL)
    secret_model,  secret_tokenizer  = load_model(config.SECRET_MODEL)

    # ── Carbon tracking ──────────────────────────────────────────────────────
    # One tracker per game, one epoch per game. The epoch wraps the whole
    # game.play() call so both guesser and secret keeper inference (and tool
    # calls in tool mode) are measured together. Per-turn bracketing was
    # dropped because individual generate_answer calls are often shorter than
    # the power sampling interval, which made small-model measurements noisy.
    # Offline mode: no API keys → carbontracker falls back to the DK national
    # average carbon intensity (no live measurement needed).
    os.makedirs(config.CARBON_LOG_DIR, exist_ok=True)

    records = []
    category_stats = {}
    per_game_carbon: list[dict] = []

    for i, secret in enumerate(SECRETS, start=1):
        game_log_dir = os.path.join(config.CARBON_LOG_DIR, f"game_{i:03d}")
        # Wipe before each game so parse_all_logs sees only this run's log.
        # Without this, stale logs from earlier runs in the same directory
        # were picked up and produced incorrect per-round energy numbers.
        if os.path.exists(game_log_dir):
            shutil.rmtree(game_log_dir)
        os.makedirs(game_log_dir, exist_ok=True)
        tracker = CarbonTracker(
            epochs=1,               # one epoch = one full game
            epochs_before_pred=0,   # disable prediction (we want actuals only)
            monitor_epochs=-1,      # monitor every epoch
            update_interval=1,      # sample every second
            log_dir=game_log_dir,
            verbose=1,
        )

        game = GameClass(
            secret_prompt=secret.system_prompt,
            secret_label=secret.label,
            round_number=i,
            guesser_model=guesser_model,
            guesser_tokenizer=guesser_tokenizer,
            secret_model=secret_model,
            secret_tokenizer=secret_tokenizer,
            guesser_system_prompt=guesser_prompt,
        )

        tracker.epoch_start()
        record = game.play()
        tracker.epoch_end()
        tracker.stop()  # flushes the log file we then parse
        _postgame_secret_audit(game, secret.label)

        # Parse the carbontracker log this game just wrote. The dir was wiped
        # at the top of this iteration so exactly one log lives in it.
        game_kwh, game_co2 = _read_latest_carbon_log(game_log_dir)
        per_game_carbon.append({
            "round": i,
            "secret": secret.label,
            "energy_kwh": game_kwh,
            "co2_g": game_co2,
        })
        print(
            f"[CARBON] Round {i} ({secret.label}): "
            f"{game_kwh:.6f} kWh, {game_co2:.4f} g CO2eq"
        )

        records.append((record, secret))

        stats = category_stats.setdefault(
            secret.category,
            {"rounds": 0, "turns": 0, "correct": 0},
        )
        stats["rounds"] += 1
        stats["turns"]  += record.turns_used
        if record.was_correct:
            stats["correct"] += 1

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_results = []
    for record, secret in records:
        result = evaluate_game(record)
        eval_results.append(result)
        print(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_rounds  = len(SECRETS)
    total_turns   = sum(s["turns"] for s in category_stats.values())
    total_correct = sum(s["correct"] for s in category_stats.values())

    print("\n=== SUMMARY ===")
    print(f"Mode           : {config.MODE}")
    print(f"Guesser model  : {config.GUESSER_MODEL}")
    print(f"Secret model   : {config.SECRET_MODEL}")
    print(f"Total rounds   : {total_rounds}")
    print(f"Total correct  : {total_correct}")
    print(f"Avg turns      : {total_turns / total_rounds:.2f}")

    for cat, stats in category_stats.items():
        if stats["rounds"] > 0:
            print(f"\nCategory: {cat}s")
            print(f"  Rounds  : {stats['rounds']}")
            print(f"  Correct : {stats['correct']}")
            print(f"  Avg turns: {stats['turns'] / stats['rounds']:.2f}")

    # ── Carbon summary (whole game: guesser + secret keeper) ─────────────────
    total_kwh = sum(g["energy_kwh"] for g in per_game_carbon)
    total_co2 = sum(g["co2_g"]      for g in per_game_carbon)
    n_games   = max(len(per_game_carbon), 1)
    print("\n=== CARBON SUMMARY ===")
    print(f"  Scope            : whole game (guesser + secret keeper + tool calls)")
    print(f"  Games measured   : {len(per_game_carbon)}")
    # Format matches the regexes in gather_results.py (Actual consumption / Energy / CO2eq).
    print("Actual consumption:")
    print(f"  Energy: {total_kwh:.12f} kWh")
    print(f"  CO2eq:  {total_co2:.12f} g")
    print(f"  Energy per game  : {total_kwh / n_games * 1000:.4f} Wh")
    print(f"  CO2eq per game   : {total_co2 / n_games:.4f} g")
    print("  Per-game breakdown:")
    for g in per_game_carbon:
        print(
            f"    Round {g['round']:>3} | {g['secret']:<30} | "
            f"{g['energy_kwh']:.6f} kWh | {g['co2_g']:.4f} g CO2eq"
        )

    print("\n=== EVALUATION SUMMARY (all rounds) ===")
    summary = summarise_results(eval_results)
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nguesser_model_name: {config.GUESSER_MODEL}")
    print(f"secret_model_name:  {config.SECRET_MODEL}")
    print(f"experiment_seed:    {EXPERIMENT_SEED}")


if __name__ == "__main__":
    main()
