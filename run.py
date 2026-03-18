"""
run.py
──────
Single entry point for all game modes.

Configure the run in config.py, then execute:
    python3 run.py

The MODE flag in config.py selects which secrets and game class to use:
    "standard" — 9 harder secrets  (golden retriever, dove, python, ...)
    "easy"     — 9 simpler secrets (dog, cat, elephant, ...)
    "hint"     — 9 secrets with USE_HINT mechanic
"""

from __future__ import annotations

import os

from carbontracker.tracker import CarbonTracker

import config
from evaluation import evaluate_game, load_judge_model, summarise_results
from models import load_model


def main() -> None:
    """Load models, run all rounds for the configured mode, print evaluation summary."""

    # ── Select mode ──────────────────────────────────────────────────────────
    if config.MODE == "standard":
        from word_bank.standard import SECRETS
        from game.base import BaseGame as GameClass
        from prompts import GUESSER_SYSTEM_PROMPT as guesser_prompt

    elif config.MODE == "easy":
        from word_bank.easy import SECRETS
        from game.base import BaseGame as GameClass
        from prompts import GUESSER_SYSTEM_PROMPT as guesser_prompt

    elif config.MODE == "hint":
        from word_bank.hint import SECRETS
        from game.hint import HintGame as GameClass
        from prompts import HINT_GUESSER_SYSTEM_PROMPT as guesser_prompt

    else:
        raise ValueError(f"Unknown MODE '{config.MODE}'. Choose: standard | easy | hint")

    # ── Load models ──────────────────────────────────────────────────────────
    guesser_model, guesser_tokenizer = load_model(config.GUESSER_MODEL)
    secret_model,  secret_tokenizer  = load_model(config.SECRET_MODEL)
    load_judge_model(config.JUDGE_MODEL)

    # ── Carbon tracking ──────────────────────────────────────────────────────
    os.makedirs(config.CARBON_LOG_DIR, exist_ok=True)
    tracker = CarbonTracker(
        epochs=1,
        monitor_epochs=True,
        log_dir=config.CARBON_LOG_DIR,
        verbose=2,
    )
    tracker.epoch_start()

    # ── Run rounds ───────────────────────────────────────────────────────────
    eval_results = []
    category_stats = {
        "animal": {"rounds": 0, "turns": 0, "correct": 0},
        "food":   {"rounds": 0, "turns": 0, "correct": 0},
        "object": {"rounds": 0, "turns": 0, "correct": 0},
    }

    for i, secret in enumerate(SECRETS, start=1):

        # Build the right game instance for this mode
        if config.MODE == "hint":
            game = GameClass(
                secret_prompt=secret.system_prompt,
                secret_label=secret.label,
                round_number=i,
                guesser_model=guesser_model,
                guesser_tokenizer=guesser_tokenizer,
                secret_model=secret_model,
                secret_tokenizer=secret_tokenizer,
                guesser_system_prompt=guesser_prompt,
                hints=secret.hints,
            )
        else:
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

        record = game.play()

        # Accumulate category stats
        stats = category_stats[secret.category]
        stats["rounds"] += 1
        stats["turns"]  += record.turns_used
        if record.was_correct:
            stats["correct"] += 1

        # Evaluate and print
        result = evaluate_game(
            record,
            dataset_path=config.DATASET_PATH,
            judge_model_name=config.JUDGE_MODEL,
            run_judge=config.RUN_JUDGE,
        )
        eval_results.append(result)
        print(result)

    tracker.epoch_end()
    tracker.stop()

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

    print("\n=== EVALUATION SUMMARY (all rounds) ===")
    summary = summarise_results(eval_results)
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
