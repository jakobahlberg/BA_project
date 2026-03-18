"""
game/base.py
────────────
BaseGame: the core 20 Questions loop shared by all game modes.

To add a new game variant:
  1. Create a new file in game/ (e.g. game/tool_use.py)
  2. Subclass BaseGame
  3. Override only the methods that differ:
       - _parse_action()    to recognise new action keywords
       - _pre_turn_hook()   to inject logic before each turn (e.g. forced hints)
     Everything else (loop control, Q/A handling, GameRecord building) is inherited.
"""

from __future__ import annotations

import config
from evaluation.records import GameRecord
from models import generate_answer


class BaseGame:
    """
    Core 20 Questions game loop.

    Handles QUESTION and GUESS actions. Builds a GameRecord on completion.
    Subclasses extend _parse_action() or override _pre_turn_hook() to add
    new behaviour without touching the main loop.
    """

    def __init__(
        self,
        secret_prompt: str,
        secret_label: str,
        round_number: int,
        guesser_model,
        guesser_tokenizer,
        secret_model,
        secret_tokenizer,
        guesser_system_prompt: str,
    ) -> None:
        """
        Args:
            secret_prompt:         System prompt for the secret keeper model.
            secret_label:          Human-readable label (must match dataset.json).
            round_number:          Round index used for console output only.
            guesser_model:         Loaded guesser LLM.
            guesser_tokenizer:     Matching tokenizer for guesser.
            secret_model:          Loaded secret keeper LLM.
            secret_tokenizer:      Matching tokenizer for secret keeper.
            guesser_system_prompt: System prompt for the guesser model.
        """
        self.secret_prompt        = secret_prompt
        self.secret_label         = secret_label
        self.round_number         = round_number
        self.guesser_model        = guesser_model
        self.guesser_tokenizer    = guesser_tokenizer
        self.secret_model         = secret_model
        self.secret_tokenizer     = secret_tokenizer
        self.guesser_system_prompt = guesser_system_prompt

        # ── Game state ───────────────────────────────────────────────────
        self.questions:             list[str]   = []
        self.answers:               list[str]   = []
        self.guesses:               list[str]   = []
        self.secret_raw_responses:  list[str]   = []
        self.turn_log:              list[tuple] = []
        self.final_guess:           str         = ""
        self.game_over:             bool        = False
        self.turn:                  int         = 0

        # ── Message histories ────────────────────────────────────────────
        self.guesser_messages: list[dict] = [
            {"role": "system", "content": self.guesser_system_prompt},
            {"role": "user",   "content": "Start the game."},
        ]
        self.secret_messages: list[dict] = [
            {"role": "system", "content": self.secret_prompt},
            {"role": "user",   "content": "Awaiting first question."},
        ]

    # ── Action parsing ───────────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str, str | None]:
        """
        Parse a single line of guesser output into (action, content).

        Returns:
            action:  "question" | "guess" | "unknown"
            content: The text after the prefix, or the raw text for "unknown".

        Subclasses should call super()._parse_action() first and extend with
        additional action keywords (e.g. "hint").
        """
        if text.startswith("QUESTION:"):
            return "question", text.replace("QUESTION:", "").strip()
        if text.startswith("GUESS:"):
            return "guess", text.replace("GUESS:", "").strip()
        return "unknown", text

    # ── Turn handlers ────────────────────────────────────────────────────────

    def _handle_question(self, question: str) -> str:
        """
        Send a yes/no question to the secret keeper and record the result.

        Updates questions, answers, secret_raw_responses, turn_log, and
        guesser_messages in-place.

        Args:
            question: The yes/no question string.

        Returns:
            Normalised answer: "YES" or "NO".
        """
        self.secret_messages.append({"role": "user", "content": question})
        raw = generate_answer(self.secret_messages, self.secret_model, self.secret_tokenizer)
        self.secret_raw_responses.append(raw)

        normalised = "YES" if "YES" in raw.upper() else "NO"
        self.questions.append(question)
        self.answers.append(normalised)
        self.turn_log.append(("question", question, raw))

        self.guesser_messages.append({
            "role": "user",
            "content": (
                f"Turn {self.turn} result:\n"
                f"Your question: {question}\n"
                f"Secret answered: {raw}"
            ),
        })
        return normalised

    def _handle_guess(self, guess: str) -> bool:
        """
        Submit a guess to the secret keeper and record the result.

        Updates guesses, secret_raw_responses, turn_log, and (on failure)
        guesser_messages in-place.

        Args:
            guess: The specific guess string.

        Returns:
            True if the guess was correct (game won), False otherwise.
        """
        self.guesses.append(guess)
        self.final_guess = guess

        self.secret_messages.append({"role": "user", "content": f"My guess is: {guess}"})
        result = generate_answer(self.secret_messages, self.secret_model, self.secret_tokenizer)
        self.secret_raw_responses.append(result)
        self.turn_log.append(("guess", guess, result))

        if "CORRECT" in result.strip().upper():
            return True

        self.guesser_messages.append({
            "role": "user",
            "content": (
                f"Turn {self.turn} result:\n"
                f"Your guess: {guess}\n"
                f"Secret response: WRONG\n\n"
                "This guess was incorrect. Do not repeat it.\n"
                "Continue reasoning and ask a new question."
            ),
        })
        return False

    # ── Hook for subclasses ──────────────────────────────────────────────────

    def _pre_turn_hook(self) -> bool:
        """
        Called at the start of every turn, before the guesser generates output.

        Override in subclasses to inject pre-turn logic (e.g. force a hint
        after N failed guesses). Return True to skip the turn counter increment
        for this iteration (the loop will not advance self.turn).

        Returns:
            False by default (turn counter increments normally).
        """
        return False

    # ── Main loop ────────────────────────────────────────────────────────────

    def play(self) -> GameRecord:
        """
        Run the full game loop and return a completed GameRecord.

        The loop continues until MAX_TURNS is reached or the guesser wins.
        Unknown actions (malformed output) are treated as questions to keep
        the game moving rather than silently looping.

        Returns:
            GameRecord with all questions, answers, guesses, and metadata.
        """
        print(f"\n=== ROUND {self.round_number} START ===")

        while self.turn < config.MAX_TURNS and not self.game_over:

            skip_increment = self._pre_turn_hook()
            if skip_increment:
                continue

            guesser_output = generate_answer(
                self.guesser_messages, self.guesser_model, self.guesser_tokenizer
            )
            print(f"Guesser: {guesser_output}")

            action, content = self._parse_action(guesser_output)

            if action == "question":
                answer = self._handle_question(content)
                print(f"Secret: {answer}")

            elif action == "guess":
                correct = self._handle_guess(content)
                print(f"Secret: {'CORRECT' if correct else 'WRONG'}")
                if correct:
                    print("Guesser won!")
                    self.game_over = True

            else:
                # Malformed output — treat as a question to keep the game moving
                answer = self._handle_question(content)
                print(f"Secret: {answer}")

            self.turn += 1

        if not self.game_over:
            print(f"Guesser failed after {config.MAX_TURNS} turns")
            if self.guesses:
                self.final_guess = self.guesses[-1]

        return GameRecord(
            secret=self.secret_label,
            questions=self.questions,
            answers=self.answers,
            guesses=self.guesses,
            final_guess=self.final_guess,
            was_correct=self.game_over,
            turns_used=self.turn,
            secret_raw_responses=self.secret_raw_responses,
            turn_log=self.turn_log,
            max_turns=config.MAX_TURNS,
        )
