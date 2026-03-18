"""
game/hint.py
────────────
HintGame: extends BaseGame with a USE_HINT action.

Hints are stored in a per-round PDF file and revealed one page at a time
when the guesser outputs USE_HINT. Requesting a hint does NOT consume a turn.

The number of hints used is recorded in GameRecord.hints_used.
"""

from __future__ import annotations

import config
from evaluation.records import GameRecord
from game.base import BaseGame
from models import generate_answer
from word_bank.hints import get_hints_for_secret

try:
    import fitz         # PyMuPDF — reads hints back from PDF
    from fpdf import FPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False
    print("[HintGame] Warning: fpdf/fitz not installed. Hints will be served from memory.")


class HintGame(BaseGame):
    """
    20 Questions with an optional hint mechanic.

    The guesser can output USE_HINT to receive the next hint in the sequence.
    Hints are written to a PDF at game start and read back page-by-page so the
    delivery mechanism mirrors a real "sealed envelope" reveal.

    USE_HINT does not count as a turn — the loop continues without incrementing
    self.turn, so the guesser still has the full turn budget for questions/guesses.

    Inherits all Q/A and guess handling from BaseGame.
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
        super().__init__(
            secret_prompt, secret_label, round_number,
            guesser_model, guesser_tokenizer,
            secret_model, secret_tokenizer,
            guesser_system_prompt,
        )
        self.hints      = get_hints_for_secret(secret_label)
        self.hints_used = 0
        self._hints_pdf_path = f"hints_{self.secret_label.replace(' ', '_')}_r{round_number}.pdf"

        if _PDF_AVAILABLE and hints:
            self._build_hints_pdf()

    # ── PDF helpers ──────────────────────────────────────────────────────────

    def _build_hints_pdf(self) -> None:
        """Write all hints to a PDF, one hint per page."""
        pdf = FPDF()
        for i, hint in enumerate(self.hints):
            pdf.add_page()
            pdf.set_font("Helvetica", size=16)
            pdf.cell(0, 10, f"Hint {i + 1}:", ln=True)
            pdf.set_font("Helvetica", size=13)
            pdf.cell(0, 10, hint, ln=True)
        pdf.output(self._hints_pdf_path)

    def _read_hint_from_pdf(self, index: int) -> str:
        """Read a single hint page from the PDF by zero-based index."""
        doc = fitz.open(self._hints_pdf_path)
        text = doc[index].get_text().strip()
        doc.close()
        return text

    # ── Hint delivery ────────────────────────────────────────────────────────

    def _use_hint(self) -> str:
        """
        Reveal the next hint and return its text.

        Returns a "no more hints" message if all hints have been used.
        Falls back to serving from self.hints if PDF is unavailable.
        """
        if self.hints_used >= len(self.hints):
            return "No more hints available."

        if _PDF_AVAILABLE:
            text = self._read_hint_from_pdf(self.hints_used)
        else:
            text = self.hints[self.hints_used]

        self.hints_used += 1
        print(f"[HINT {self.hints_used}/{len(self.hints)}] {text}")
        return text

    # ── Action parsing ───────────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str, str | None]:
        """
        Extend base parsing to also recognise the USE_HINT action.

        USE_HINT is detected before QUESTION/GUESS so it cannot be shadowed.
        """
        if "USE_HINT" in text.strip().upper():
            return "hint", None
        return super()._parse_action(text)

    # ── Main loop (override to handle hint action) ───────────────────────────

    def play(self) -> GameRecord:
        """
        Run the hint game loop.

        USE_HINT does not increment self.turn — the guesser keeps its full
        turn budget for actual questions and guesses.

        Returns:
            GameRecord with hints_used populated.
        """
        print(f"\n=== ROUND {self.round_number} START (HINT MODE) ===")

        while self.turn < config.MAX_TURNS and not self.game_over:

            guesser_output = generate_answer(
                self.guesser_messages, self.guesser_model, self.guesser_tokenizer
            )
            print(f"Guesser: {guesser_output}")

            action, content = self._parse_action(guesser_output)

            if action == "hint":
                hint_text = self._use_hint()
                self.guesser_messages.append({
                    "role": "user",
                    "content": (
                        f"[HINT {self.hints_used}/{len(self.hints)}] {hint_text}\n"
                        "Now continue — ask a question or make a guess."
                    ),
                })
                # Does NOT increment self.turn
                continue

            elif action == "question":
                answer = self._handle_question(content)
                print(f"Secret: {answer}")

            elif action == "guess":
                correct = self._handle_guess(content)
                print(f"Secret: {'CORRECT' if correct else 'WRONG'}")
                if correct:
                    print(f"Guesser won in {self.turn + 1} turns using {self.hints_used} hints!")
                    self.game_over = True

            else:
                answer = self._handle_question(content)
                print(f"Secret: {answer}")

            self.turn += 1

        if not self.game_over:
            print(
                f"Guesser failed after {config.MAX_TURNS} turns "
                f"using {self.hints_used} hints."
            )
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
            hints_used=self.hints_used,
        )
