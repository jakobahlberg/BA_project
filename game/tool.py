from __future__ import annotations

import config
from evaluation.records import GameRecord
from game.base import BaseGame
from models import generate_answer
from word_bank.hints import get_hints_for_secret

try:
    import fitz
    from fpdf import FPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False
    print("[ToolGame] Warning: fpdf/fitz not installed. Hints will be served from memory.")


def _run_web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo full-text search via DDGS. Returns a short snippet string."""
    q = query.strip()
    if not q:
        return "No query provided."
    try:
        from ddgs import DDGS
        results = DDGS().text(q, max_results=max_results)
        if not results:
            return "No web results found."
        snippets = [f"{r['title']}: {r['body']}" for r in results]
        return " | ".join(snippets)
    except Exception as e:
        return f"Web search unavailable: {e}"


class ToolGame(BaseGame):
    """
    20 Questions with USE_HINT and WEB_SEARCH tool actions.

    USE_HINT reveals the next hint from a pre-built sequence.
    WEB_SEARCH: <query> fetches a DuckDuckGo snippet.
    Neither action increments the turn counter.
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
        self.hints             = get_hints_for_secret(secret_label)
        self.hints_used        = 0
        self.web_searches_used = 0
        self._hints_pdf_path   = f"hints_{self.secret_label.replace(' ', '_')}_r{round_number}.pdf"

        if _PDF_AVAILABLE and self.hints:
            self._build_hints_pdf()

    # ── PDF helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_pdf_text(text: str) -> str:
        """Normalize text for core FPDF fonts (latin-1 compatible)."""
        return (
            text.replace("’", "'")
                .replace("‘", "'")
                .replace("“", '"')
                .replace("”", '"')
                .replace("–", "-")
                .replace("—", "-")
        )

    def _build_hints_pdf(self) -> None:
        pdf = FPDF()
        for i, hint in enumerate(self.hints):
            pdf.add_page()
            pdf.set_font("Helvetica", size=16)
            pdf.cell(0, 10, f"Hint {i + 1}:", ln=True)
            pdf.set_font("Helvetica", size=13)
            pdf.cell(0, 10, self._safe_pdf_text(hint), ln=True)
        pdf.output(self._hints_pdf_path)

    def _read_hint_from_pdf(self, index: int) -> str:
        doc = fitz.open(self._hints_pdf_path)
        text = self._safe_pdf_text(doc[index].get_text().strip())
        doc.close()
        return text

    # ── Tool actions ─────────────────────────────────────────────────────────

    def _use_hint(self) -> str:
        if self.hints_used >= len(self.hints):
            return "No more hints available."
        text = self._read_hint_from_pdf(self.hints_used) if _PDF_AVAILABLE else self.hints[self.hints_used]
        self.hints_used += 1
        print(f"[HINT {self.hints_used}/{len(self.hints)}] {text}")
        return text

    def _use_web_search(self, query: str) -> str:
        if self.web_searches_used >= config.MAX_WEB_SEARCHES:
            return "No more web searches available."
        result = _run_web_search(query)
        self.web_searches_used += 1
        print(f"[WEB SEARCH {self.web_searches_used}/{config.MAX_WEB_SEARCHES}] {query}")
        print(f"[WEB SEARCH RESULT] {result}")
        return result

    def _remaining_tools_msg(self) -> str:
        """Human-readable summary of the tool budgets that still have uses left."""
        hints_left = len(self.hints) - self.hints_used
        searches_left = config.MAX_WEB_SEARCHES - self.web_searches_used
        parts = []
        if hints_left > 0:
            parts.append(f"USE_HINT ({hints_left} left)")
        if searches_left > 0:
            parts.append(f"WEB_SEARCH ({searches_left} left)")
        return ", ".join(parts)

    # ── Action parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_action_line(line: str) -> str:
        """
        Make minor formatting repairs before parsing an action.

        Some models emit near-miss tool names such as USE_ HINT or
        WEB_ SEARCH. These are clearly intended as tool calls, so normalize
        whitespace around underscores rather than treating them as questions.
        """
        line = " ".join(line.split())
        line = line.strip("`*_ ")
        line = line.replace("_ ", "_").replace(" _", "_")
        line = line.strip("`*_ ")
        return line

    def _parse_action(self, text: str) -> tuple[str, str | None]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        attempted_depleted_tool = False
        for line in lines:
            line = self._normalize_action_line(line)
            if line.upper().startswith("ACTION:"):
                line = self._normalize_action_line(line.split(":", 1)[-1].strip())
            upper = line.upper()
            if upper == "USE_HINT":
                if self.hints_used < len(self.hints):
                    return "hint", None
                attempted_depleted_tool = True
                continue
            if upper.startswith("WEB_SEARCH:"):
                query = line.split(":", 1)[-1].strip()
                if query and self.web_searches_used < config.MAX_WEB_SEARCHES:
                    return "web_search", query
                attempted_depleted_tool = True
                continue
            if upper.startswith("QUESTION:"):
                question = line.split(":", 1)[-1].strip()
                if question:
                    return "question", question
                continue
            if upper.startswith("GUESS:"):
                guess = line.split(":", 1)[-1].strip()
                if guess:
                    return "guess", guess
                continue
        for line in lines:
            if line.endswith("?"):
                return "question", line
        # The model only tried to use a tool whose budget is exhausted. Nudge it
        # to switch strategy rather than silently demoting the line to a question
        # (which would otherwise be answered by the keeper and pollute the record).
        if attempted_depleted_tool:
            return "tool_exhausted", None
        return "unknown", lines[0] if lines else text

    # ── Main loop ────────────────────────────────────────────────────────────

    def play(self) -> GameRecord:
        max_actions_per_round = config.MAX_ACTIONS_PER_ROUND
        action_count = 0
        ended_by_action_cap = False
        header = (
            f"\n=== ROUND {self.round_number} START (TOOL MODE) ===\n"
            f"[INSPECT] Secret target: {self.secret_label}"
        )
        print(header)
        transcript_lines: list[str] = [header]

        while self.turn < config.MAX_TURNS and not self.game_over:
            action_count += 1
            if action_count > max_actions_per_round:
                guard_line = (
                    f"Guardrail triggered: exceeded {max_actions_per_round} actions in one round. "
                    "Round is counted as a loss."
                )
                print(guard_line)
                transcript_lines.append(guard_line)
                ended_by_action_cap = True
                break

            guesser_output = generate_answer(
                self.guesser_messages, self.guesser_model, self.guesser_tokenizer
            )
            print(f"Guesser: {guesser_output}")
            transcript_lines.append(f"Guesser: {guesser_output}")

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
                continue  # does NOT increment turn

            elif action == "web_search":
                search_result = self._use_web_search(content or "")
                self.guesser_messages.append({
                    "role": "user",
                    "content": (
                        f"[WEB_SEARCH] query: {content}\n"
                        f"[WEB_SEARCH_RESULT] {search_result}\n"
                        "Now continue — ask a question or make a guess."
                    ),
                })
                continue  # does NOT increment turn

            elif action == "tool_exhausted":
                avail = self._remaining_tools_msg()
                if avail:
                    guidance = (
                        f"That tool is used up. Tools still available: {avail}. "
                        "Otherwise ask a QUESTION or make a GUESS."
                    )
                else:
                    guidance = (
                        "All tools are used up (no hints or web searches remain). "
                        "Ask a QUESTION or make a GUESS."
                    )
                print(f"[TOOL EXHAUSTED] {guidance}")
                self.guesser_messages.append({"role": "user", "content": guidance})
                continue  # does NOT increment turn

            if action == "question":
                answer = self._handle_question(content)
                print(f"Secret: {answer}")
                transcript_lines.append(f"Secret: {answer}")

            elif action == "guess":
                correct = self._handle_guess(content)
                secret_line = f"Secret: {'CORRECT' if correct else 'WRONG'}"
                print(secret_line)
                transcript_lines.append(secret_line)
                if correct:
                    won_line = (
                        f"Guesser won in {self.turn + 1} turns using "
                        f"{self.hints_used} hints and {self.web_searches_used} web searches!"
                    )
                    print(won_line)
                    transcript_lines.append(won_line)
                    self.game_over = True

            else:
                answer = self._handle_question(content)
                print(f"Secret: {answer}")
                transcript_lines.append(f"Secret: {answer}")

            self.turn += 1

            if not self.game_over:
                avail = self._remaining_tools_msg()
                reminder = (
                    f"Reminder: Consider using {avail}."
                    if avail
                    else "Reminder: No tools remain — ask a QUESTION or make a GUESS."
                )
                self.guesser_messages.append({"role": "user", "content": reminder})

        if not self.game_over:
            if ended_by_action_cap:
                failed_line = (
                    f"Guesser lost due to action cap ({max_actions_per_round}) "
                    f"after {self.turn} turns using {self.hints_used} hints "
                    f"and {self.web_searches_used} web searches."
                )
            else:
                failed_line = (
                    f"Guesser failed after {config.MAX_TURNS} turns "
                    f"using {self.hints_used} hints and {self.web_searches_used} web searches."
                )
            print(failed_line)
            transcript_lines.append(failed_line)
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
            web_searches_used=self.web_searches_used,
            raw_transcript="\n".join(transcript_lines),
        )
