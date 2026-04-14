#!/usr/bin/env python3
"""
view_run.py - Display a clean, paired transcript+evaluation view of a .out run file.

Usage:
    python view_run.py <path_to_out_file>
    python view_run.py                      # auto-picks the first seed-1 .out file
"""

import re
import sys
from pathlib import Path

try:
    from word_bank.hints import get_hints_for_secret
    _HINTS_AVAILABLE = True
except ImportError:
    _HINTS_AVAILABLE = False

# ── Regex patterns ──────────────────────────────────────────────────────────

ROUND_START   = re.compile(r"^=== ROUND (\d+) START")
GUESSER_LINE  = re.compile(r"^Guesser: (.+)$", re.DOTALL)
SECRET_RESP   = re.compile(r"^Secret: (YES|NO|CORRECT|WRONG)$")
HINT_REVEAL   = re.compile(r"^\[HINT (\d+)/(\d+)\]")
WIN_LINE      = re.compile(r"^Guesser (won in \d+ turns.+|failed after \d+ turns.+)")
EVAL_SEP      = re.compile(r"^={30,}$")
ACTION_LINE   = re.compile(r"^(QUESTION|GUESS|USE_HINT|WEB_SEARCH)([:\s]|$)", re.IGNORECASE)
KV_LINE       = re.compile(r"^\s{2,}(\S[^:]+?)\s*:\s*(.+)$")
SUMMARY_START = re.compile(r"^=== SUMMARY ===")

# Transcript entry types:
#   str                          — plain line (Guesser action, Secret, win/fail)
#   ("hint", n, total, model_text)  — hint event; model_text is first buf occurrence


# ── Helpers ──────────────────────────────────────────────────────────────────

def _first_action(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if ACTION_LINE.match(line):
            return line
    return raw.splitlines()[0].strip() if raw else raw


def _fmt_float(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


# ── Round transcript parser ──────────────────────────────────────────────────

def parse_rounds(lines: list[str]) -> dict[int, list]:
    """
    Return {round_number: [transcript_entries]}.

    Each entry is either a str (plain line) or a
    ("hint", n, total, model_text) tuple for hint events.
    model_text is the FIRST occurrence of that hint number in the buffer —
    if the model hallucinated hint text before USE_HINT, that shows up here.
    The authoritative hint text is looked up at display time via get_hints_for_secret.
    """
    rounds: dict[int, list] = {}
    cur_round: int | None = None
    transcript: list = []
    in_guesser = False
    guesser_buf: list[str] = []

    def _flush_guesser():
        nonlocal in_guesser, guesser_buf
        if guesser_buf:
            action = _first_action("\n".join(guesser_buf))
            transcript.append(f"Guesser: {action}")

            # Scan buffer for hint lines.
            # First occurrence = what the model produced (possibly hallucinated).
            # Last occurrence = real hint printed by game loop after processing.
            # We store the first occurrence as model_text so display can compare
            # it against the known hint list.
            hint_first: dict[int, str] = {}
            hint_last:  dict[int, tuple] = {}
            for j, buf_line in enumerate(guesser_buf):
                hm = re.search(r"\[HINT (\d+)/(\d+)\]", buf_line)
                if hm:
                    n, total = int(hm.group(1)), int(hm.group(2))
                    text = guesser_buf[j + 1].strip() if j + 1 < len(guesser_buf) else ""
                    if n not in hint_first:
                        hint_first[n] = text
                    hint_last[n] = (n, total, text)

            for n in sorted(hint_last):
                _, total, _ = hint_last[n]
                model_text = hint_first.get(n, "")
                transcript.append(("hint", n, total, model_text))

        in_guesser = False
        guesser_buf = []

    i = 0
    while i < len(lines):
        line = lines[i]

        m = ROUND_START.match(line)
        if m:
            _flush_guesser()
            if cur_round is not None:
                rounds[cur_round] = transcript[:]
            cur_round = int(m.group(1))
            transcript = []
            i += 1
            continue

        if SUMMARY_START.match(line) or EVAL_SEP.match(line):
            _flush_guesser()
            if cur_round is not None:
                rounds[cur_round] = transcript[:]
                cur_round = None
            i += 1
            continue

        if cur_round is None:
            i += 1
            continue

        gm = GUESSER_LINE.match(line)
        if gm:
            _flush_guesser()
            in_guesser = True
            guesser_buf = [gm.group(1)]
            i += 1
            continue

        if in_guesser:
            if SECRET_RESP.match(line) or WIN_LINE.match(line):
                _flush_guesser()
                # fall through
            else:
                guesser_buf.append(line)
                i += 1
                continue

        if SECRET_RESP.match(line):
            transcript.append(f"  Secret:  {line.split(': ', 1)[1]}")
        elif WIN_LINE.match(line):
            transcript.append(f"  >>> {line}")

        i += 1

    _flush_guesser()
    if cur_round is not None:
        rounds[cur_round] = transcript[:]

    return rounds


# ── Evaluation block parser ───────────────────────────────────────────────────

def parse_eval_blocks(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    in_eval = False
    block: dict = {}

    i = 0
    while i < len(lines):
        line = lines[i]

        if EVAL_SEP.match(line):
            nxt  = lines[i + 1].strip() if i + 1 < len(lines) else ""
            nxt2 = lines[i + 2].strip() if i + 2 < len(lines) else ""

            if nxt == "EVALUATION RESULTS" and EVAL_SEP.match(nxt2):
                if block:
                    blocks.append(block)
                block = {}
                in_eval = True
                i += 3
                continue
            elif in_eval:
                blocks.append(block)
                block = {}
                in_eval = False

        if in_eval:
            kv = KV_LINE.match(line)
            if kv:
                block[kv.group(1).strip()] = kv.group(2).strip()

        i += 1

    if in_eval and block:
        blocks.append(block)

    return blocks


# ── Summary parser ────────────────────────────────────────────────────────────

def parse_summary(lines: list[str]) -> list[str]:
    in_summary = False
    out = []
    for line in lines:
        if SUMMARY_START.match(line):
            in_summary = True
        if in_summary:
            out.append(line)
    return out


# ── Display ───────────────────────────────────────────────────────────────────

def display(rounds: dict, evals: list[dict], summary: list[str], file=sys.stdout):
    WIDTH = 72

    def hr(prefix="  "):
        print(prefix + "─" * (WIDTH - len(prefix)), file=file)

    print(file=file)

    for rn in sorted(rounds.keys()):
        transcript = rounds[rn]
        ev = evals[rn - 1] if rn - 1 < len(evals) else {}

        secret = ev.get("Secret", "???")
        won    = ev.get("Won", "?")
        turns  = ev.get("Turns used", "?")
        icon   = "✓" if won == "True" else "✗"

        # Look up authoritative hints for this round's secret
        known_hints: list[str] = []
        if _HINTS_AVAILABLE and secret != "???":
            try:
                known_hints = get_hints_for_secret(secret)
            except Exception:
                pass

        print("═" * WIDTH, file=file)
        print(f"  ROUND {rn}  —  {secret}   {icon} {turns} turns", file=file)
        print("═" * WIDTH, file=file)
        print(file=file)

        for entry in transcript:
            if isinstance(entry, tuple) and entry[0] == "hint":
                _, n, total, model_text = entry

                # Authoritative text from the hints list
                known_text = known_hints[n - 1] if n - 1 < len(known_hints) else None

                if known_text:
                    print(f"  [HINT {n}/{total}] {known_text}", file=file)
                    if model_text and model_text != known_text:
                        print(f"    ~ hallucinated: \"{model_text}\"", file=file)
                else:
                    # No hint list available — fall back to what appeared in buf
                    print(f"  [HINT {n}/{total}] {model_text}", file=file)

            elif isinstance(entry, str):
                if entry.startswith("Guesser:"):
                    print(f"  {entry}", file=file)
                else:
                    print(entry, file=file)

        print(file=file)

        if ev:
            l1  = ev.get("Layer 1 score", "—")
            l2  = ev.get("Layer 2 score", "—")
            l3  = ev.get("Layer 3 score", "—")
            eff = ev.get("Efficiency", "—")
            rel = ev.get("Semantic relevance", "—")
            cov = ev.get("Canonical coverage", "—")
            ig  = ev.get("Information gain", "—")
            vw  = ev.get("Verified win", "—")

            print(f"  EVALUATION", file=file)
            hr()
            print(f"  Layer 1 (Outcome):   {_fmt_float(l1):>6}   "
                  f"Efficiency: {_fmt_float(eff)}", file=file)
            print(f"  Layer 2 (Questions): {_fmt_float(l2):>6}   "
                  f"Relevance: {_fmt_float(rel)}  "
                  f"Coverage: {_fmt_float(cov)}  "
                  f"Info-gain: {_fmt_float(ig)}", file=file)
            print(f"  Layer 3 (Judge):     {_fmt_float(l3):>6}", file=file)
            print(f"  Win verified: {vw}", file=file)

        print(file=file)

    if summary:
        print("═" * WIDTH, file=file)
        print("  SUMMARY", file=file)
        print("═" * WIDTH, file=file)
        for line in summary:
            if line.strip() and not SUMMARY_START.match(line):
                print(f"  {line}", file=file)
        print(file=file)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if args and args[0] not in ("--all",):
        path = Path(args[0])
    else:
        candidates = sorted(
            Path(".").glob("results/**/*seed1*.out"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            candidates = sorted(
                Path(".").glob("results/**/*.out"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if not candidates:
            print("No .out files found. Pass a path explicitly.", file=sys.stderr)
            sys.exit(1)
        path = candidates[0]

    print(f"Parsing: {path}", file=sys.stderr)
    lines = path.read_text(errors="replace").splitlines()

    rounds  = parse_rounds(lines)
    evals   = parse_eval_blocks(lines)
    summary = parse_summary(lines)

    if len(rounds) != len(evals):
        print(
            f"  Warning: {len(rounds)} rounds but {len(evals)} eval blocks "
            f"— some rounds may be unmatched.",
            file=sys.stderr,
        )

    display(rounds, evals, summary)


if __name__ == "__main__":
    main()
