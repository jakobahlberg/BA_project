from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np


@dataclass
class Case:
    secret: str
    guess: str
    expected: bool
    bucket: str


CASES: List[Case] = [
    Case("golden retriever", "golden retriever",                          True,  "easy_correct"),
    Case("London",           "London",                                    True,  "easy_correct"),
    Case("Eiffel Tower",     "Eiffel Tower",                              True,  "easy_correct"),
    Case("dog",              "dog",                                       True,  "easy_correct"),
    Case("banana",           "a banana",                                  True,  "easy_correct"),

    Case("London",           "Tokyo",                                     False, "easy_wrong"),
    Case("golden retriever", "car",                                       False, "easy_wrong"),
    Case("Eiffel Tower",     "spoon",                                     False, "easy_wrong"),
    Case("dog",              "computer",                                  False, "easy_wrong"),
    Case("banana",           "skyscraper",                                False, "easy_wrong"),

    Case("golden retriever", "a black golden retriever dog",              True,  "hard_correct"),
    Case("pizza",            "italian pizza with cheese",                 True,  "hard_correct"),
    Case("Albert Einstein",  "the german physicist Einstein",             True,  "hard_correct"),
    Case("Eiffel Tower",     "the famous Eiffel Tower in Paris",          True,  "hard_correct"),
    Case("banana",           "a ripe yellow banana fruit",                True,  "hard_correct"),

    Case("London",           "Which river is connected to London?",       False, "hard_wrong"),
    Case("golden retriever", "labrador",                                  False, "hard_wrong"),
    Case("London",           "Paris",                                     False, "hard_wrong"),
    Case("dog",              "Is the dog brown?",                         False, "hard_wrong"),
    Case("golden retriever", "a dog",                                     False, "hard_wrong"),

    Case("London",           "the capital of England",                    False, "description_wrong"),
    Case("London",           "the London Eye",                            False, "description_wrong"),
    Case("Albert Einstein",  "the physicist known for relativity theory", False, "description_wrong"),
    Case("Eiffel Tower",     "the tower in Paris",                        False, "description_wrong"),
    Case("banana",           "a yellow curved fruit",                     False, "description_wrong"),

    Case("pizza",            "something topped with cheese and sauce",    False, "hint_wrong"),
    Case("Albert Einstein",  "the theory of relativity",                  False, "hint_wrong"),
    Case("London",           "the capital of the United Kingdom",         False, "hint_wrong"),
    Case("Marie Curie",      "the scientist who discovered radioactivity", False, "hint_wrong"),
    Case("submarine",        "a vehicle that travels underwater",         False, "hint_wrong"),

    Case("golden retriever", "labrador retriever",                        False, "close_wrong"),
    Case("Albert Einstein",  "Isaac Newton",                              False, "close_wrong"),
    Case("violin",           "viola",                                     False, "close_wrong"),
    Case("Tokyo",            "Kyoto",                                     False, "close_wrong"),
    Case("pizza",            "calzone",                                   False, "close_wrong"),
]

def make_bi_encoder_scorer(model_name: str) -> Callable[[str, str], float]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def score(secret: str, guess: str) -> float:
        embs = model.encode([secret, guess], convert_to_numpy=True, normalize_embeddings=True)
        return float(embs[0] @ embs[1])
    return score


def make_cross_encoder_scorer(model_name: str) -> Callable[[str, str], float]:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)

    def score(secret: str, guess: str) -> float:
        s = model.predict([(secret, guess)])
        return float(s[0])
    return score


METHODS = [
    ("A_minilm",    "A. MiniLM-L6-v2 (cosine)",
        lambda: make_bi_encoder_scorer("sentence-transformers/all-MiniLM-L6-v2")),
    ("C_crossenc",  "C. stsb-roberta-large (cross-encoder)",
        lambda: make_cross_encoder_scorer("cross-encoder/stsb-roberta-large")),
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def required_token(secret: str) -> str:
    """
    Pick the longest content word of the secret as the required token.
    On length ties, prefer the later word (typical English head / surname).
    """
    words = _normalize(secret).split()
    if not words:
        return ""
    max_len = max(len(w) for w in words)
    candidates = [w for w in words if len(w) == max_len]
    return candidates[-1]


def structural_pass(secret: str, guess: str) -> bool:
    """The longest content word of secret must appear as a token in guess."""
    req = required_token(secret)
    if not req:
        return False
    return req in _normalize(guess).split()


def confusion(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(( pred &  y).sum())
    fp = int(( pred & ~y).sum())
    fn = int((~pred &  y).sum())
    tn = int((~pred & ~y).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep(scores: List[float], expected: List[bool],
          structural: List[bool] = None, n_steps: int = 21):
    """
    Sweep across the observed score range; return curve and best-F1 row.

    If structural is given, predicted = structural AND (sim >= t).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(expected, dtype=bool)
    struct = (np.asarray(structural, dtype=bool)
              if structural is not None else np.ones_like(y, dtype=bool))

    relevant = s[struct] if struct.any() else s
    lo, hi = relevant.min(), relevant.max()
    if hi - lo < 1e-9:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo - 0.01, hi + 0.01, n_steps)

    curve, best = [], {"f1": -1.0}
    for t in thresholds:
        pred = struct & (s >= t)
        c = confusion(pred, y)
        row = {"t": float(t), **c}
        curve.append(row)
        if row["f1"] > best["f1"]:
            best = row
    return curve, best


def print_structural(structural: List[bool], y: np.ndarray):
    print(f"\n{'='*78}\n  STRUCTURAL RULE  (longest content word of secret must appear in guess)\n{'='*78}")
    print(f"  {'bucket':<18} {'exp':<4} {'struct':<7} {'required':<14} {'secret':<22} → guess")
    for c, sp in zip(CASES, structural):
        mark = "T" if c.expected else "F"
        st = "PASS" if sp else "FAIL"
        print(f"  {c.bucket:<18} {mark:<4} {st:<7} {required_token(c.secret):<14} {c.secret:<22} → {c.guess}")
    pred = np.asarray(structural, dtype=bool)
    cm = confusion(pred, y)
    print(f"\n  Structural alone:  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}  "
          f"→  P={cm['p']:.2f}  R={cm['r']:.2f}  F1={cm['f1']:.2f}")


def print_per_case(label: str, scores: List[float], structural: List[bool]):
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    print(f"  {'bucket':<18} {'exp':<4} {'struct':<7} {'score':>7}   {'secret':<22} → guess")
    for c, s, sp in zip(CASES, scores, structural):
        mark = "T" if c.expected else "F"
        st = "PASS" if sp else "FAIL"
        print(f"  {c.bucket:<18} {mark:<4} {st:<7} {s:7.3f}   {c.secret:<22} → {c.guess}")


def print_curve(label: str, curve, best):
    print(f"\n  Threshold sweep ({label}):")
    print(f"  {'thr':>6}  {'P':>5}  {'R':>5}  {'F1':>5}   {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
    for row in curve:
        marker = "  ←best" if row is best else ""
        print(f"  {row['t']:6.3f}  {row['p']:5.2f}  {row['r']:5.2f}  {row['f1']:5.2f}   "
              f"{row['tp']:3d} {row['fp']:3d} {row['fn']:3d} {row['tn']:3d}{marker}")


def write_scores_csv(path: Path, all_scores: Dict[str, List[float]], structural: List[bool]):
    keys = list(all_scores.keys())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "expected", "secret", "guess", "required_token", "structural"] + keys)
        for i, c in enumerate(CASES):
            row = [c.bucket, int(c.expected), c.secret, c.guess,
                   required_token(c.secret), int(structural[i])]
            for k in keys:
                row.append(f"{all_scores[k][i]:.4f}")
            w.writerow(row)


def write_curve_csv(path: Path, all_curves: Dict[str, list]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"])
        for method, curve in all_curves.items():
            for row in curve:
                w.writerow([method, f"{row['t']:.4f}",
                            f"{row['p']:.4f}", f"{row['r']:.4f}", f"{row['f1']:.4f}",
                            row["tp"], row["fp"], row["fn"], row["tn"]])


def main():
    print(f"Cases: {len(CASES)}  "
          f"(easy_correct={sum(c.bucket=='easy_correct' for c in CASES)}, "
          f"easy_wrong={sum(c.bucket=='easy_wrong' for c in CASES)}, "
          f"hard_correct={sum(c.bucket=='hard_correct' for c in CASES)}, "
          f"hard_wrong={sum(c.bucket=='hard_wrong' for c in CASES)}, "
          f"description_wrong={sum(c.bucket=='description_wrong' for c in CASES)}, "
          f"hint_wrong={sum(c.bucket=='hint_wrong' for c in CASES)}, "
          f"close_wrong={sum(c.bucket=='close_wrong' for c in CASES)})")

    expected = [c.expected for c in CASES]
    y = np.asarray(expected, dtype=bool)
    structural = [structural_pass(c.secret, c.guess) for c in CASES]
    print_structural(structural, y)

    all_scores: Dict[str, List[float]] = {}
    all_curves: Dict[str, list] = {}
    summary = []   # (key, label, mode, best)  mode ∈ {"sim", "sim+struct"}

    for key, label, builder in METHODS:
        print(f"\n[Loading] {label} ...", flush=True)
        try:
            scorer = builder()
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}: {e})")
            continue

        try:
            scores = [scorer(c.secret, c.guess) for c in CASES]
        except Exception as e:
            print(f"  Scoring failed ({type(e).__name__}: {e})")
            continue

        # Sim only
        curve_sim, best_sim = sweep(scores, expected)
        # Sim AND structural
        curve_comb, best_comb = sweep(scores, expected, structural=structural)

        all_scores[key] = scores
        all_curves[f"{key}__sim"] = curve_sim
        all_curves[f"{key}__sim+struct"] = curve_comb
        summary.append((key, label, "sim only",    best_sim))
        summary.append((key, label, "sim+struct",  best_comb))

        print_per_case(label, scores, structural)
        print_curve(f"{label} — sim only",        curve_sim,  best_sim)
        print_curve(f"{label} — sim + structural", curve_comb, best_comb)

    if not summary:
        print("\nNo methods completed.")
        return

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scores_csv(out_dir / "calibration_scores.csv", all_scores, structural)
    write_curve_csv(out_dir / "calibration_curve.csv", all_curves)
    print(f"\nWrote {out_dir/'calibration_scores.csv'} and {out_dir/'calibration_curve.csv'}")

    print(f"\n{'='*78}\n  RANKING (by best F1)\n{'='*78}")
    summary.sort(key=lambda x: -x[3]["f1"])
    for key, label, mode, best in summary:
        tag = f"{label}  [{mode}]"
        print(f"  {tag:<58}  F1={best['f1']:.2f}  @ t={best['t']:.3f}  "
              f"(P={best['p']:.2f}, R={best['r']:.2f})")


if __name__ == "__main__":
    main()
