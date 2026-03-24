"""
evaluate.py — Three-layer evaluation framework for 20 Questions LLM games.

Layer 1 — Game Outcome        (win, efficiency, secret-keeper reliability)
Layer 2 — Question Quality    (semantic relevance, diversity, information gain)
Layer 3 — LLM-as-a-Judge     (local model scores strategy & logic)

Final score = weighted average of the three layers (default: 30 / 40 / 30).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    """Everything captured during one round of 20 Questions."""
    secret: str                     # The ground-truth secret object
    questions: List[str]            # Yes/no questions in order
    answers: List[str]              # "YES" / "NO" responses (parallel to questions)
    guesses: List[str]              # All explicit guesses (GUESS: …)
    final_guess: str                # Last guess made (whether correct or not)
    was_correct: bool               # Did the guesser win?
    turns_used: int                 # Total turns consumed
    secret_raw_responses: List[str] # Raw text from the secret keeper (for reliability)
    max_turns: int = 20


@dataclass
class EvaluationResult:
    # --- Layer 1 ---
    win_score: float                 # 1.0 won / 0.0 lost
    efficiency_score: float          # Normalised turns efficiency (0–1)
    secret_reliability_score: float  # Fraction of valid secret responses
    layer1_score: float              # Weighted composite

    # --- Layer 2 ---
    semantic_relevance_score: float  # Mean cosine-sim(question, secret)
    semantic_diversity_score: float  # 1 − mean pairwise sim between questions
    information_gain_score: float    # Mean normalised IG per Q/A pair
    layer2_score: float              # Weighted composite

    # --- Layer 3 ---
    llm_judge_strategy: float        # 0–1
    llm_judge_question_quality: float
    llm_judge_logical_consistency: float
    llm_judge_efficiency: float
    layer3_score: float              # Mean of sub-scores

    # --- Overall ---
    overall_score: float

    details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"  Secret          : {self.details.get('secret', '?')}",
            f"  Won             : {bool(self.win_score)}",
            f"  Turns used      : {self.details.get('turns_used', '?')} / {self.details.get('max_turns', 20)}",
            "",
            "LAYER 1 — Game Outcome",
            f"  Win                   : {self.win_score:.3f}",
            f"  Efficiency            : {self.efficiency_score:.3f}",
            f"  Secret reliability    : {self.secret_reliability_score:.3f}",
            f"  Layer 1 score         : {self.layer1_score:.3f}",
            "",
            "LAYER 2 — Question Quality",
            f"  Semantic relevance    : {self.semantic_relevance_score:.3f}",
            f"  Semantic diversity    : {self.semantic_diversity_score:.3f}",
            f"  Information gain      : {self.information_gain_score:.3f}",
            f"  Layer 2 score         : {self.layer2_score:.3f}",
            "",
            "LAYER 3 — LLM Judge",
            f"  Strategy              : {self.llm_judge_strategy:.3f}",
            f"  Question quality      : {self.llm_judge_question_quality:.3f}",
            f"  Logical consistency   : {self.llm_judge_logical_consistency:.3f}",
            f"  Efficiency            : {self.llm_judge_efficiency:.3f}",
            f"  Layer 3 score         : {self.layer3_score:.3f}",
            "",
            "=" * 50,
            f"  OVERALL SCORE         : {self.overall_score:.3f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 1 — Game Outcome
# ---------------------------------------------------------------------------

_VALID_SECRET_RESPONSES = {"YES", "NO", "CORRECT", "WRONG"}


def layer1_game_outcome(record: GameRecord) -> Tuple[float, float, float, float]:
    """
    Returns (win_score, efficiency_score, secret_reliability_score, layer1_score).

    win_score        : 1.0 if won, else 0.0
    efficiency_score : if won → (max_turns − turns_used + 1) / max_turns
                       if lost → 0.0
    secret_reliability: fraction of raw responses containing a valid token
    """
    win_score = 1.0 if record.was_correct else 0.0

    if record.was_correct and record.turns_used > 0:
        efficiency_score = (record.max_turns - record.turns_used + 1) / record.max_turns
        efficiency_score = max(0.0, min(1.0, efficiency_score))
    else:
        efficiency_score = 0.0

    if record.secret_raw_responses:
        valid = sum(
            1 for r in record.secret_raw_responses
            if any(v in r.upper() for v in _VALID_SECRET_RESPONSES)
        )
        secret_reliability_score = valid / len(record.secret_raw_responses)
    else:
        secret_reliability_score = 1.0  # no responses to check

    # Composite: win 50%, efficiency 30%, reliability 20%
    layer1_score = (
        0.50 * win_score
        + 0.30 * efficiency_score
        + 0.20 * secret_reliability_score
    )

    return win_score, efficiency_score, secret_reliability_score, layer1_score


# ---------------------------------------------------------------------------
# Layer 2 — Question Quality  (embeddings + information gain)
# ---------------------------------------------------------------------------

_EMBED_MODEL: Optional[SentenceTransformer] = None


def _get_embed_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"[Evaluator] Loading embedding model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def _semantic_relevance(
    questions: List[str],
    secret: str,
    embed_model: SentenceTransformer,
) -> float:
    """Mean cosine similarity between each question and the secret string."""
    if not questions:
        return 0.0
    texts = questions + [secret]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    q_embs = embeddings[:-1]   # shape (n_questions, dim)
    s_emb  = embeddings[-1]    # shape (dim,)
    sims = q_embs @ s_emb      # dot product (already L2-normalised = cosine sim)
    # Shift from [-1, 1] → [0, 1]
    return float(np.mean((sims + 1.0) / 2.0))


def _semantic_diversity(
    questions: List[str],
    embed_model: SentenceTransformer,
) -> float:
    """
    1 − mean pairwise cosine similarity between questions.
    Higher = more diverse questions (good).
    """
    if len(questions) < 2:
        return 1.0
    embs = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = embs @ embs.T  # (n, n) cosine similarities
    n = len(questions)
    # Extract upper triangle (excluding diagonal)
    upper_indices = np.triu_indices(n, k=1)
    mean_pairwise = float(np.mean(sim_matrix[upper_indices]))
    diversity = 1.0 - (mean_pairwise + 1.0) / 2.0   # shift then invert
    return max(0.0, min(1.0, diversity))


def _load_dataset(path: str = "dataset.json") -> Tuple[List[Dict], List[str]]:
    """Load objects and canonical questions from the dataset file."""
    with open(path) as f:
        data = json.load(f)
    return data["objects"], data["canonical_questions"]


def _attribute_key_for_canonical(canonical: str) -> str:
    """
    Maps a canonical question string to the attribute key used in dataset.json.
    Uses a simple lookup table aligned with dataset.json keys.
    """
    mapping = {
        "Is it an animal?": "is_animal",
        "Is it a mammal?": "is_mammal",
        "Is it a bird?": "is_bird",
        "Is it a reptile?": "is_reptile",
        "Is it a fish or sea creature?": "is_fish_or_sea_creature",
        "Is it an insect or bug?": "is_insect",
        "Is it a domestic pet?": "is_domestic_pet",
        "Is it a wild animal?": "is_wild_animal",
        "Is it a dog?": "is_dog",
        "Is it a cat?": "is_cat",
        "Is it food or something you can eat?": "is_food",
        "Is it a drink or beverage?": "is_drink",
        "Is it a fruit?": "is_fruit",
        "Is it a vegetable?": "is_vegetable",
        "Is it a meat or protein?": "is_meat",
        "Is it a sweet or dessert?": "is_sweet",
        "Is it a man-made object?": "is_man_made",
        "Is it a vehicle or mode of transport?": "is_vehicle",
        "Is it a piece of furniture?": "is_furniture",
        "Is it an electronic device or technology?": "is_electronic",
        "Is it a tool or instrument?": "is_tool",
        "Is it a weapon?": "is_weapon",
        "Is it clothing or worn on the body?": "is_clothing",
        "Is it a building or structure?": "is_building",
        "Is it art or creative work?": "is_art",
        "Is it a person or human?": "is_person",
        "Is it a fictional character?": "is_fictional",
        "Is it a place or location?": "is_place",
        "Is it a plant or tree?": "is_plant",
        "Is it found indoors?": "found_indoors",
        "Is it found outdoors?": "found_outdoors",
        "Can it fly?": "can_fly",
        "Does it live in water?": "lives_in_water",
        "Is it larger than a car?": "larger_than_car",
        "Is it smaller than a book?": "smaller_than_book",
        "Does it have fur or hair?": "has_fur",
        "Does it have feathers?": "has_feathers",
        "Does it have scales?": "has_scales",
        "Is it alive or was once alive?": "is_alive",
        "Is it used for entertainment or fun?": "is_entertainment",
        "Is it used in sports?": "used_in_sports",
        "Is it a toy?": "is_toy",
        "Is it made of metal?": "made_of_metal",
        "Is it made of wood?": "made_of_wood",
        "Is it made of fabric or cloth?": "made_of_fabric",
        "Is it something you read?": "is_something_you_read",
        "Is it something you wear?": "is_something_you_wear",
        "Can you hold it in one hand?": "can_hold_in_hand",
        "Does it have legs?": "has_legs",
        "Does it make a sound or noise?": "makes_sound",
    }
    return mapping.get(canonical, "")


def _information_gain_score(
    questions: List[str],
    answers: List[str],
    dataset_objects: List[Dict],
    canonical_questions: List[str],
    embed_model: SentenceTransformer,
) -> float:
    """
    For each question/answer pair:
      1. Find the closest canonical question via embedding similarity.
      2. Determine which objects would be eliminated by the answer.
      3. Compute normalised IG = objects_eliminated / objects_remaining_before.

    Returns the mean normalised IG across all questions.
    """
    if not questions or not dataset_objects:
        return 0.0

    canon_embs = embed_model.encode(
        canonical_questions, convert_to_numpy=True, normalize_embeddings=True
    )

    remaining_objects = list(dataset_objects)   # track surviving objects
    ig_scores: List[float] = []

    for question, answer in zip(questions, answers):
        n_before = len(remaining_objects)
        if n_before == 0:
            break

        # --- Find closest canonical question ---
        q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = canon_embs @ q_emb
        best_idx = int(np.argmax(sims))
        best_canon = canonical_questions[best_idx]
        attr_key = _attribute_key_for_canonical(best_canon)

        if not attr_key:
            # Can't map to dataset attribute — skip this question for IG
            ig_scores.append(0.0)
            continue

        # --- Filter remaining objects ---
        answer_bool = answer.strip().upper().startswith("Y")  # YES → True, NO → False

        new_remaining = [
            obj for obj in remaining_objects
            if obj["attributes"].get(attr_key, not answer_bool) == answer_bool
        ]

        n_after = len(new_remaining)
        eliminated = n_before - n_after
        ig = eliminated / n_before  # fraction of space eliminated

        ig_scores.append(ig)
        remaining_objects = new_remaining if new_remaining else remaining_objects

    return float(np.mean(ig_scores)) if ig_scores else 0.0


def layer2_question_quality(
    record: GameRecord,
    dataset_path: str = "dataset.json",
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[float, float, float, float]:
    """
    Returns (semantic_relevance, semantic_diversity, information_gain, layer2_score).
    """
    if not record.questions:
        return 0.0, 0.0, 0.0, 0.0

    embed_model = _get_embed_model(embed_model_name)

    sem_rel   = _semantic_relevance(record.questions, record.secret, embed_model)
    sem_div   = _semantic_diversity(record.questions, embed_model)

    try:
        objects, canonicals = _load_dataset(dataset_path)
        ig_score = _information_gain_score(
            record.questions, record.answers, objects, canonicals, embed_model
        )
    except FileNotFoundError:
        print(f"[Evaluator] Warning: dataset file '{dataset_path}' not found — skipping IG.")
        ig_score = 0.0

    # Composite: relevance 40%, diversity 20%, IG 40%
    layer2_score = 0.40 * sem_rel + 0.20 * sem_div + 0.40 * ig_score

    return sem_rel, sem_div, ig_score, layer2_score


# ---------------------------------------------------------------------------
# Layer 3 — LLM-as-a-Judge (local model, no external API)
# ---------------------------------------------------------------------------

_JUDGE_MODEL: Optional[AutoModelForCausalLM] = None
_JUDGE_TOKENIZER: Optional[AutoTokenizer] = None

_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for 20 Questions games. Read the transcript below and score the GUESSER's performance on four dimensions.

SECRET: {secret}

TRANSCRIPT:
{transcript}

Score each dimension as an integer from 1 (very poor) to 10 (excellent):

1. STRATEGY     — Did the guesser use a smart binary-search / narrowing strategy?
2. QUESTION_QUALITY — Were the yes/no questions well-formed and informative?
3. LOGICAL_CONSISTENCY — Did the guesser stay consistent with previous answers and avoid contradictions?
4. EFFICIENCY   — How efficiently were turns used? (10 = won fast, 1 = wasted many turns)

Output MUST be exactly 4 lines, no extra text, no explanations, no punctuation beyond the colon and numbers.
If you add anything else, the output is invalid.

Format:
STRATEGY: <1-10>
QUESTION_QUALITY: <1-10>
LOGICAL_CONSISTENCY: <1-10>
EFFICIENCY: <1-10>"""


def _build_transcript(record: GameRecord) -> str:
    lines = []
    q_idx = 0
    g_idx = 0
    for i in range(record.turns_used):
        if q_idx < len(record.questions) and (g_idx >= len(record.guesses) or q_idx <= g_idx):
            lines.append(f"Turn {i+1}: Q: {record.questions[q_idx]}  A: {record.answers[q_idx] if q_idx < len(record.answers) else '?'}")
            q_idx += 1
        elif g_idx < len(record.guesses):
            correctness = "CORRECT" if (g_idx == len(record.guesses) - 1 and record.was_correct) else "WRONG"
            lines.append(f"Turn {i+1}: GUESS: {record.guesses[g_idx]}  → {correctness}")
            g_idx += 1
    return "\n".join(lines) if lines else "(no turns recorded)"


def load_judge_model(model_name: str) -> None:
    """
    Pre-load a local judge model. Call this once before evaluate_game() if you
    want to reuse the same model instance across multiple evaluations.
    """
    global _JUDGE_MODEL, _JUDGE_TOKENIZER
    print(f"[Evaluator] Loading judge model: {model_name}")
    _JUDGE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _JUDGE_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("[Evaluator] Judge model loaded.")


def _parse_judge_scores(text: str) -> Dict[str, float]:
    """Extract scores from judge output. Returns values in [0, 1]."""
    keys = [
        ("STRATEGY", "strategy"),
        ("QUESTION_QUALITY", "question_quality"),
        ("LOGICAL_CONSISTENCY", "logical_consistency"),
        ("EFFICIENCY", "efficiency"),
    ]
    scores: Dict[str, float] = {}
    missing = []
    for raw_key, out_key in keys:
        match = re.search(rf"{raw_key}\s*:\s*(\d+)", text, re.IGNORECASE)
        if match:
            raw = int(match.group(1))
            scores[out_key] = max(0.0, min(1.0, (raw - 1) / 9.0))  # map 1–10 → 0–1
        else:
            missing.append(out_key)

    if missing:
        # Fallback: take the first four integers in 1–10 range
        ints = [int(x) for x in re.findall(r"(10|[1-9])", text)]
        if len(ints) >= 4:
            fallback = {
                "strategy": ints[0],
                "question_quality": ints[1],
                "logical_consistency": ints[2],
                "efficiency": ints[3],
            }
            for k, v in fallback.items():
                if k in missing:
                    scores[k] = max(0.0, min(1.0, (v - 1) / 9.0))

    # Neutral fallback if still missing
    for _, out_key in keys:
        if out_key not in scores:
            scores[out_key] = 0.5
    return scores


def layer3_llm_judge(
    record: GameRecord,
    judge_model_name: str,
    max_new_tokens: int = 64,
) -> Tuple[float, float, float, float, float]:
    """
    Uses a local LLM to judge the game transcript.
    Returns (strategy, question_quality, logical_consistency, efficiency, layer3_score).

    If the judge model is not yet loaded, it loads it automatically.
    Pass judge_model_name="<already_loaded>" to reuse _JUDGE_MODEL/_JUDGE_TOKENIZER.
    """
    global _JUDGE_MODEL, _JUDGE_TOKENIZER

    if _JUDGE_MODEL is None:
        load_judge_model(judge_model_name)

    transcript = _build_transcript(record)
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        secret=record.secret,
        transcript=transcript,
    )

    messages = [
        {"role": "system", "content": "You are a fair and precise game evaluator."},
        {"role": "user",   "content": prompt},
    ]

    text = _JUDGE_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _JUDGE_TOKENIZER([text], return_tensors="pt").to(_JUDGE_MODEL.device)

    with torch.no_grad():
        output_ids = _JUDGE_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy for reproducibility
            pad_token_id=_JUDGE_TOKENIZER.eos_token_id,
        )

    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    raw_text = _JUDGE_TOKENIZER.decode(generated, skip_special_tokens=True).strip()
    raw_text_lines = raw_text.splitlines()
    if len(raw_text_lines) > 4:
        raw_text = "\n".join(raw_text_lines[:4])
    print(f"[Judge raw output]: {raw_text}")

    scores = _parse_judge_scores(raw_text)
    strategy    = scores["strategy"]
    q_quality   = scores["question_quality"]
    logic       = scores["logical_consistency"]
    efficiency  = scores["efficiency"]
    layer3_score = (strategy + q_quality + logic + efficiency) / 4.0

    return strategy, q_quality, logic, efficiency, layer3_score


# ---------------------------------------------------------------------------
# Master evaluation function
# ---------------------------------------------------------------------------

def evaluate_game(
    record: GameRecord,
    dataset_path: str = "dataset.json",
    embed_model_name: str = "all-MiniLM-L6-v2",
    judge_model_name: str = "Qwen/Qwen3-4B-Instruct",
    layer_weights: Tuple[float, float, float] = (0.30, 0.40, 0.30),
    run_judge: bool = True,
) -> EvaluationResult:
    """
    Full three-layer evaluation of a single game.

    Args:
        record          : completed game record
        dataset_path    : path to dataset.json
        embed_model_name: sentence-transformers model for Layer 2
        judge_model_name: local HuggingFace model for Layer 3
        layer_weights   : (w1, w2, w3) must sum to 1
        run_judge       : set False to skip Layer 3 (faster, no extra model needed)
    """
    assert abs(sum(layer_weights) - 1.0) < 1e-6, "layer_weights must sum to 1"

    # --- Layer 1 ---
    win, eff, rel, l1 = layer1_game_outcome(record)

    # --- Layer 2 ---
    sem_rel, sem_div, ig, l2 = layer2_question_quality(
        record, dataset_path=dataset_path, embed_model_name=embed_model_name
    )

    # --- Layer 3 ---
    if run_judge:
        strat, qq, lc, eff_j, l3 = layer3_llm_judge(record, judge_model_name=judge_model_name)
    else:
        strat = qq = lc = eff_j = l3 = 0.0

    # --- Overall ---
    w1, w2, w3 = layer_weights
    overall = w1 * l1 + w2 * l2 + (w3 * l3 if run_judge else 0.0)
    if not run_judge:
        # Re-normalise to w1+w2 if judge is skipped
        total_w = w1 + w2
        overall = (w1 * l1 + w2 * l2) / total_w

    result = EvaluationResult(
        win_score=win,
        efficiency_score=eff,
        secret_reliability_score=rel,
        layer1_score=l1,
        semantic_relevance_score=sem_rel,
        semantic_diversity_score=sem_div,
        information_gain_score=ig,
        layer2_score=l2,
        llm_judge_strategy=strat,
        llm_judge_question_quality=qq,
        llm_judge_logical_consistency=lc,
        llm_judge_efficiency=eff_j,
        layer3_score=l3,
        overall_score=overall,
        details={
            "secret": record.secret,
            "turns_used": record.turns_used,
            "max_turns": record.max_turns,
            "was_correct": record.was_correct,
            "num_questions": len(record.questions),
            "num_guesses": len(record.guesses),
        },
    )

    return result


# ---------------------------------------------------------------------------
# Aggregate helper — summarise multiple games
# ---------------------------------------------------------------------------

def summarise_results(results: List[EvaluationResult]) -> Dict[str, float]:
    """Average all numeric scores across a list of EvaluationResult objects."""
    if not results:
        return {}

    fields = [
        "win_score", "efficiency_score", "secret_reliability_score", "layer1_score",
        "semantic_relevance_score", "semantic_diversity_score", "information_gain_score", "layer2_score",
        "llm_judge_strategy", "llm_judge_question_quality",
        "llm_judge_logical_consistency", "llm_judge_efficiency", "layer3_score",
        "overall_score",
    ]
    summary = {}
    for f in fields:
        summary[f"avg_{f}"] = float(np.mean([getattr(r, f) for r in results]))
    summary["num_games"] = len(results)
    summary["num_wins"] = int(sum(r.win_score for r in results))
    return summary