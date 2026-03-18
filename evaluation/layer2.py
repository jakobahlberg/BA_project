"""
evaluation/layer2.py
────────────────────
Layer 2: Question Quality metrics.

Three sub-scores:
  - Semantic relevance:   how related questions are to the secret (embedding cosine sim)
  - Canonical coverage:   how many distinct question dimensions were covered
  - Information gain:     how much each question narrowed the candidate pool

Requires dataset.json and the sentence-transformers embedding model.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from evaluation.records import GameRecord

# Module-level embedding model cache (loaded once, reused across all rounds)
_EMBED_MODEL: Optional[SentenceTransformer] = None


def _get_embed_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the sentence embedding model."""
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
    """
    Mean cosine similarity between each question embedding and the secret string.

    High score = questions closely relate to the concept of the secret.
    Scaled from [-1,1] cosine range to [0,1].
    """
    if not questions:
        return 0.0
    texts = questions + [secret]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    q_embs = embeddings[:-1]
    s_emb  = embeddings[-1]
    sims = q_embs @ s_emb
    return float(np.mean((sims + 1.0) / 2.0))


def _diversity_trace(
    questions: List[str],
    embed_model: SentenceTransformer,
) -> List[float]:
    """
    Per-question novelty relative to all prior questions.

    Returns a list of length len(questions):
      - Index 0 is always 1.0 (no prior questions).
      - Index i = 1 − mean_cosine_sim(Q_i, Q_0..Q_{i-1}), scaled to [0,1].

    High early values = broad exploration.
    Low later values  = narrowing in (expected and good).
    Consistently low  = repetitive questions (bad strategy).

    This trace is logged in details but NOT included in the layer2 composite score.
    """
    if len(questions) <= 1:
        return [1.0] * len(questions)

    embs = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    trace = [1.0]
    for i in range(1, len(questions)):
        prev_sims = embs[:i] @ embs[i]
        mean_sim = float(np.mean(prev_sims))
        trace.append(1.0 - (mean_sim + 1.0) / 2.0)
    return trace


def _canonical_coverage(
    questions: List[str],
    canonical_questions: List[str],
    embed_model: SentenceTransformer,
) -> float:
    """
    Fraction of questions that each covered a distinct canonical dimension.

    Maps each question to its closest canonical question (by embedding cosine sim),
    then counts how many DISTINCT canonicals were covered.

    score = distinct_canonicals_hit / questions_asked

    1.0 = every question covered a unique conceptual dimension (ideal breadth).
    Low = guesser kept asking variants of the same question.
    """
    if not questions:
        return 0.0

    q_embs = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    c_embs = embed_model.encode(canonical_questions, convert_to_numpy=True, normalize_embeddings=True)

    covered = set()
    for q_emb in q_embs:
        sims = c_embs @ q_emb
        best_idx = int(np.argmax(sims))
        covered.add(best_idx)

    return len(covered) / len(questions)


def _load_dataset(path: str = "dataset.json") -> Tuple[List[Dict], List[str]]:
    """Load objects and canonical questions from the dataset JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["objects"], data["canonical_questions"]


def _attribute_key_for_canonical(canonical: str) -> str:
    """Map a canonical question string to its attribute key in the dataset objects."""
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
    Mean fraction of candidates eliminated by each question.

    For each Q/A pair:
      1. Map question → closest canonical question (embedding similarity)
      2. Look up the corresponding attribute key
      3. Filter the remaining candidate pool using the answer
      4. Score = fraction of pool eliminated

    The secret must be present in dataset_objects by exact name match for
    accurate filtering. If absent, scores degrade gracefully to 0 for
    questions where the filter becomes inconsistent.

    Args:
        questions:          List of yes/no questions asked.
        answers:            Corresponding YES/NO answers.
        dataset_objects:    Objects from dataset.json.
        canonical_questions: Canonical question strings from dataset.json.
        embed_model:        Sentence embedding model.

    Returns:
        Mean information gain score in [0, 1].
    """
    if not questions or not dataset_objects:
        return 0.0

    canon_embs = embed_model.encode(
        canonical_questions, convert_to_numpy=True, normalize_embeddings=True
    )

    remaining_objects = list(dataset_objects)
    ig_scores: List[float] = []

    for question, answer in zip(questions, answers):
        n_before = len(remaining_objects)
        if n_before == 0:
            break

        q_emb = embed_model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        sims = canon_embs @ q_emb
        best_idx = int(np.argmax(sims))
        attr_key = _attribute_key_for_canonical(canonical_questions[best_idx])

        if not attr_key:
            ig_scores.append(0.0)
            continue

        answer_bool = answer.strip().upper().startswith("Y")
        new_remaining = [
            obj for obj in remaining_objects
            if obj["attributes"].get(attr_key, not answer_bool) == answer_bool
        ]

        n_after = len(new_remaining)
        ig_scores.append((n_before - n_after) / n_before)
        remaining_objects = new_remaining if new_remaining else remaining_objects

    return float(np.mean(ig_scores)) if ig_scores else 0.0


def layer2_question_quality(
    record: GameRecord,
    dataset_path: str = "dataset.json",
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[float, float, float, float, List[float]]:
    """
    Compute Layer 2 scores from a completed GameRecord.

    Sub-scores:
        semantic_relevance:  How related questions are to the secret (embedding cosine sim)
        canonical_coverage:  Fraction of questions covering distinct dimensions
        information_gain:    Mean fraction of candidates eliminated per question

    Composite: relevance 30%, coverage 30%, IG 40%

    Args:
        record:           Completed GameRecord.
        dataset_path:     Path to dataset.json.
        embed_model_name: Sentence-transformers model name.

    Returns:
        (semantic_relevance, canonical_coverage, information_gain,
         layer2_score, diversity_trace)
    """
    if not record.questions:
        return 0.0, 0.0, 0.0, 0.0, []

    embed_model = _get_embed_model(embed_model_name)

    sem_rel   = _semantic_relevance(record.questions, record.secret, embed_model)
    div_trace = _diversity_trace(record.questions, embed_model)

    try:
        objects, canonicals = _load_dataset(dataset_path)
        coverage = _canonical_coverage(record.questions, canonicals, embed_model)
        ig_score = _information_gain_score(
            record.questions, record.answers, objects, canonicals, embed_model
        )
    except FileNotFoundError:
        print(f"[Evaluator] Warning: '{dataset_path}' not found — skipping coverage & IG.")
        coverage = 0.0
        ig_score = 0.0

    layer2_score = 0.30 * sem_rel + 0.30 * coverage + 0.40 * ig_score

    return sem_rel, coverage, ig_score, layer2_score, div_trace
