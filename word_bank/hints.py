"""
word_bank/hints.py
──────────────────
Hint lookup for hint mode.

get_hints_for_secret(label) first checks HINTS_BY_SECRET for handcrafted hints,
then falls back to auto-generating hints from dataset.json attributes.
"""

from __future__ import annotations

import json
from typing import Dict, List

import config
from word_bank.standard import SECRETS as STANDARD_SECRETS


# ─── Handcrafted hints ────────────────────────────────────────────────────────

HINTS_BY_SECRET: Dict[str, List[str]] = {
    "golden retriever": [
        "It is a living creature.",
        "It is an animal commonly kept as a pet.",
        "It is a mammal with four legs.",
        "It is a breed of dog.",
        "It is known for its friendly nature and golden/toasty brown coat.",
    ],
    "dove": [
        "It is a living creature.",
        "It is an animal that can fly.",
        "It is a bird.",
        "It is often associated with peace.",
        "It is typically white or light gray.",
    ],
    "python snake": [
        "It is a living creature.",
        "It is an animal without legs.",
        "It is a reptile.",
        "It is a type of snake.",
        "It is known for constricting prey.",
    ],
    "pizza": [
        "It is not a living creature.",
        "It is food you can eat.",
        "It is usually round and sliced.",
        "It often has cheese and tomato sauce.",
        "It is baked.",
    ],
    "milk": [
        "It is not a living creature.",
        "It is something you can drink.",
        "It is a common dairy product.",
        "It is white or off-white.",
        "It is often used with cereal.",
    ],
    "burger": [
        "It is not a living creature.",
        "It is food you can eat.",
        "It is typically served in a bun.",
        "It often contains a patty.",
        "It is commonly eaten as fast food.",
    ],
    "painting": [
        "It is not a living creature.",
        "It is a man-made object.",
        "It is a work of art.",
        "It is typically visual and flat.",
        "It is often displayed on a wall.",
    ],
    "car": [
        "It is not a living creature.",
        "It is a man-made object.",
        "It is a vehicle.",
        "It typically has four wheels.",
        "It is used for transportation.",
    ],
    "door": [
        "It is not a living creature.",
        "It is a man-made object.",
        "It is part of a building.",
        "It can open and close.",
        "It allows people to enter or exit rooms.",
    ],
}


# ─── Auto-generation from dataset.json ───────────────────────────────────────

def _load_dataset_objects(path: str = "dataset.json") -> Dict[str, Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return {obj["name"]: obj for obj in data.get("objects", []) if obj.get("name")}
    except Exception:
        return {}


_DATASET_OBJECTS = _load_dataset_objects()
_STANDARD_CATEGORIES = {entry.label: entry.category for entry in STANDARD_SECRETS}


_CATEGORY_BASE_HINTS: Dict[str, List[str]] = {
    "animal": [
        "It is a living creature.",
        "It belongs to the animal kingdom.",
        "It is found in nature (wild or domesticated).",
    ],
    "food": [
        "It is not a living creature.",
        "It is something you can eat.",
        "It is commonly consumed as part of a meal or snack.",
    ],
    "drink": [
        "It is not a living creature.",
        "It is something you can drink.",
        "It is consumed as a beverage.",
    ],
    "plant": [
        "It is a living organism.",
        "It belongs to the plant kingdom.",
        "It is associated with nature and growth.",
    ],
    "clothing": [
        "It is not a living creature.",
        "It is something people wear.",
        "It is used as part of personal attire.",
    ],
    "sport": [
        "It is not a living creature.",
        "It is an activity people do.",
        "It is associated with physical play or competition.",
    ],
    "musical instrument": [
        "It is not a living creature.",
        "It is related to music.",
        "It is used to produce sound intentionally.",
    ],
    "body part": [
        "It is associated with a living body.",
        "It is part of human anatomy.",
        "It has a biological function.",
    ],
    "country": [
        "It is not a living creature.",
        "It is a geographical/political place.",
        "It is a sovereign nation.",
    ],
    "city": [
        "It is not a living creature.",
        "It is a place people can live in.",
        "It is an urban location.",
    ],
    "vehicle": [
        "It is not a living creature.",
        "It is used for transportation.",
        "It helps move people or goods from one place to another.",
    ],
    "video game": [
        "It is not a living creature.",
        "It is related to digital entertainment.",
        "It is something people play.",
    ],
    "famous person": [
        "It is a real human being.",
        "It is a person known publicly.",
        "It is associated with history, culture, or achievements.",
    ],
    "furniture": [
        "It is not a living creature.",
        "It is a household object.",
        "It is commonly found inside homes or buildings.",
    ],
    "kitchenware": [
        "It is not a living creature.",
        "It is used in a kitchen.",
        "It helps with food preparation or cooking.",
    ],
    "tool": [
        "It is not a living creature.",
        "It is used to perform practical tasks.",
        "It helps with building, fixing, or shaping things.",
    ],
    "school subject": [
        "It is not a living creature.",
        "It is related to education.",
        "It is taught in schools.",
    ],
    "architecture": [
        "It is not a living creature.",
        "It is a built structure or landmark.",
        "It is associated with construction/design.",
    ],
    "mythical creature": [
        "It is not a real living creature.",
        "It appears in myths, legends, or fantasy.",
        "It is imagined rather than biologically real.",
    ],
    "movie": [
        "It is not a living creature.",
        "It is a form of storytelling/entertainment.",
        "It is something people watch.",
    ],
}


def _hints_from_attributes(obj: Dict) -> List[str]:
    attrs = obj.get("attributes", {}) if obj else {}
    hints: List[str] = []

    if attrs.get("is_alive") or attrs.get("is_animal"):
        hints.append("It is a living creature.")
    else:
        hints.append("It is not a living creature.")

    category = obj.get("category")
    if category == "animal":
        hints.append("It is an animal.")
    elif category == "food":
        hints.append("It is food you can eat.")
    elif category == "object":
        hints.append("It is a man-made object.")

    if attrs.get("is_dog"):
        hints.append("It is a type of dog.")
    if attrs.get("is_cat"):
        hints.append("It is a type of cat.")
    if attrs.get("is_bird"):
        hints.append("It is a bird.")
    if attrs.get("is_reptile"):
        hints.append("It is a reptile.")
    if attrs.get("is_fish_or_sea_creature"):
        hints.append("It lives in water.")
    if attrs.get("is_drink"):
        hints.append("It is something you can drink.")
    if attrs.get("is_vehicle"):
        hints.append("It is a vehicle.")
    if attrs.get("is_art"):
        hints.append("It is a work of art.")
    if attrs.get("can_fly"):
        hints.append("It can fly.")
    if attrs.get("has_fur"):
        hints.append("It has fur or hair.")
    if attrs.get("has_feathers"):
        hints.append("It has feathers.")
    if attrs.get("has_scales"):
        hints.append("It has scales.")
    if attrs.get("can_hold_in_hand"):
        hints.append("It can be held in one hand.")

    seen: set = set()
    deduped = [h for h in hints if not (h in seen or seen.add(h))]
    return deduped[:config.MAX_HINTS]


# ─── Generic fallbacks for full coverage ─────────────────────────────────────

def _label_shape_hints(secret_label: str) -> List[str]:
    words = [w for w in secret_label.split() if w]
    first = words[0][0].upper() if words else "?"
    hints = [
        f"It contains {len(words)} word(s).",
        f"It starts with the letter '{first}'.",
    ]
    if "-" in secret_label:
        hints.append("Its name includes a hyphen.")
    return hints


def _category_fallback_hints(secret_label: str, category: str) -> List[str]:
    base = _CATEGORY_BASE_HINTS.get(
        category,
        [
            "It is not a living creature.",
            "It belongs to a recognizable category.",
            "It has a specific proper name.",
        ],
    )
    hints = list(base) + _label_shape_hints(secret_label)
    seen: set = set()
    deduped = [h for h in hints if not (h in seen or seen.add(h))]
    return deduped[:config.MAX_HINTS]


# ─── Public API ───────────────────────────────────────────────────────────────

def get_hints_for_secret(secret_label: str) -> List[str]:
    """Return up to MAX_HINTS hints for the given secret label.

    Uses HINTS_BY_SECRET if available, otherwise auto-generates from dataset.json.
    """
    if secret_label in HINTS_BY_SECRET:
        return HINTS_BY_SECRET[secret_label][:config.MAX_HINTS]
    obj = _DATASET_OBJECTS.get(secret_label)
    category = _STANDARD_CATEGORIES.get(secret_label) or (obj or {}).get("category", "")

    # dataset.json is still from a smaller taxonomy; for new categories, prefer
    # our category-based hint templates to avoid misleading generic attributes.
    use_dataset_attrs = category in {"animal", "food", "drink", "plant"} and obj is not None

    if use_dataset_attrs:
        hints = _hints_from_attributes(obj)
        if len(hints) < config.MAX_HINTS:
            hints.extend(_category_fallback_hints(secret_label, category))
        seen: set = set()
        deduped = [h for h in hints if not (h in seen or seen.add(h))]
        return deduped[:config.MAX_HINTS]

    return _category_fallback_hints(secret_label, category)
