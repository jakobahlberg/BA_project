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
    "dog": [
        "It is a living creature.",
        "It is an animal.",
        "It is a mammal.",
        "It is commonly kept as a pet.",
        "It is known for its loyalty and barking.",
    ],
    "cat": [
        "It is a living creature.",
        "It is an animal.",
        "It is a mammal.",
        "It is commonly kept as a pet.",
        "It is known for purring and meowing.",
    ],
    "elephant": [
        "It is a living creature.",
        "It is an animal.",
        "It is a mammal.",
        "It is much larger than a car.",
        "It has a long nose called a trunk.",
    ],
    "apple": [
        "It is not alive.",
        "You can eat it.",
        "It grows on a tree.",
        "It is a fruit.",
        "It is typically red or green.",
    ],
    "banana": [
        "It is not alive.",
        "You can eat it.",
        "It is a fruit.",
        "It is yellow when ripe.",
        "It has a long curved shape.",
    ],
    "chair": [
        "It is not alive.",
        "It is man-made.",
        "It is found indoors.",
        "It is a piece of furniture.",
        "It is used for sitting.",
    ],
    "ball": [
        "It is not alive.",
        "It is man-made.",
        "It is used for play or sport.",
        "It is spherical in shape.",
        "You can throw, kick, or bounce it.",
    ],
    "mug": [
        "It is not alive.",
        "It is man-made.",
        "It is found indoors.",
        "It is used for drinking.",
        "It has a handle on the side.",
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


# ─── Public API ───────────────────────────────────────────────────────────────

def get_hints_for_secret(secret_label: str) -> List[str]:
    """Return up to MAX_HINTS hints for the given secret label.

    Uses HINTS_BY_SECRET if available, otherwise auto-generates from dataset.json.
    """
    if secret_label in HINTS_BY_SECRET:
        return HINTS_BY_SECRET[secret_label][:config.MAX_HINTS]
    obj = _DATASET_OBJECTS.get(secret_label)
    return _hints_from_attributes(obj)
