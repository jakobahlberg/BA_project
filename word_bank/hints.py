"""
word_bank/hints.py
──────────────────
Single-source hints for tool mode.

Design goal:
- No legacy fallback logic.
- No dataset-derived hint generation.
- No ambiguous code paths.

Each secret gets exactly MAX_HINTS hints from one deterministic map.
"""

from __future__ import annotations

from typing import Dict, List

import config
from word_bank.standard import SECRETS as STANDARD_SECRETS

# 3 category-level hints per category
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

# 2 secret-specific hints per secret
_SECRET_SPECIFIC_HINTS: Dict[str, List[str]] = {
    "golden retriever": ["It is a popular family pet.", "It is known for being friendly and trainable."],
    "dove": ["It is often a symbol of peace.", "It is commonly seen in parks and cities."],
    "python snake": ["It is a constrictor, not venomous.", "It can be found in tropical regions."],
    "elephant": ["It has a trunk and tusks.", "It is among the largest land animals."],
    "dolphin": ["It lives in the ocean.", "It is known for high intelligence."],
    "pizza": ["It is often topped with cheese and sauce.", "It is commonly sliced into pieces."],
    "burger": ["It is often served in a bun.", "It usually includes a patty."],
    "sushi": ["It is strongly associated with Japanese cuisine.", "It often includes rice."],
    "ice cream": ["It is served cold or frozen.", "It is usually sweet."],
    "chocolate": ["It is made from cacao/cocoa.", "It is commonly eaten as a sweet treat."],
    "milk": ["It is a dairy product.", "It is commonly used with cereal."],
    "coffee": ["It often contains caffeine.", "It is commonly served hot."],
    "orange juice": ["It is made from oranges.", "It is commonly served at breakfast."],
    "tea": ["It is made by steeping leaves or herbs.", "It can be served hot or iced."],
    "lemonade": ["It is typically made with lemon flavor.", "It is often sweet and refreshing."],
    "rose": ["It is known for fragrant blossoms.", "It often has thorns on its stem."],
    "oak tree": ["It is a large hardwood tree.", "It produces acorns."],
    "cactus": ["It is adapted to dry climates.", "It usually has spines instead of leaves."],
    "sunflower": ["It has a large yellow bloom.", "It is known for following sunlight."],
    "bamboo": ["It grows in tall hollow stalks.", "It is one of the fastest-growing plants."],
    "jacket": ["It is typically worn on the upper body.", "It is used for warmth or weather protection."],
    "boots": ["It is worn on the feet.", "It usually covers the ankle or higher."],
    "scarf": ["It is commonly worn around the neck.", "It is often used in cold weather."],
    "jeans": ["It is commonly made of denim.", "It is a very common type of pants."],
    "hat": ["It is worn on the head.", "It can be used for style or sun protection."],
    "basketball": ["It is played with a ball and hoops.", "Teams score by shooting through a basket."],
    "tennis": ["It is played with rackets.", "The game is played over a net."],
    "swimming": ["It takes place in water.", "It is both a sport and survival skill."],
    "soccer": ["It is mainly played with the feet.", "Goals are scored into a net."],
    "skiing": ["It is typically done on snow.", "It uses skis for movement."],
    "guitar": ["It is typically played by strumming or plucking strings.", "It has a neck and a body."],
    "piano": ["It is played using keys.", "It produces sound via strings and hammers."],
    "violin": ["It is played with a bow.", "It is held near the shoulder/chin."],
    "drums": ["It is played by striking surfaces.", "It is central to rhythm in music."],
    "trumpet": ["It is a brass instrument.", "It is played by buzzing into a mouthpiece."],
    "heart": ["It pumps blood through the body.", "It is essential for circulation."],
    "brain": ["It controls thoughts and body functions.", "It is part of the nervous system."],
    "hand": ["It is used for grasping and touching.", "It includes fingers and a palm."],
    "eye": ["It is used for vision.", "It detects light and color."],
    "knee": ["It connects upper and lower leg bones.", "It acts as a major bending joint."],
    "France": ["It is a country in Western Europe.", "Its capital is Paris."],
    "Japan": ["It is an island nation in East Asia.", "Its capital is Tokyo."],
    "Brazil": ["It is the largest country in South America.", "It is famous for football and the Amazon."],
    "Australia": ["It is both a country and a continent.", "Its major cities include Sydney and Melbourne."],
    "Egypt": ["It is in North Africa.", "It is famous for ancient pyramids."],
    "Paris": ["It is the capital of France.", "It is known for the Eiffel Tower."],
    "Tokyo": ["It is the capital of Japan.", "It is one of the world’s largest metro areas."],
    "New York": ["It is a major U.S. city.", "It is known for Manhattan and Times Square."],
    "London": ["It is the capital of the United Kingdom.", "It is known for the Thames and Big Ben."],
    "Sydney": ["It is a major city in Australia.", "It is known for its iconic opera house."],
    "bicycle": ["It usually has two wheels.", "It is powered by pedaling."],
    "airplane": ["It flies through the air.", "It is used for long-distance travel."],
    "submarine": ["It travels underwater.", "It can dive below the ocean surface."],
    "motorcycle": ["It is a two-wheeled motor vehicle.", "It is ridden with handlebars."],
    "train": ["It moves on rails.", "It usually consists of linked cars."],
    "Minecraft": ["It is known for block-based building.", "It is a sandbox-style game."],
    "Super Mario": ["It is centered on an iconic plumber character.", "It is one of Nintendo’s best-known franchises."],
    "Pac-Man": ["It features maze navigation and pellet collection.", "It is a classic arcade game."],
    "Tetris": ["It is based on arranging falling blocks.", "Its main goal is clearing lines."],
    "The Legend of Zelda": ["It is an adventure game series by Nintendo.", "It often features dungeons and puzzle solving."],
    "Albert Einstein": ["He is famous for the theory of relativity.", "He was a major 20th-century physicist."],
    "Cleopatra": ["She was a ruler of ancient Egypt.", "She is one of history’s most famous queens."],
    "Leonardo da Vinci": ["He was a Renaissance polymath.", "He painted the Mona Lisa."],
    "Napoleon Bonaparte": ["He was a French military and political leader.", "He became emperor of France."],
    "Marie Curie": ["She pioneered research on radioactivity.", "She won Nobel Prizes in two sciences."],
    "sofa": ["It is designed for sitting.", "It often seats multiple people."],
    "wardrobe": ["It is used to store clothing.", "It is a tall cabinet-like piece."],
    "bookshelf": ["It stores books.", "It has multiple horizontal shelves."],
    "bed": ["It is used for sleeping.", "It typically includes a mattress."],
    "dining table": ["It is used for eating meals.", "It is usually surrounded by chairs."],
    "frying pan": ["It is used on a stove for cooking.", "It has a flat surface and a handle."],
    "knife": ["It has a sharp edge for cutting.", "It is common in kitchens."],
    "cutting board": ["It protects surfaces during food prep.", "It is used under a knife while cutting."],
    "blender": ["It mixes ingredients at high speed.", "It is often used for smoothies."],
    "whisk": ["It is used to beat or mix by hand.", "It has looped wire ends."],
    "hammer": ["It is used for driving nails.", "It has a heavy striking head."],
    "screwdriver": ["It is used to turn screws.", "It has a shaped tip matched to screw heads."],
    "wrench": ["It is used to grip and turn nuts/bolts.", "It applies torque to fasteners."],
    "saw": ["It is used to cut material like wood.", "It has a toothed cutting edge."],
    "drill": ["It makes holes in materials.", "It rotates a bit at high speed."],
    "mathematics": ["It involves numbers, patterns, and logic.", "It includes topics like algebra and geometry."],
    "history": ["It studies past events and societies.", "It focuses on timelines and causes of change."],
    "chemistry": ["It studies substances and reactions.", "It includes elements, molecules, and compounds."],
    "geography": ["It studies places, regions, and environments.", "It includes maps and physical features."],
    "literature": ["It studies written works and texts.", "It includes novels, poetry, and analysis."],
    "Eiffel Tower": ["It is a famous landmark in Paris.", "It is made of iron lattice structure."],
    "pyramid": ["It has a broad base and pointed top.", "It is strongly associated with ancient Egypt."],
    "lighthouse": ["It helps guide ships near coasts.", "It usually has a bright warning light."],
    "castle": ["It is a fortified historic structure.", "It is often associated with royalty or nobility."],
    "bridge": ["It connects two points over an obstacle.", "It spans things like rivers or roads."],
    "dragon": ["It is often depicted as reptilian and powerful.", "It appears in many fantasy stories."],
    "mermaid": ["It is described as part human and part fish.", "It is associated with the sea."],
    "unicorn": ["It is horse-like with a single horn.", "It is a symbol in fantasy and folklore."],
    "phoenix": ["It is associated with fire and rebirth.", "It is said to rise from ashes."],
    "werewolf": ["It is linked to human-wolf transformation myths.", "It is often associated with the full moon."],
    "Titanic": ["It is a film based on a famous ship disaster.", "It became one of the highest-grossing films."],
    "The Lion King": ["It is centered on a young lion prince.", "It is strongly associated with Disney animation."],
    "Jurassic Park": ["It features dinosaurs brought back to life.", "It is based on a theme park concept gone wrong."],
    "Star Wars": ["It is a famous space-opera franchise.", "It includes themes of Jedi and the Force."],
    "The Wizard of Oz": ["It follows a girl’s journey to a magical land.", "It features the Yellow Brick Road."],
}


def _build_hints_by_secret() -> Dict[str, List[str]]:
    built: Dict[str, List[str]] = {}
    for entry in STANDARD_SECRETS:
        category_hints = _CATEGORY_BASE_HINTS.get(entry.category)
        specific_hints = _SECRET_SPECIFIC_HINTS.get(entry.label)

        if category_hints is None:
            raise KeyError(f"Missing category base hints for category='{entry.category}'")
        if specific_hints is None:
            raise KeyError(f"Missing secret-specific hints for secret='{entry.label}'")

        hints = list(category_hints) + list(specific_hints)
        if len(hints) != config.MAX_HINTS:
            raise ValueError(
                f"Secret '{entry.label}' has {len(hints)} hints; expected {config.MAX_HINTS}."
            )

        built[entry.label] = hints
    return built


HINTS_BY_SECRET: Dict[str, List[str]] = _build_hints_by_secret()


def get_hints_for_secret(secret_label: str) -> List[str]:
    """Return the exact hint list for the given secret label."""
    if secret_label not in HINTS_BY_SECRET:
        raise KeyError(f"No hints defined for secret '{secret_label}'")
    return HINTS_BY_SECRET[secret_label][:config.MAX_HINTS]
