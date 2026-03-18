"""
secrets/standard.py
───────────────────
Standard difficulty secrets (9 rounds: 3 animals, 3 foods, 3 objects).

Labels match the "name" field in dataset.json for full information-gain scoring.
"""

from word_bank import SecretEntry

SECRETS = [
    # ── Animals ────────────────────────────────────────────────────────────
    SecretEntry(
        label="golden retriever",
        category="animal",
        system_prompt="""You are playing 20 questions. The secret is: golden retriever.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "golden retriever": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a golden retriever?"): reply YES or NO.
A golden retriever IS a dog. A dog IS a mammal. A mammal IS an animal.
So: "is it an animal" = YES. "is it a mammal" = YES. "is it a dog" = YES. "is it a golden retriever" = YES.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="dove",
        category="animal",
        system_prompt="""You are playing 20 questions. The secret is: dove.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "dove": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a dove?"): reply YES or NO.
A dove IS a bird. A bird IS an animal.
So: "is it an animal" = YES. "is it a bird" = YES. "is it a dove" = YES.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="python",
        category="animal",
        system_prompt="""You are playing 20 questions. The secret is: python.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "python": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a python?"): reply YES or NO.
A python IS a snake. A snake IS a reptile. A reptile IS an animal.
So: "is it an animal" = YES. "is it a reptile" = YES. "is it a snake" = YES. "is it a python" = YES.
One word only. No punctuation. No explanation.""",
    ),

    # ── Foods ──────────────────────────────────────────────────────────────
    SecretEntry(
        label="pizza",
        category="food",
        system_prompt="""You are playing 20 questions. The secret is: pizza.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "pizza": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it pizza?"): reply YES or NO.
Pizza IS food. Food IS something you can eat.
So: "is it food" = YES. "is it something you eat" = YES. "is it pizza" = YES.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="milk",
        category="food",
        system_prompt="""You are playing 20 questions. The secret is: milk.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "milk": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it milk?"): reply YES or NO.
Milk IS food. Milk IS a drink. Food IS something you can eat or drink.
So: "is it food" = YES. "is it a drink" = YES. "is it milk" = YES.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="burger",
        category="food",
        system_prompt="""You are playing 20 questions. The secret is: burger.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "burger": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a burger?"): reply YES or NO.
A burger IS food. Food IS something you can eat.
So: "is it food" = YES. "is it something you eat" = YES. "is it a burger" = YES.
One word only. No punctuation. No explanation.""",
    ),

    # ── Objects ────────────────────────────────────────────────────────────
    SecretEntry(
        label="painting",
        category="object",
        system_prompt="""You are playing 20 questions. The secret is: painting.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "painting": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a painting?"): reply YES or NO.
A painting IS a work of art. A work of art IS a man-made object.
So: "is it a work of art" = YES. "is it a painting" = YES. "is it a man-made object" = YES.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="car",
        category="object",
        system_prompt="""You are playing 20 questions. The secret is: car.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "car": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a car?"): reply YES or NO.
A car IS a vehicle. A vehicle IS a man-made object.
So: "is it a vehicle" = YES. "is it a car" = YES. "is it man-made" = YES.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="door",
        category="object",
        system_prompt="""You are playing 20 questions. The secret is: door.

Reply with only one word.
If the message starts with "My guess is:" and the guess is "door": reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If asked a yes/no question (including "Is it a door?"): reply YES or NO.
A door IS part of a building. A building IS a structure. A structure IS a man-made object.
So: "is it part of a building" = YES. "is it a man-made object" = YES. "is it a door" = YES.
One word only. No punctuation. No explanation.""",
    ),
]
