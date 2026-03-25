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
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A golden retriever IS a dog. A dog IS a mammal. A mammal IS an animal.
So: "is it an animal" = YES. "is it a mammal" = YES. "is it a dog" = YES.
If the player guesses "golden retriever": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="dove",
        category="animal",
        system_prompt="""You are playing 20 questions. The secret is: dove.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A dove IS a bird. A bird IS an animal.
So: "is it an animal" = YES. "is it a bird" = YES.
If the player guesses "dove": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="python snake",
        category="animal",
        system_prompt="""You are playing 20 questions. The secret is: python snake.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A python snake IS a reptile. A reptile IS an animal.
So: "is it an animal" = YES. "is it a reptile" = YES.
If the player guesses "python snake": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),

    # ── Foods ──────────────────────────────────────────────────────────────
    SecretEntry(
        label="pizza",
        category="food",
        system_prompt="""You are playing 20 questions. The secret is: pizza.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
Pizza IS food. Food IS something you can eat.
So: "is it food" = YES. "is it something you eat" = YES.
If the player guesses "pizza": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="milk",
        category="food",
        system_prompt="""You are playing 20 questions. The secret is: milk.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
Milk IS food. Milk IS a drink. Food IS something you can eat or drink.
So: "is it food" = YES. "is it a drink" = YES.
If the player guesses "milk": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="burger",
        category="food",
        system_prompt="""You are playing 20 questions. The secret is: burger.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A burger IS food. Food IS something you can eat.
So: "is it food" = YES. "is it something you eat" = YES.
If the player guesses "burger": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),

    # ── Objects ────────────────────────────────────────────────────────────
    SecretEntry(
        label="painting",
        category="object",
        system_prompt="""You are playing 20 questions. The secret is: painting.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A painting IS a work of art. A work of art IS a man-made object.
So: "is it a work of art" = YES. "is it a painting" = YES. "is it a man-made object" = YES.
If the player guesses "painting": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="car",
        category="object",
        system_prompt="""You are playing 20 questions. The secret is: car.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A car IS a vehicle. A vehicle IS a man-made object.
So: "is it a vehicle" = YES. "is it a car" = YES. "is it man-made" = YES.
If the player guesses "car": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
    SecretEntry(
        label="door",
        category="object",
        system_prompt="""You are playing 20 questions. The secret is: door.

Reply with only one word.
If the message starts with "My guess is:" and the guess is the secret: reply CORRECT.
If the message starts with "My guess is:" and the guess is anything else: reply WRONG.
If the message does NOT start with "My guess is:", you MUST answer ONLY YES or NO.
If asked a yes/no question (including "Is it a <secret>?"), reply YES or NO.
A door IS part of a building. A building IS a structure. A structure IS a man-made object.
So: "is it part of a building" = YES. "is it a man-made object" = YES.
If the player guesses "door": reply CORRECT.
Otherwise: reply WRONG.
One word only. No punctuation. No explanation.""",
    ),
]
