from carbontracker.tracker import CarbonTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import random
from typing import Dict, List, Optional, Tuple

from evaluate import GameRecord

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

MAX_TURNS = 20
MAX_HINTS = 5


EXPERIMENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(EXPERIMENT_SEED)
print(f"Experiment seed: {EXPERIMENT_SEED}")

#MAYBE ADD A PROVIDER API KEY TO CARBON TRACKER
tracker = CarbonTracker(
    epochs=1,          #
    monitor_epochs=True,
    log_dir="carbon_logs",
    verbose=2
)

guesser_model_name = "Qwen/Qwen3-4B-Instruct-2507"
secret_model_name  = "Qwen/Qwen3-4B-Instruct-2507"
judge_model_name = "Qwen/Qwen3-8B"
# --- Guesser model ---
guesser_tokenizer = AutoTokenizer.from_pretrained(guesser_model_name)
guesser_model = AutoModelForCausalLM.from_pretrained(
    guesser_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("Guesser model loaded!")

# --- Secret-keeper model (separate instance / can be a different checkpoint) ---
secret_tokenizer = AutoTokenizer.from_pretrained(secret_model_name)
secret_model = AutoModelForCausalLM.from_pretrained(
    secret_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("Secret model loaded!")




def generate_answer(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generation_settings = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.0,
    }

    outputs = model.generate(**inputs, **generation_settings)

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    messages.append({"role": "assistant", "content": answer})
    return answer


GUESSER_SYSTEM_PROMPT_TEMPLATE = """
You are the GUESSER in a game of 20 Questions.

Your goal:
Identify the secret object/person using at most {max_turns} turns.

You are given the full history of previous questions and answers.

You may take ONE action per turn:

ACTION 1 — Ask a yes or no question
Format exactly:
QUESTION: <yes/no question>

ACTION 2 — Make a guess (only if highly confident)
Format exactly:
GUESS: <specific object/person>

ACTION 3 — Use a hint (when stuck)
Format exactly:
USE_HINT

Hints available: {max_hints}
Hints are valuable and should be used early when uncertain.
If you are still unsure after ~4 questions, consider using USE_HINT.

STRICT RULES:

- First reduce the possibilities using broad questions.
- Start general (animal, object, person) then narrow down.
- Do not guess early.
- Do not repeat previous questions.
- Do not repeat previous guesses.
- Guesses must be extremely specific.
- Questions must be unambiguous YES/NO only (no multi-part or subjective questions).
- NEVER embed a guess inside a QUESTION. "QUESTION: Is it a dog?" is FORBIDDEN.
- If you are confident enough to name the specific thing, you MUST use GUESS (not QUESTION).
- When a category is confirmed (e.g., DOG), do not GUESS the generic category unless the secret is actually that generic term. Prefer a specific instance (breed/type).
- QUESTION is only for yes/no questions that gather information.
- GUESS is the only way to win the game.
- If evidence strongly supports a category and a guess within that category is wrong, do NOT switch categories; stay in that category and refine using distinguishing sub-features (type, size, color, habitat, function).
- Never ask or guess a category that contradicts a confirmed YES answer (e.g., if DOG=YES, do not ask/guess CAT).
- After a wrong guess, restate the strongest confirmed facts in your next question and refine based on them.
- Output exactly one line.
- No explanations.
- No extra text.
- Follow format exactly.

If a guess was WRONG, change strategy and ask a new question.
""".strip()

HINTS_BY_SECRET = {
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


def parse_guesser_output(text: str) -> Tuple[str, Optional[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    upper_first = first.upper()

    if upper_first == "USE_HINT" or "USE_HINT" in upper_first:
        return "hint", None
    if upper_first.startswith("QUESTION:"):
        return "question", first.split(":", 1)[-1].strip()
    if upper_first.startswith("GUESS:"):
        return "guess", first.split(":", 1)[-1].strip()

    # If the model emitted analysis first but includes a valid action line later, use it.
    for ln in lines[1:]:
        up = ln.upper()
        if up.startswith("GUESS:"):
            return "guess", ln.split(":", 1)[-1].strip()
        if up.startswith("QUESTION:"):
            return "question", ln.split(":", 1)[-1].strip()
        if up == "USE_HINT" or "USE_HINT" in up:
            return "hint", None

    if first.endswith("?"):
        return "question", first
    return "question", first



def _load_dataset_objects(path: str = "dataset.json") -> Dict[str, Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        objects = data.get("objects", [])
        return {obj.get("name", ""): obj for obj in objects if obj.get("name")}
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
    if attrs.get("is_building"):
        hints.append("It is a building or structure.")

    if attrs.get("has_fur"):
        hints.append("It has fur or hair.")
    if attrs.get("has_feathers"):
        hints.append("It has feathers.")
    if attrs.get("has_scales"):
        hints.append("It has scales.")
    if attrs.get("can_fly"):
        hints.append("It can fly.")
    if attrs.get("can_hold_in_hand"):
        hints.append("It can be held in one hand.")

    seen: Dict[str, bool] = {}
    deduped = []
    for h in hints:
        if not seen.get(h):
            deduped.append(h)
            seen[h] = True
    return deduped[:MAX_HINTS]


def get_hints_for_secret(secret_label: str) -> List[str]:
    if secret_label in HINTS_BY_SECRET:
        return HINTS_BY_SECRET[secret_label][:MAX_HINTS]
    obj = _DATASET_OBJECTS.get(secret_label)
    return _hints_from_attributes(obj)



    q = question.strip().lower()
    s = secret_label.strip().lower()
    if not q or not s:
        return False
    # Only treat direct identity questions as guesses
    if not (q.startswith("is it") or q.startswith("is this") or q.startswith("is that")):
        return False
    return s in q
def play_round(secret_system_prompt: str, round_number: int, secret_label: str = "") -> GameRecord:
    """
    Run one game of 20 Questions and return a fully-populated GameRecord
    that can be passed directly to evaluate.evaluate_game().
    """
    turn = 0
    game_over = False

    # --- Tracking lists for evaluation ---
    questions: list[str] = []
    answers: list[str] = []
    guesses: list[str] = []
    secret_raw_responses: list[str] = []
    final_guess: str = ""

    print(f"\n=== ROUND {round_number} START ===")

    hints = get_hints_for_secret(secret_label)
    max_hints = len(hints)
    hints_used = 0
    questions_since_hint = 0

    def use_hint() -> str:
        nonlocal hints_used
        if hints_used >= max_hints:
            return "No more hints available."
        hint_text = hints[hints_used]
        hints_used += 1
        print(f"[HINT TOOL] Hint {hints_used} revealed: {hint_text}")
        return hint_text

    guesser_system_prompt = GUESSER_SYSTEM_PROMPT_TEMPLATE.format(
        max_turns=MAX_TURNS,
        max_hints=max_hints,
    )

    guesser_messages = [
        {"role": "system", "content": guesser_system_prompt},
        {"role": "user", "content": "Start the game."}
    ]

    secret_messages = [
        {"role": "system", "content": secret_system_prompt},
        {"role": "user", "content": "Awaiting first question."}
    ]

    while turn < MAX_TURNS and not game_over:

        guesser_output = generate_answer(guesser_messages, guesser_model, guesser_tokenizer)
        print("Guesser:", guesser_output)

        action, content = parse_guesser_output(guesser_output)

        if action == "hint":
            hint_text = use_hint()
            questions_since_hint = 0
            guesser_messages.append({
                "role": "user",
                "content": f"[HINT] {hint_text}\nNow continue — ask a question or make a guess."
            })
            continue

        if action == "question":
            question = content or ""
            questions.append(question)
            questions_since_hint += 1

            secret_messages.append({"role": "user", "content": question})
            answer = generate_answer(secret_messages, secret_model, secret_tokenizer)
            secret_raw_responses.append(answer)

            # Normalise to YES/NO for the record
            normalised = "YES" if "YES" in answer.upper() else "NO"
            answers.append(normalised)

            print("Secret:", answer)

            guesser_messages.append({
                "role": "user",
                "content": f"""
            Turn {turn} result:
            Your question: {question}
            Secret answered: {answer}
            """
            })

        elif action == "guess":
            guess = content or ""
            guesses.append(guess)
            final_guess = guess

            secret_messages.append({
                "role": "user",
                "content": f"My guess is: {guess}"
            })

            result = generate_answer(secret_messages, secret_model, secret_tokenizer)
            secret_raw_responses.append(result)
            print("Secret:", result)

            if "CORRECT" in result.strip().upper():
                print("Guesser won!")
                game_over = True
            else:
                questions_since_hint += 1
                guesser_messages.append({
                    "role": "user",
                    "content": f"""
                Turn {turn} result:
                Your guess: {guess}
                Secret response: WRONG

                This guess was incorrect.
                Do not repeat it.
                Continue reasoning and ask a new question.
                """
                })

        if hints_used < max_hints and questions_since_hint >= 4:
            guesser_messages.append({
                "role": "user",
                "content": "Reminder: You may use USE_HINT now if still uncertain."
            })

        turn += 1

    if not game_over:
        print("Guesser failed after 20 turns")
        if guesses:
            final_guess = guesses[-1]

    return GameRecord(
        secret=secret_label,
        questions=questions,
        answers=answers,
        guesses=guesses,
        final_guess=final_guess,
        was_correct=game_over,
        turns_used=turn,
        secret_raw_responses=secret_raw_responses,
        max_turns=MAX_TURNS,
    )


from evaluate import evaluate_game, summarise_results, load_judge_model

# Labels must match the "name" field in dataset.json for information-gain to work
secret_labels = [
    "golden retriever", "dove", "python snake",
    "pizza", "milk", "burger",
    "painting", "car", "door",
]

secrets_and_prompts = [
    # Animals (first 3)
    """You are playing 20 questions. The secret is: golden retriever.

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
    """You are playing 20 questions. The secret is: dove.

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
    """You are playing 20 questions. The secret is: python snake.

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

    # Foods (next 3)
    """You are playing 20 questions. The secret is: pizza.

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
    """You are playing 20 questions. The secret is: milk.

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
    """You are playing 20 questions. The secret is: burger.

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

    # Objects (last 3)
    """You are playing 20 questions. The secret is: painting.

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
    """You are playing 20 questions. The secret is: car.

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
    """You are playing 20 questions. The secret is: door.

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
]


secret_categories = [
    "animal", "animal", "animal",
    "food", "food", "food",
    "object", "object", "object",
]


# Pre-load the judge model once so it's shared across all rounds
load_judge_model(judge_model_name)

tracker.epoch_start()

total_turns = 0
total_correct = 0

category_stats = {
    "animal": {"rounds": 0, "turns": 0, "correct": 0},
    "food": {"rounds": 0, "turns": 0, "correct": 0},
    "object": {"rounds": 0, "turns": 0, "correct": 0},
}

eval_results = []

for i, (secret_prompt, label) in enumerate(zip(secrets_and_prompts, secret_labels), start=1):
    record = play_round(secret_prompt, round_number=i, secret_label=label)

    total_turns += record.turns_used
    if record.was_correct:
        total_correct += 1

    category = secret_categories[i - 1]
    stats = category_stats[category]
    stats["rounds"] += 1
    stats["turns"] += record.turns_used
    if record.was_correct:
        stats["correct"] += 1

    # --- Evaluate this round ---
    result = evaluate_game(
        record,
        dataset_path="dataset.json",
        judge_model_name=judge_model_name,  # reuse the model loaded above
        run_judge=True,
    )
    eval_results.append(result)
    print(result)

tracker.epoch_end()
tracker.stop()

total_rounds = len(secrets_and_prompts)
overall_avg_turns = total_turns / total_rounds if total_rounds > 0 else 0.0

print("\n=== SUMMARY ===")
print(f"Total rounds: {total_rounds}")
print(f"Total correct: {total_correct}")
print(f"Overall average turns per round: {overall_avg_turns:.2f}")

for cat_label, stats in category_stats.items():
    if stats["rounds"] > 0:
        avg_turns = stats["turns"] / stats["rounds"]
        print(f"\nCategory: {cat_label}s")
        print(f"  Rounds: {stats['rounds']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Average turns per round: {avg_turns:.2f}")

print("\n=== EVALUATION SUMMARY (all rounds) ===")
summary = summarise_results(eval_results)
for k, v in summary.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nguesser_model_name:", guesser_model_name)
print("secret_model_name: ", secret_model_name)
print("experiment_seed:", EXPERIMENT_SEED)