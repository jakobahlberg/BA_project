from carbontracker.tracker import CarbonTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from fpdf import FPDF
import fitz 

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

#MAYBE ADD A PROVIDER API KEY TO CARBON TRACKER
tracker = CarbonTracker(
    epochs=1,          #
    monitor_epochs=True,
    log_dir="carbon_logs",
    verbose=2
)

SECRET = "a golden retriever"
MAX_HINTS = 5
MAX_TURNS = 20

HINTS = [
    "It is a living creature.",
    "It is an animal commonly kept as a pet.",
    "It is a mammal with four legs.",
    "It is a breed of dog.",
    "It is known for its friendly nature and golden/toasty brown coat.",
]

pdf = FPDF()
for i, hint in enumerate(HINTS):
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, f"Hint {i+1}:", ln=True)
    pdf.set_font("Helvetica", size=13)
    pdf.cell(0, 10, hint, ln=True)

pdf.output("hints.pdf")
print("hints.pdf created!")


hints_used = 0

def use_hint():
    global hints_used
    if hints_used >= MAX_HINTS:
        return "No more hints available."

    doc = fitz.open("hints.pdf")
    text = doc[hints_used].get_text().strip()
    doc.close()

    hints_used += 1
    print(f"[HINT TOOL] Hint {hints_used} revealed: {text}")
    return text



# Guesser — smaller model
guesser_model_name = "Qwen/Qwen3.5-4B"
guesser_tokenizer = AutoTokenizer.from_pretrained(guesser_model_name)
guesser_model = AutoModelForCausalLM.from_pretrained(
    guesser_model_name,
    device_map="auto",
)
print("Guesser model loaded!")

# Secret keeper — bigger, more reliable
secret_model_name = "Qwen/Qwen3.5-4B"
secret_tokenizer = AutoTokenizer.from_pretrained(secret_model_name)
secret_model = AutoModelForCausalLM.from_pretrained(
    secret_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
print("Secret keeper model loaded!")


def generate_answer(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generation_settings = {
        "max_new_tokens": 100,
        "do_sample": True,            # enable sampling for top_p & temperature
        "temperature": 0.7,
        "top_p": 0.8,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.0
    }

    outputs = model.generate(
        **inputs,
        **generation_settings
    )

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": answer})
    return answer


# ============================================================
# CELL 6 — Secret keeper functions (separate prompts)
# ============================================================

secret_yn_prompt = f"""You are playing 20 questions. The secret is: {SECRET}.
A golden retriever is a dog. A dog is a mammal. A mammal is an animal. It has fur and four legs.

The player is asking a YES or NO question.
Reply with only the word YES or NO. Nothing else."""

secret_guess_prompt = f"""You are playing 20 questions. The secret is: {SECRET}.

The player is making a guess.
If the guess means "golden retriever" reply: CORRECT
Otherwise reply: WRONG
One word only."""

def get_secret_answer(question):
    messages = [
        {"role": "system", "content": secret_yn_prompt},
        {"role": "user", "content": question}
    ]
    text = secret_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = secret_tokenizer([text], return_tensors="pt").to(secret_model.device)
    generation_settings = {
        "max_new_tokens": 100,
        "do_sample": True,            # enable sampling for top_p & temperature
        "temperature": 0.7,
        "top_p": 0.8,
        "pad_token_id": secret_tokenizer.eos_token_id,
        "repetition_penalty": 1.0
    }

    outputs = secret_model.generate(
        **inputs,
        **generation_settings
    )
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    answer = secret_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()
    print(f"[Secret raw]: {answer}")
    return "YES" if "YES" in answer else "NO"

def check_guess(guess):
    messages = [
        {"role": "system", "content": secret_guess_prompt},
        {"role": "user", "content": f"The player guesses: {guess}"}
    ]
    text = secret_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = secret_tokenizer([text], return_tensors="pt").to(secret_model.device)

    generation_settings = {
        "max_new_tokens": 100,
        "do_sample": True,            # enable sampling for top_p & temperature
        "temperature": 0.7,
        "top_p": 0.8,
        "pad_token_id": secret_tokenizer.eos_token_id,
        "repetition_penalty": 1.0
    }

    outputs = secret_model.generate(
        **inputs,
        **generation_settings
    )
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    result = secret_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()
    print(f"[Secret raw]: {result}")
    secret_words = set(SECRET.lower().split())
    guess_words = set(guess.lower().split())
    if "CORRECT" in result and bool(secret_words & guess_words):
        return "CORRECT"
    return "WRONG"


secret_system_prompt = f"""
You are the SECRET in a game of 20 Questions.

Your hidden identity is: {SECRET}.

FACTS ABOUT THE SECRET (always true):
- It is a dog.
- A dog is an animal.
- Animals are living things.
- It is not man-made.
- It is not a person.
- It is not an object.

STRICT RULES:
1. Never reveal the secret.
2. Only output exactly one of:
   YES - if the question is true
   NO - if the question is false
   CORRECT - if the guess is correct
   WRONG - if the guess is wrong
3. Answer based on the facts above and general world knowledge.
4. If guess contains "golden retriever" → CORRECT, otherwise → WRONG
5. No explanations. No extra text.
""".strip()

guesser_system_prompt = f"""You are the GUESSER in a game of 20 Questions.
Your goal: Identify the secret using at most {MAX_TURNS} turns.

You have {MAX_HINTS} hints available.

Each turn output EXACTLY ONE line in ONE of these formats:
QUESTION: <your yes/no question>
GUESS: <your specific guess>
USE_HINT

RULES:
- One line only. No extra text. No explanations.
- Start broad then narrow down based on answers.
- If a guess is wrong, ask more questions before guessing again.
- Use USE_HINT only when genuinely stuck.
- Only GUESS when highly confident.
""".strip()

secret_messages = [{"role": "system", "content": secret_system_prompt}]
guesser_messages = [
    {"role": "system", "content": guesser_system_prompt},
    {"role": "user", "content": "The game starts now. Make your first move."}  # ← add this
]

print("Prompts ready!")

print("Prompts ready!")



# ============================================================
# CELL 8 — Run the game
# ============================================================

turn = 0
game_over = False
hints_used = 0

secret_messages = [{"role": "system", "content": secret_system_prompt}]
guesser_messages = [{"role": "system", "content": guesser_system_prompt}]

def parse_guesser_output(text):
    """Force-parse the guesser output regardless of format."""
    upper = text.strip().upper()

    if "USE_HINT" in upper:
        return "hint", None

    # Already formatted correctly
    if upper.startswith("QUESTION:"):
        return "question", text.split(":", 1)[-1].strip()
    if upper.startswith("GUESS:"):
        return "guess", text.split(":", 1)[-1].strip()

    # Model skipped the prefix — detect intent from content
    stripped = text.strip()
    if stripped.endswith("?"):
        # It's clearly a question, just missing the prefix
        return "question", stripped

    # Treat anything else as a question too rather than looping forever
    return "question", stripped

tracker.epoch_start()
while turn < MAX_TURNS and not game_over:
    print(f"\n--- Turn {turn + 1} | Hints used: {hints_used}/{MAX_HINTS} ---")

    guesser_output = generate_answer(guesser_messages, guesser_model, guesser_tokenizer)
    print(f"Guesser raw: {guesser_output}")

    action, content = parse_guesser_output(guesser_output)

    # --- HINT ---
    if action == "hint":
        hint_text = use_hint()
        guesser_messages.append({
            "role": "user",
            "content": f"[HINT] {hint_text}\nNow continue — ask a question or make a guess."
        })
        continue  # doesn't count as a turn

    # --- QUESTION ---
    elif action == "question":
        print(f"Guesser asks: {content}")
        answer = get_secret_answer(content)
        print(f"Secret: {answer}")
        guesser_messages.append({
            "role": "user",
            "content": f"Turn {turn+1}: Your question: '{content}' → Answer: {answer}"
        })

    # --- GUESS ---
    elif action == "guess":
        print(f"Guesser guesses: {content}")
        result = check_guess(content)
        print(f"Secret: {result}")
        if result == "CORRECT":
            print(f"\n🎉 Guesser won in {turn+1} turns using {hints_used} hints! Answer: {SECRET}")
            game_over = True
        else:
            guesser_messages.append({
                "role": "user",
                "content": f"Turn {turn+1}: Guess '{content}' was WRONG. Keep going."
            })

    turn += 1

if not game_over:
    print(f"\n❌ Guesser failed after {MAX_TURNS} turns using {hints_used} hints. Answer was: {SECRET}")
tracker.epoch_end()
tracker.stop()

print("guesser_model_name:", guesser_model_name)
print("secret_model_name:", secret_model_name)