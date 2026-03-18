# Evaluation Metrics

A breakdown of every metric computed in `evaluate.py`, organised by layer.

---

## Layer 1 — Game Outcome

### Win Score
Binary. `1.0` if the guesser correctly identified the secret within the turn limit, `0.0` if not.

### Efficiency Score
Purely mechanical — no model involved. A tiered lookup on `turns_used`:

| Turns used | Score |
|---|---|
| ≤ 5  | 1.00 |
| ≤ 8  | 0.85 |
| ≤ 12 | 0.65 |
| ≤ 16 | 0.40 |
| ≤ 20 | 0.20 |
| Lost | 0.00 |

The tiers are intentional — a linear scale would make "won in 19 turns" almost identical to "lost", which doesn't reflect the real difference in quality.

### Secret Reliability Score
Checks what fraction of the secret keeper's raw responses were well-formed — meaning they contained at least one of `YES`, `NO`, `CORRECT`, or `WRONG`. If the model outputs a long explanation instead of a one-word answer, that response counts as invalid.

```
reliability = valid_responses / total_responses
```

This is a **format** check, not a factual accuracy check. A response of `"WRONG"` to every question counts as reliable even if factually wrong.

### Layer 1 Score (composite)
```
layer1 = 0.50 × win + 0.30 × efficiency + 0.20 × reliability
```

---

## Layer 2 — Question Quality

### Semantic Relevance
Embeds every question and the secret word into vectors using `sentence-transformers/all-MiniLM-L6-v2`. Since the vectors are L2-normalised, their dot product equals cosine similarity. Takes the mean across all questions and shifts from [-1, 1] to [0, 1]:

```
relevance = mean( (cosine_sim(Q_i, secret) + 1) / 2   for all i )
```

Measures whether the questions were topically related to the secret. General early questions like "Is it alive?" will drag the score down even for a good game.

### Canonical Coverage
For each question, finds the single closest canonical question (out of ~50 pre-defined questions in `dataset.json`) using cosine similarity. Counts how many **distinct** canonical dimensions were matched across the whole game:

```
coverage = distinct_canonicals_hit / total_questions_asked
```

A score of `1.0` means every question covered a completely new conceptual dimension. A low score means the guesser kept circling the same topic (e.g. asking five slightly different questions about whether it has fur). This replaces a raw pairwise diversity metric, which unfairly penalised good narrowing strategies where consecutive questions are naturally related.

### Diversity Per Question *(logged only, not scored)*
For question `i`, computes the cosine similarity between `Q_i` and **every individual prior question**, takes the mean, then inverts:

```
diversity[i] = 1 - mean( cosine_sim(Q_i, Q_j)  for all j < i )
```

`Q1` is always `1.0` by convention (nothing to compare against). This produces a trace over the game showing whether the guesser explored new ground or got stuck. Expected pattern for a good game: high early (broad exploration), gradually lower (narrowing in). Consistently low values indicate repetitive questions.

### Information Gain
For each Q/A pair:
1. Maps the question to its closest canonical question via embedding similarity
2. Looks up the corresponding attribute key (e.g. `is_vehicle`, `is_animal`)
3. Filters the object pool by the answer — if `YES`, keep only objects where that attribute is `True`
4. Computes the fraction of objects eliminated: `(n_before - n_after) / n_before`

Averages this fraction across all questions:

```
ig = mean( (n_before - n_after) / n_before   for all Q/A pairs )
```

Uses the 28-object pool in `dataset.json`. A discriminating question like "Is it a living thing?" against a mixed pool might eliminate ~50% of objects (IG ≈ 0.5). Vague or repeated questions eliminate very few (IG close to 0).

**Important:** the secret object should always be present in `dataset.json` for this score to be meaningful. If it is absent, the filtering can produce inconsistent results because the ground truth of the secret is not anchored in the pool.

### Layer 2 Score (composite)
```
layer2 = 0.30 × semantic_relevance + 0.30 × canonical_coverage + 0.40 × information_gain
```

---

## Layer 3 — LLM-as-a-Judge (Qwen3-8B)

A single inference call to `Qwen/Qwen3-8B` with the full game transcript. The model is given the secret, the outcome, and the transcript, and asked to score four dimensions each with a one-sentence reason and an integer 1–10. Scores are mapped to [0, 1]:

```
score_01 = (raw_score - 1) / 9
```

`enable_thinking=False` is set so the model outputs the structured scores directly without a reasoning block, keeping the call fast and parseable.

### Strategy
Did the guesser use binary search to efficiently narrow down candidates?
- `10` = optimal bisection each turn, halving the remaining space
- `1` = random or repetitive questions with no visible narrowing

### Question Quality
Were questions clear, unambiguous, and non-redundant?
- `10` = all crisp yes/no questions each covering new ground
- `1` = vague, compound, or multi-part questions

### Logical Consistency
Did the guesser stay consistent with all prior answers?
- `10` = never contradicted a prior answer or repeated a concept
- `1` = frequently contradicted known facts or asked equivalent questions

### Secret Accuracy
Did the secret keeper give factually correct YES/NO responses throughout the game?
- `10` = every answer was factually correct and internally consistent
- `1` = multiple wrong or contradictory answers given the secret

Note: this is a **factual** accuracy check, unlike the Layer 1 reliability score which only checks response format.

### Layer 3 Score (composite)
```
layer3 = mean(strategy, question_quality, logical_consistency, secret_accuracy)
```

---

## Summary (across rounds)

There is no single combined score. `summarise_results()` returns the **per-round average of every individual metric**, so you can compare models on each dimension independently:

| Field | Description |
|---|---|
| `win_score` | Fraction of rounds won |
| `efficiency_score` | Average tiered efficiency |
| `secret_reliability_score` | Average format reliability of secret keeper |
| `layer1_score` | Average Layer 1 composite |
| `semantic_relevance_score` | Average question-to-secret relevance |
| `canonical_coverage_score` | Average canonical dimension coverage |
| `information_gain_score` | Average information gain per question |
| `layer2_score` | Average Layer 2 composite |
| `llm_judge_strategy` | Average judge score — strategy |
| `llm_judge_question_quality` | Average judge score — question quality |
| `llm_judge_logical_consistency` | Average judge score — logical consistency |
| `llm_judge_secret_accuracy` | Average judge score — secret accuracy |
| `layer3_score` | Average Layer 3 composite |
