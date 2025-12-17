Task 2 - Description Extraction

Approach
- Input: Task 1’s resolved contexts (`outputs/task1/all_contexts.json`) already replace pronouns with entity mentions, which lets dependency parsing focus on explicit shooter/victim tokens.
- Parsing: load `en_core_web_sm` and parse each context sentence. Targets are matched with curated shooter/victim lexicons (same as Task 1, with plural “suspects” added).
- Descriptive spans per target:
  - Noun chunk containing the target (captures determiners and immediate modifiers, e.g., “armed suspect”, “two children”).
  - Adjectival/numeric/compound modifiers attached to the target (`amod`, `nummod`, `compound`), concatenated with the head.
  - Governing verb or adjective phrase for the target (advmod/neg + head lemma + particle) when the head is a content verb/adj (skipping light verbs like be/have/do/say/tell/get).
  - Appositions and relative/participial clauses (`appos`, `relcl`, `acl`) spanning the child subtree.
- Outputs: per-article JSON under `outputs/task2/descriptions/<outlet>/<article>.json` with sentence-level descriptions plus deduped phrase lists. Aggregate file: `outputs/task2/all_descriptions.json`.

Strengths
- Dependency-based features stay close to the entity mention, yielding compact, interpretable phrases without model hallucination.
- Using resolved text reduces pronoun ambiguity; noun chunks keep enough local context for interpretability.
- Lightweight (spaCy small model) and deterministic—good for reproducibility and grading.

Limitations
- Keyword targeting misses euphemisms or outlet-specific labels not in the lexicons; adding terms in `task2_description_extraction.py` can broaden coverage.
- Light-verb filtering removes “be” phrases; some meaningful copular descriptions (“is armed”) may be dropped if not captured via modifiers.
- Verb lemmas sacrifice tense/aspect nuances, and short verb phrases like “leave”/“include” may be less descriptive; further filtering or sentiment scoring could refine this.
- No LLM or sentiment classifier is used here; emotional tone beyond explicit modifiers is not captured. This keeps runtime small but may under-represent evaluative language.
