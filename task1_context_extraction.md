Task 1 - Coreference Context Extraction

Approach
- Preprocess: normalize whitespace in each article before processing.
- Coreference: load spaCy `en_core_web_sm` with the `fastcoref` component (configures the transformer-based model once). When running the component, `resolve_text=True` substitutes pronouns/referring expressions with the head mention of each cluster, so subsequent work uses fully resolved text.
- Sentence segmentation: pass the resolved text through a lightweight blank spaCy pipeline with the `sentencizer` to avoid a second (costly) coref pass while keeping sentence boundaries.
- Context tagging: classify each resolved sentence with two keyword lexicons. Shooter cues include variants of “shooter/gunman/suspect/attacker/assailant/killer/perpetrator”; victim cues cover “victim(s)/student(s)/child(ren)/teacher/bystander” plus injury and fatality terms. Sentences are deduped while preserving order.
- Outputs: per-article JSON files under `outputs/task1/contexts/<outlet>/<article>.json` (separate victim/shooter lists) and an aggregate `outputs/task1/all_contexts.json` that also retains the fully resolved text to support downstream clustering and stats.

Challenges and notes
- Environment: `fastcoref` pulls `datasets`/`aiohttp`; the latest `aiohttp` conflicted with Python 3.9.1 typing, so `aiohttp<3.12` is pinned in `requirements.txt` for stability.
- Accuracy limits: keyword heuristics are intentionally conservative and may miss euphemisms (“gunman suspect”, “perp”) or misfire when casualty language refers to the shooter; adjust the term lists in `task1_coref_extraction.py` if outlet-specific phrasing emerges.
- Performance: the first run downloads model weights; subsequent runs reuse the cached model and are CPU-friendly for the 100-article set.
