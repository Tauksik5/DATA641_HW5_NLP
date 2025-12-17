"""
Context extraction for gun violence articles (Task 1).

This script runs coreference resolution to replace pronouns with entity
mentions, splits the resolved text into sentences, and collects victim- and
shooter-related contexts for each article.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import spacy
import fastcoref.spacy_component  # registers the "fastcoref" factory
from tqdm import tqdm


# Terms used to flag shooter- and victim-related sentences.
SHOOTER_TERMS = {
    "shooter",
    "gunman",
    "gunmen",
    "gunwoman",
    "suspect",
    "attacker",
    "assailant",
    "killer",
    "perpetrator",
}

VICTIM_TERMS = {
    "victim",
    "victims",
    "student",
    "students",
    "child",
    "children",
    "teacher",
    "bystander",
    "wounded",
    "injured",
    "killed",
    "dead",
    "fatalities",
    "hospitalized",
}

DATA_DIRS = {
    "cnn": Path("cnn_five_para"),
    "fox": Path("FOX_five_para"),
    "nyt": Path("NYT_five_para"),
    "wsj": Path("WSJ_five_para"),
}


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    # Keep first occurrence of each string in order
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _has_term(text: str, terms: Iterable[str]) -> bool:
    # Check if any term appears as a whole word
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _normalize_text(text: str) -> str:
    # Collapse whitespace to single spaces
    return re.sub(r"\s+", " ", text).strip()


def _load_spacy_model(device: Optional[str] = None) -> spacy.language.Language:
    """
    Load spaCy with the fastcoref component. Downloads the small English model
    if it is missing.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    if "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe("fastcoref", config={"device": device})
    return nlp


def _load_sentence_splitter() -> spacy.language.Language:
    # Lightweight sentence splitter
    splitter = spacy.blank("en")
    splitter.add_pipe("sentencizer")
    return splitter


def resolve_and_split(
    text: str, nlp: spacy.language.Language, splitter: spacy.language.Language
) -> Tuple[str, List[str]]:
    """
    Resolve coreferences and return the resolved text plus sentence strings.
    """
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    resolved_text = doc._.resolved_text if doc._.resolved_text else doc.text
    sent_doc = splitter(resolved_text)
    sentences = [s.text.strip() for s in sent_doc.sents if s.text.strip()]
    return resolved_text, sentences


def classify_sentences(sentences: List[str]) -> Tuple[List[str], List[str]]:
    # Separate shooter vs victim sentences by keyword hit
    shooter_sentences: List[str] = []
    victim_sentences: List[str] = []
    for sent in sentences:
        lowered = sent.lower()
        if _has_term(lowered, SHOOTER_TERMS):
            shooter_sentences.append(sent)
        if _has_term(lowered, VICTIM_TERMS):
            victim_sentences.append(sent)
    return _dedupe_preserve_order(shooter_sentences), _dedupe_preserve_order(
        victim_sentences
    )


def process_article(
    path: Path, nlp: spacy.language.Language, splitter: spacy.language.Language
) -> Dict:
    # Resolve coref, split sentences, and classify contexts for one article
    text = _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    resolved_text, sentences = resolve_and_split(text, nlp, splitter)
    shooter_ctx, victim_ctx = classify_sentences(sentences)
    return {
        "source_path": str(path),
        "resolved_text": resolved_text,
        "shooter_contexts": shooter_ctx,
        "victim_contexts": victim_ctx,
    }


def collect_articles(device: Optional[str] = None, output_dir: Optional[Path] = None):
    # Process all outlets and write per-article and aggregate JSONs
    nlp = _load_spacy_model(device=device)
    splitter = _load_sentence_splitter()

    if output_dir is None:
        output_dir = Path("outputs") / "task1"
    per_article_dir = output_dir / "contexts"
    per_article_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    for outlet, directory in DATA_DIRS.items():
        if not directory.exists():
            print(f"Warning: missing directory {directory}, skipping.")
            continue
        for path in tqdm(sorted(directory.glob("*.txt")), desc=f"{outlet} articles"):
            record = process_article(path, nlp, splitter)
            record["outlet"] = outlet
            record["article"] = path.stem
            all_records.append(record)

            outlet_dir = per_article_dir / outlet
            outlet_dir.mkdir(parents=True, exist_ok=True)
            with (outlet_dir / f"{path.stem}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "outlet": outlet,
                        "article": path.stem,
                        "shooter_contexts": record["shooter_contexts"],
                        "victim_contexts": record["victim_contexts"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    with (output_dir / "all_contexts.json").open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)


def main():
    # CLI entrypoint
    parser = argparse.ArgumentParser(description="Extract victim/shooter contexts.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device for coref model (e.g., 'cpu' or 'cuda'). Defaults to auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "task1",
        help="Directory to write context JSON outputs.",
    )
    args = parser.parse_args()
    collect_articles(device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
