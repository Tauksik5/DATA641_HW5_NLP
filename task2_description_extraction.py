"""
Task 2: Extract descriptive phrases for shooter/victim mentions.

Reads Task 1 outputs (all_contexts.json) and uses spaCy dependency parsing to
pull adjectival modifiers, noun chunks, and governing verb/adjective phrases
connected to shooter/victim mentions.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import spacy

SHOOTER_TERMS = {
    "shooter",
    "gunman",
    "gunmen",
    "gunwoman",
    "suspect",
    "suspects",
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

STOP_VERBS = {"be", "have", "do", "say", "tell", "get"}


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    # Keep first occurrence of each string in order
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _clean(text: str) -> str:
    # Normalize whitespace inside phrases
    return " ".join(text.split())


def _load_spacy() -> spacy.language.Language:
    # Load or download spaCy English model
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def _extract_from_sentence(
    sentence: str, targets: Set[str], nlp: spacy.language.Language
) -> List[Dict]:
    # Pull noun chunks/modifiers/verbs tied to target tokens in one sentence
    doc = nlp(sentence)
    noun_chunks = list(doc.noun_chunks)

    def chunk_for_token(tok: spacy.tokens.Token) -> Optional[spacy.tokens.Span]:
        for nc in noun_chunks:
            if nc.start <= tok.i < nc.end:
                return nc
        return None

    results: List[Dict] = []
    for tok in doc:
        if tok.text.lower() not in targets:
            continue
        phrases: List[str] = []

        chunk = chunk_for_token(tok)
        if chunk:
            phrases.append(_clean(chunk.text))

        modifiers = [
            child.text
            for child in sorted(
                [c for c in tok.children if c.dep_ in {"amod", "compound", "nummod"}],
                key=lambda c: c.i,
            )
        ]
        if modifiers:
            phrases.append(_clean(" ".join(modifiers + [tok.text])))

        head = tok.head
        if head.pos_ in {"VERB", "AUX"}:
            if head.lemma_ in STOP_VERBS:
                verb_phrase = ""
            else:
                advmods = [
                    child.text
                    for child in sorted(
                        [c for c in head.children if c.dep_ in {"advmod", "neg"}],
                        key=lambda c: c.i,
                    )
                ]
                particles = [c.text for c in head.children if c.dep_ == "prt"]
                verb_phrase = _clean(" ".join(advmods + [head.lemma_] + particles))
            if verb_phrase:
                phrases.append(verb_phrase)
        elif head.pos_ == "ADJ":
            advmods = [
                child.text
                for child in sorted(
                    [c for c in head.children if c.dep_ in {"advmod", "neg"}],
                    key=lambda c: c.i,
                )
            ]
            phrases.append(_clean(" ".join(advmods + [head.text])))

        for child in tok.children:
            if child.dep_ in {"acl", "relcl", "appos"}:
                span = doc[child.left_edge.i : child.right_edge.i + 1]
                phrases.append(_clean(span.text))

        deduped = _dedupe_preserve_order(phrases)
        if deduped:
            results.append({"mention": tok.text, "descriptions": deduped})
    return results


def _process_article(record: Dict, nlp: spacy.language.Language) -> Dict:
    # Run extraction across one article's contexts
    shooter_sentences = []
    shooter_phrases: List[str] = []
    for sent in record.get("shooter_contexts", []):
        targets = _extract_from_sentence(sent, SHOOTER_TERMS, nlp)
        if targets:
            shooter_sentences.append({"sentence": sent, "targets": targets})
            for t in targets:
                shooter_phrases.extend(t["descriptions"])

    victim_sentences = []
    victim_phrases: List[str] = []
    for sent in record.get("victim_contexts", []):
        targets = _extract_from_sentence(sent, VICTIM_TERMS, nlp)
        if targets:
            victim_sentences.append({"sentence": sent, "targets": targets})
            for t in targets:
                victim_phrases.extend(t["descriptions"])

    return {
        "outlet": record.get("outlet"),
        "article": record.get("article"),
        "shooter_sentences": shooter_sentences,
        "victim_sentences": victim_sentences,
        "shooter_phrases": _dedupe_preserve_order(shooter_phrases),
        "victim_phrases": _dedupe_preserve_order(victim_phrases),
    }


def run(input_path: Path, output_dir: Path):
    # Batch process all records and write per-article plus aggregate outputs
    nlp = _load_spacy()
    data = json.loads(input_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    per_article_dir = output_dir / "descriptions"
    per_article_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    for record in data:
        processed = _process_article(record, nlp)
        all_records.append(processed)

        outlet_dir = per_article_dir / str(processed["outlet"])
        outlet_dir.mkdir(parents=True, exist_ok=True)
        with (outlet_dir / f"{processed['article']}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "outlet": processed["outlet"],
                    "article": processed["article"],
                    "shooter_phrases": processed["shooter_phrases"],
                    "victim_phrases": processed["victim_phrases"],
                    "shooter_sentences": processed["shooter_sentences"],
                    "victim_sentences": processed["victim_sentences"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    with (output_dir / "all_descriptions.json").open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)


def main():
    # CLI entrypoint
    parser = argparse.ArgumentParser(description="Extract descriptive phrases for shooter/victim mentions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "task1" / "all_contexts.json",
        help="Path to Task 1 aggregated contexts JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "task2",
        help="Directory to write description outputs.",
    )
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
