"""Text normalisation and metric helpers shared by the evaluation scripts."""

import json
import re
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"

_REPLACEMENTS = {
    "ﬁ": "fi", "ﬂ": "fl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-",
}


def normalize(text: str) -> str:
    """Lowercase and clean PDF extraction artefacts so substring checks are reliable."""
    text = text.lower()
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    # PDFs hyphenate across line breaks ("dual- encoder"); close the gap.
    return re.sub(r"-\s+", "-", text).strip()


def load_dataset(path=DATASET_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contains_evidence(chunk_text: str, evidence: list) -> bool:
    """True if the chunk contains any of the gold evidence phrases."""
    chunk = normalize(chunk_text)
    return any(normalize(e) in chunk for e in evidence)


def answer_is_correct(answer: str, item: dict) -> bool:
    """Keyword check: every answer_all phrase present, and at least one answer_any phrase."""
    answer = normalize(answer)
    all_ok = all(normalize(a) in answer for a in item["answer_all"])
    any_ok = not item["answer_any"] or any(normalize(a) in answer for a in item["answer_any"])
    return all_ok and any_ok


def first_hit_rank(retrieved_texts: list, evidence: list):
    """1-based rank of the first retrieved chunk containing the evidence, or None."""
    for rank, text in enumerate(retrieved_texts, start=1):
        if contains_evidence(text, evidence):
            return rank
    return None


def retrieval_metrics(ranks: list, ks=(1, 3, 5)) -> dict:
    """Hit@k and MRR from a list of first-hit ranks (None = never found)."""
    n = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    metrics["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return metrics
