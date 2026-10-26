"""Checks that every gold evidence phrase in dataset.json exists in its source PDF."""

import sys
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from evaluation.metrics import load_dataset, normalize


def main():
    dataset = load_dataset(sys.argv[1]) if len(sys.argv) > 1 else load_dataset()
    full_text = {}
    for item in dataset:
        if item["type"] != "answerable":
            continue
        source = item["source"]
        if source not in full_text:
            reader = PdfReader(config.DOCUMENTS_DIR / source)
            full_text[source] = normalize(" ".join(p.extract_text() or "" for p in reader.pages))

    problems = 0
    for item in dataset:
        if item["type"] != "answerable":
            continue
        for phrase in item["evidence"]:
            if normalize(phrase) not in full_text[item["source"]]:
                problems += 1
                print(f"MISSING  {item['id']}: '{phrase}' not found in {item['source']}")

    answerable = sum(1 for i in dataset if i["type"] == "answerable")
    print(f"{answerable} answerable items checked, {problems} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
