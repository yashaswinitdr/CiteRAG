"""
Download the five open-access arXiv papers used as the demo corpus into documents/.
Files that already exist are skipped.

    python scripts/download_papers.py
"""

import time
import urllib.request
from pathlib import Path

PAPERS = {
    "lewis2020_rag.pdf": "2005.11401",               # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "karpukhin2020_dpr.pdf": "2004.04906",           # Dense Passage Retrieval for Open-Domain QA
    "reimers2019_sbert.pdf": "1908.10084",           # Sentence-BERT
    "gao2023_rag_survey.pdf": "2312.10997",          # RAG for Large Language Models: A Survey
    "liu2023_lost_in_the_middle.pdf": "2307.03172",  # Lost in the Middle
}

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"


def main():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    for filename, arxiv_id in PAPERS.items():
        target = DOCUMENTS_DIR / filename
        if target.exists():
            print(f"skip  {filename} (already there)")
            continue
        print(f"fetch {filename} <- arxiv.org/abs/{arxiv_id}")
        urllib.request.urlretrieve(f"https://arxiv.org/pdf/{arxiv_id}", target)
        time.sleep(3)  # be polite to arxiv.org
    print("done")


if __name__ == "__main__":
    main()
