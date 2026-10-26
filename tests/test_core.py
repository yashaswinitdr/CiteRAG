"""Fast unit tests. They need no LLM, no Ollama and no model downloads."""

import pytest
from langchain_core.documents import Document

from evaluation.metrics import (
    answer_is_correct,
    contains_evidence,
    first_hit_rank,
    normalize,
    retrieval_metrics,
)
from rag_chain import is_refusal
from retriever import Retriever


class FakeVectorStore:
    """Dense results are fixed; get() exposes the corpus the BM25 index is built on."""

    def __init__(self, docs, dense_order):
        self._docs = docs
        self._dense_order = dense_order

    def similarity_search(self, query, k=4):
        return [self._docs[i] for i in self._dense_order[:k]]

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }


def test_normalize_fixes_pdf_artifacts():
    assert normalize("Dual-\nencoder  ﬁne") == "dual-encoder fine"


def test_contains_evidence_matches_any_phrase():
    assert contains_evidence("It uses a Dual- encoder framework.", ["dual-encoder"])
    assert not contains_evidence("Nothing relevant here.", ["dual-encoder"])


def test_answer_is_correct_needs_all_and_any():
    item = {"answer_all": ["bm25"], "answer_any": ["mini-batch", "in-batch"]}
    assert answer_is_correct("BM25 plus mini-batch gold passages", item)
    assert not answer_is_correct("Only BM25", item)
    assert not answer_is_correct("Only in-batch negatives", item)


def test_first_hit_rank_and_metrics():
    assert first_hit_rank(["a", "needle here", "c"], ["needle"]) == 2
    assert first_hit_rank(["a", "b"], ["needle"]) is None
    metrics = retrieval_metrics([1, 3, None, 2])
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


@pytest.mark.parametrize(
    "answer",
    [
        "The documents do not contain this information.",
        "The context does not contain the answer.",
        "I couldn't find any information about that.",
    ],
)
def test_is_refusal_detects_refusals(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "SBERT takes about 5 seconds.",
        "No. The robot is designed for single-level flooring only.",
    ],
)
def test_is_refusal_ignores_real_answers(answer):
    assert not is_refusal(answer)


CORPUS = [
    "alpha apples orchard harvest",
    "beta dual encoder dense retrieval",
    "gamma bm25 keyword lexical retrieval",
    "delta cooking recipe pasta",
    "epsilon weather forecast rain",
    "zeta football match score",
]


def _corpus_docs():
    return [Document(page_content=t, metadata={"source": str(i)}) for i, t in enumerate(CORPUS)]


def test_hybrid_ranks_documents_found_by_both_searches_first():
    # Dense search and BM25 both put document 2 near the top for this query.
    store = FakeVectorStore(_corpus_docs(), dense_order=[2, 1, 0, 3, 4, 5])
    retriever = Retriever(store, k=3, mode="hybrid", fetch_k=3)

    results = retriever.retrieve("keyword lexical retrieval")

    assert results[0].metadata["source"] == "2"


def test_hybrid_includes_keyword_match_that_dense_search_missed():
    # Dense search ranks document 2 last, outside its top 3; BM25 still finds it.
    store = FakeVectorStore(_corpus_docs(), dense_order=[1, 0, 3, 4, 5, 2])
    retriever = Retriever(store, k=3, mode="hybrid", fetch_k=3)

    sources = [d.metadata["source"] for d in retriever.retrieve("keyword lexical retrieval")]

    assert "2" in sources


def test_dense_mode_respects_k():
    docs = [Document(page_content=f"doc {i}", metadata={"source": str(i)}) for i in range(5)]
    retriever = Retriever(FakeVectorStore(docs, [4, 3, 2, 1, 0]), k=2, mode="dense")
    assert [d.metadata["source"] for d in retriever.retrieve("q")] == ["4", "3"]


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        Retriever(FakeVectorStore([], []), mode="nonsense")
