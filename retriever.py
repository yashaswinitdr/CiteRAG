"""
Retrieval strategies over a Chroma vector store.

  dense          embedding similarity only
  dense_rerank   embedding similarity, then a cross-encoder re-orders the candidates
  hybrid         embedding similarity + BM25 keyword search, merged with
                 Reciprocal Rank Fusion
  hybrid_rerank  hybrid candidates, re-ordered by a cross-encoder
"""

import re

from langchain_core.documents import Document

RRF_CONSTANT = 60
MODES = ("dense", "dense_rerank", "hybrid", "hybrid_rerank")


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, vectorstore, k=3, mode="dense", fetch_k=20, reranker_model=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.vectorstore = vectorstore
        self.k = k
        self.mode = mode
        self.fetch_k = fetch_k
        self._rerank = mode.endswith("rerank")

        if mode.startswith("hybrid"):
            from rank_bm25 import BM25Okapi

            data = vectorstore.get(include=["documents", "metadatas"])
            self._docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(data["documents"], data["metadatas"])
            ]
            self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._docs])

        if self._rerank:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(reranker_model)

    def _hybrid_candidates(self, query):
        dense = self.vectorstore.similarity_search(query, k=self.fetch_k)

        scores = self._bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[: self.fetch_k]
        keyword = [self._docs[i] for i in top_idx]

        fused, by_key = {}, {}
        for ranking in (dense, keyword):
            for rank, doc in enumerate(ranking, start=1):
                key = doc.page_content
                by_key[key] = doc
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_CONSTANT + rank)
        ordered = sorted(fused, key=fused.get, reverse=True)
        return [by_key[key] for key in ordered]

    def retrieve(self, query):
        if self.mode.startswith("hybrid"):
            candidates = self._hybrid_candidates(query)
        else:
            limit = self.fetch_k if self._rerank else self.k
            candidates = self.vectorstore.similarity_search(query, k=limit)

        if self._rerank:
            candidates = candidates[: self.fetch_k]
            scores = self._reranker.predict([(query, d.page_content) for d in candidates])
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            candidates = [candidates[i] for i in order]

        return candidates[: self.k]
