"""
Central settings. Every value can be overridden with an environment variable,
e.g.  RAG_OLLAMA_MODEL=llama3.2  RAG_RETRIEVAL_K=3  uvicorn main:app
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"RAG_{name}", default)


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(_env("DOCUMENTS_DIR", str(BASE_DIR / "documents")))
CHROMA_DIR = Path(_env("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = _env("COLLECTION_NAME", "rag_documents")

# Embedding model: runs locally on CPU, no API key needed.
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local LLM served by Ollama - no API key needed.
# Must match a model you've pulled with `ollama pull <name>`.
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL") or None  # None = http://localhost:11434

# Retrieval: "dense", "dense_rerank", "hybrid" (dense + BM25 keyword), or
# "hybrid_rerank" (hybrid, then a cross-encoder re-orders the candidates).
RETRIEVAL_MODE = _env("RETRIEVAL_MODE", "hybrid_rerank")
FETCH_K = int(_env("FETCH_K", "20"))  # candidates gathered before fusion / reranking
RERANKER_MODEL = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chunking: how documents get split before embedding.
CHUNK_SIZE = int(_env("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "80"))

# How many chunks to retrieve per question.
RETRIEVAL_K = int(_env("RETRIEVAL_K", "5"))

# API: if set, every /ask request must send this value in the X-API-Key header.
API_KEY = os.getenv("RAG_API_KEY") or None
MAX_QUESTION_CHARS = int(_env("MAX_QUESTION_CHARS", "1000"))
