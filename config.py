from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"

# Embedding model: runs locally on CPU, no API key needed.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM served by Ollama (http://localhost:11434) - no API key needed.
# Must match a model you've pulled with `ollama pull <name>`.
OLLAMA_MODEL = "llama3.2"

# Chunking: how documents get split before embedding.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# How many chunks to retrieve per question.
RETRIEVAL_K = 1
