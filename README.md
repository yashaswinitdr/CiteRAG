# CiteRAG: Hybrid-Retrieval Document Q&A

A fully local Retrieval-Augmented Generation (RAG) system. Ask a question about your documents and it returns an answer, the source files, and the exact passages used. It refuses questions the documents cannot answer. No API key and no cost.

**Stack:** Python, FastAPI, LangChain, ChromaDB, Ollama (Qwen2.5-7B), sentence-transformers, BM25

## How it works

**Ingestion (`ingest.py`, run once)**

1. Load the PDFs in `documents/`
2. Split them into overlapping chunks (400 characters, 80 overlap)
3. Embed each chunk with `all-MiniLM-L6-v2`
4. Store the vectors, file name and page in ChromaDB

**Answering (`retriever.py`, `rag_chain.py`)**

1. **Hybrid retrieval:** embedding search and BM25 keyword search run together, merged with Reciprocal Rank Fusion
2. **Reranking:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-reads the top 20 candidates and keeps the best 5
3. **Generation:** the local LLM answers using only those 5 passages, or replies that the documents do not contain the answer
4. **Citation:** the API returns the answer, source files and the passages

## Results

Corpus: 5 research papers (82 pages, 1,087 chunks). All numbers come from the scripts in `evaluation/`.

**Retrieval** (21 answerable dev questions, "Hit@5" = the passage containing the answer is in the top 5)

| Setup | Hit@5 | MRR |
|---|---|---|
| Dense only (MiniLM, chunk 400) | 0.48 | 0.35 |
| Dense + reranker | 0.67 | 0.56 |
| Hybrid (embeddings + BM25) | 0.67 | 0.42 |
| **Hybrid + reranker (used)** | **0.91** | **0.77** |

The full 24-configuration grid (2 embedding models x 3 chunk sizes x 4 modes) is in `evaluation/results/retrieval_ablation.md`.

**End to end** (Qwen2.5-7B, hybrid + reranker, k = 5)

| Set | Answerable | Off-topic refused | Right passage retrieved |
|---|---|---|---|
| Dev (25 questions) | 20 / 21 | 4 / 4 | 90.5% |
| **Held-out (16 questions)** | 8 / 12 | 4 / 4 | 66.7% |

The same LLM with no documents answered 2 / 21 dev and 1 / 12 held-out questions correctly. This is a sanity check that retrieval is needed, not a fair benchmark, because the questions ask for exact facts from specific papers.

**Read these numbers carefully.** I used the dev set to choose the chunk size, retrieval mode and k, so its score is optimistic. The held-out set was written before the final run and never used for tuning, so it is the more honest estimate. The gap (96% vs 75%) shows overfitting to a small dev set. Both sets are small and written by me, so differences of one question are noise. Answers are scored automatically by keyword matching, which can mark a correct paraphrase wrong.

**Failure analysis (held-out):** 3 of 4 failures were retrieval misses (the right passage never reached the LLM), and 1 was a generation error (passage retrieved, but the model said it was not in the documents). In two retrieval misses the model gave a vague answer from partial context instead of refusing.

## Run it

```bash
# 1. Install Ollama (https://ollama.com/download), then pull the model
ollama pull qwen2.5:7b

# 2. Python environment
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Get the demo papers (or put your own PDFs/.txt files in documents/), then build the vector store
python scripts/download_papers.py
python ingest.py

# 4. Start the API
uvicorn main:app
```

Open http://127.0.0.1:8000/docs, click `POST /ask`, then **Try it out**.

On a CPU, Qwen2.5-7B takes roughly 30 to 120 seconds per answer. For faster answers, use a smaller model: `RAG_OLLAMA_MODEL=llama3.2` (the retrieval part stays the same).

### API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question": "..."}` returns `answer`, `sources`, `passages`, `latency_ms` |
| `GET /health` | Liveness |
| `GET /ready` | Readiness (pipeline loaded and Ollama reachable) |

Questions must be 3 to 1000 characters. Set `RAG_API_KEY` to require an `X-API-Key` header on `/ask`.

### Configuration

Every setting in `config.py` can be overridden with a `RAG_` environment variable, for example `RAG_OLLAMA_MODEL`, `RAG_RETRIEVAL_MODE` (`dense`, `dense_rerank`, `hybrid`, `hybrid_rerank`), `RAG_RETRIEVAL_K`, `RAG_CHUNK_SIZE`.

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
pytest                                            # unit and API tests, no LLM needed

python -m evaluation.validate_dataset             # check gold evidence exists in the PDFs
python -m evaluation.run_retrieval_ablation       # retrieval experiment, no LLM
python -m evaluation.run_end_to_end --dataset evaluation/dataset.json --output dev --baseline
python -m evaluation.run_end_to_end --dataset evaluation/dataset_heldout.json --output heldout --baseline
```

`samples/` holds two tiny text files and `smoke_test.py` runs a quick end-to-end check on them.

## Project structure

```
documents/        corpus (PDFs are downloaded by scripts/download_papers.py)
samples/          tiny demo files
scripts/          download_papers.py
.github/          CI workflow (runs pytest)
evaluation/       datasets, metrics, experiment scripts, results
tests/            unit and API tests
ingest.py         build the vector store
embeddings.py     embedding model factory
retriever.py      dense / BM25 / hybrid / rerank retrieval
rag_chain.py      prompt, LLM call, refusal and citation logic
main.py           FastAPI service
config.py         settings (environment overridable)
```

## Limitations

- Small, self-written evaluation sets (25 dev and 16 held-out questions) and keyword-based scoring.
- Chunks are cut within a page, so an answer that spans a chunk boundary can be missed or incomplete.
- A local 7B model is slow on CPU and sometimes answers vaguely from partial context instead of refusing.
- No Docker image yet, and no authentication beyond an optional API key.
