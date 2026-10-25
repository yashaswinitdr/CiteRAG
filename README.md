# Document Q&A with RAG (Retrieval-Augmented Generation)

A small RAG pipeline that answers questions from your own documents and shows which file each answer came from. It runs fully locally: no API key and no cost.

**Stack:** Python, LangChain, ChromaDB, FastAPI, HuggingFace sentence-transformers, Ollama (Llama 3.2)

## What is RAG?

A normal LLM only knows what it learned during training. It cannot see your private files, and it can hallucinate. RAG fixes this by retrieving relevant text from your documents first, then giving that text to the LLM so the answer is grounded in real content.

## How it works

**Phase 1: Ingestion (`ingest.py`, run once)**

1. Load `.pdf` and `.txt` files from `documents/`
2. Split them into overlapping chunks
3. Convert each chunk into an embedding (a vector that represents its meaning) with `all-MiniLM-L6-v2`
4. Store the embeddings and the source filename in ChromaDB

**Phase 2: Query (`rag_chain.py` + `main.py`, every question)**

1. Convert the question into an embedding with the same embedding model
2. Run a similarity search in ChromaDB to find the closest chunk(s)
3. Put the retrieved chunk(s) and the question into a prompt
4. The LLM (Llama 3.2 via Ollama) writes the answer using only that context
5. The API returns the answer and the source file(s)

The `sources` field comes from metadata stored with the matching chunk, not from the LLM, so you can verify where an answer came from.

## Project structure

```
RAG/
├── documents/        # put your .pdf / .txt files here
├── config.py         # model names, chunk size, retrieval settings
├── ingest.py         # builds the vector store
├── rag_chain.py      # retrieval + answer generation
├── main.py           # FastAPI app (POST /ask)
└── requirements.txt
```

## Setup (Windows)

1. Install [Ollama](https://ollama.com/download), then download the model:

   ```bash
   ollama pull llama3.2
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Add your documents to `documents/` and build the vector store:

   ```bash
   python ingest.py
   ```

4. Start the API:

   ```bash
   uvicorn main:app --reload
   ```

5. Open http://127.0.0.1:8000/docs, click `POST /ask`, then **Try it out**.

## Example

Request:

```json
{ "question": "How many vacation days do full-time employees get per year, and how many can they carry over?" }
```

Response:

```json
{
  "answer": "Full-time employees accrue 18 days of paid vacation per calendar year. They can carry over up to 5 unused days into the following year.",
  "sources": ["documents/vacation_policy.txt"]
}
```

## Configuration

Settings are in `config.py`:

| Setting | Meaning |
|---|---|
| `EMBEDDING_MODEL_NAME` | Local embedding model. Use the same model for ingestion and querying. |
| `OLLAMA_MODEL` | Local LLM served by Ollama |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | How documents are split |
| `RETRIEVAL_K` | How many chunks to retrieve per question |

`RETRIEVAL_K` is set to 1 because the sample corpus has only two short documents. With a larger corpus, use 3 to 5.

## Limitations

- This is a learning prototype, not a production system. It has no authentication, tests, or error handling for Ollama being down.
- Retrieval and generation fail separately. In testing, the retriever found the correct chunk for a question about stairs, but the small 3B local model still gave a wrong "yes". A larger model reduces this kind of error.
- It was tested on two small sample documents.

## Swapping the LLM

Generation is isolated in `rag_chain.py`. To use a hosted model such as Claude, replace `ChatOllama` with `ChatAnthropic` from `langchain-anthropic`. The retrieval code stays the same. Note that Anthropic has no embeddings API, so keep the local embedding model or use a provider like Voyage AI.
