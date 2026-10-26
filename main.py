"""
FastAPI app exposing the RAG pipeline.

Run with:  uvicorn main:app
Docs at:   http://127.0.0.1:8000/docs

Environment variables (all optional) are listed in config.py. Set RAG_API_KEY to
require an X-API-Key header on /ask.
"""

import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import ollama
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, StringConstraints

import config
from rag_chain import RagPipeline

logger = logging.getLogger("rag.api")

pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the vector store, embedding model, reranker and LLM client once at
    # startup, not on every request.
    global pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "starting: llm=%s retrieval=%s k=%s", config.OLLAMA_MODEL, config.RETRIEVAL_MODE, config.RETRIEVAL_K
    )
    pipeline = RagPipeline()
    logger.info("pipeline ready")
    yield


app = FastAPI(title="Document RAG API", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if config.API_KEY and not hmac.compare_digest(x_api_key or "", config.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=config.MAX_QUESTION_CHARS),
]


class AskRequest(BaseModel):
    question: Question


class Passage(BaseModel):
    source: str
    page: int | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    passages: list[Passage]
    latency_ms: int


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest):
    started = time.perf_counter()
    try:
        result = pipeline.answer(request.question)
    except (httpx.ConnectError, ollama.ResponseError) as exc:
        logger.error("LLM backend unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"The local LLM is unavailable. Make sure Ollama is running and "
            f"the model is pulled (ollama pull {config.OLLAMA_MODEL}).",
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info("ask ok latency_ms=%d sources=%d", latency_ms, len(result["sources"]))
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        passages=result["passages"],
        latency_ms=latency_ms,
    )


@app.get("/health")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: the pipeline is loaded and the Ollama server answers."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")
    try:
        ollama.Client(host=config.OLLAMA_BASE_URL).list()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {exc}") from exc
    return {"status": "ready"}
