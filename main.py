"""
FastAPI app exposing the RAG pipeline as an HTTP endpoint.

Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import RagPipeline

pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the vector store, embeddings, and Claude client once at startup -
    # not on every request.
    global pipeline
    pipeline = RagPipeline()
    yield


app = FastAPI(title="Document RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = pipeline.answer(request.question)
    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.get("/health")
def health():
    return {"status": "ok"}
