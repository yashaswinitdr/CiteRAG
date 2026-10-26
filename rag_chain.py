"""
Query-time RAG pipeline: retrieve relevant chunks, ask the local LLM to answer
using only that context, and report which source document(s) were used.
"""

import re
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import config
from embeddings import make_embeddings
from retriever import Retriever

REFUSAL = "The documents do not contain this information."
_REFUSAL_PATTERN = re.compile(
    r"(documents?|context|passages?|provided (text|information))\s+(do(es)?\s+not|doesn't|don't)\s+(contain|include|mention|provide|say|specify)"
    r"|could(n't| not) find",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(answer))

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the context below.\n"
            "Rules:\n"
            "1. Read the context carefully before answering.\n"
            "2. Only if the question is a yes/no question (it starts with words "
            "like Is, Are, Can, Does, Do, Will), begin with 'Yes' or 'No'. Check "
            "for negations such as 'not', 'no', 'only' and 'cannot' so the "
            "yes/no matches what the context says. For any other question, do "
            "NOT start with Yes or No.\n"
            "3. Give the answer and the supporting fact from the context in one "
            "to three sentences.\n"
            f"4. If the context does not contain the answer, reply exactly: "
            f"'{REFUSAL}' Do not use outside knowledge.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# Used by the evaluation to measure what retrieval adds: same LLM, no documents.
NO_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question briefly from your own knowledge. If you do not "
            "know, say you do not know.",
        ),
        ("human", "{question}"),
    ]
)


def _source_name(doc) -> str:
    """File name only, so the API never exposes server directory paths."""
    return Path(doc.metadata.get("source", "unknown")).name


def _format_context(docs):
    """Join retrieved chunks into one context string, tagging each with its source."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RagPipeline:
    def __init__(
        self,
        db_dir=config.CHROMA_DIR,
        k=config.RETRIEVAL_K,
        embedding_model=config.EMBEDDING_MODEL_NAME,
        llm_model=config.OLLAMA_MODEL,
        collection_name=config.COLLECTION_NAME,
        retrieval_mode=config.RETRIEVAL_MODE,
        fetch_k=config.FETCH_K,
        reranker_model=config.RERANKER_MODEL,
    ):
        embeddings = make_embeddings(embedding_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(db_dir),
        )
        self.retriever = Retriever(
            self.vectorstore,
            k=k,
            mode=retrieval_mode,
            fetch_k=fetch_k,
            reranker_model=reranker_model,
        )
        # temperature=0 makes answers repeatable, which suits factual Q&A.
        self.llm = ChatOllama(
            model=llm_model, temperature=0, base_url=config.OLLAMA_BASE_URL
        )

    def retrieve(self, question: str):
        """Return the top-k chunks (LangChain Documents) for a question."""
        return self.retriever.retrieve(question)

    def answer(self, question: str) -> dict:
        # 1. Retrieve the most relevant chunks for this question.
        retrieved_docs = self.retrieve(question)

        # 2. Build the prompt: system message carries the retrieved context,
        #    human message carries the actual question.
        context = _format_context(retrieved_docs)
        messages = RAG_PROMPT.format_messages(context=context, question=question)

        # 3. Ask the LLM.
        answer = self.llm.invoke(messages).content.strip()

        # 4. Cite sources only when the answer actually used the documents.
        if is_refusal(answer):
            return {"answer": answer, "sources": [], "passages": []}

        sources = sorted({_source_name(doc) for doc in retrieved_docs})
        passages = [
            {
                "source": _source_name(doc),
                # PDF loaders count pages from 0; show the page number people see.
                "page": doc.metadata["page"] + 1 if isinstance(doc.metadata.get("page"), int) else None,
                "text": doc.page_content,
            }
            for doc in retrieved_docs
        ]
        return {"answer": answer, "sources": sources, "passages": passages}

    def answer_without_retrieval(self, question: str) -> str:
        """Baseline: the same LLM answering with no documents at all."""
        messages = NO_RAG_PROMPT.format_messages(question=question)
        return self.llm.invoke(messages).content.strip()
