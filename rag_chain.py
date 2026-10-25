"""
Query-time RAG pipeline: retrieve relevant chunks, ask Claude to answer using
only that context, and report which source document(s) the answer came from.
"""

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import config

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using ONLY the "
            "context provided below. If the context does not contain enough "
            "information to answer, say so clearly instead of guessing.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def _format_context(docs):
    """Join retrieved chunks into one context string, tagging each with its source."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RagPipeline:
    def __init__(self):
        self.vectorstore = _load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVAL_K}
        )
        self.llm = ChatOllama(model=config.OLLAMA_MODEL)

    def answer(self, question: str) -> dict:
        # 1. Retrieve the most relevant chunks for this question.
        retrieved_docs = self.retriever.invoke(question)

        # 2. Build the prompt: system message carries the retrieved context,
        #    human message carries the actual question.
        context = _format_context(retrieved_docs)
        messages = PROMPT.format_messages(context=context, question=question)

        # 3. Ask Claude.
        response = self.llm.invoke(messages)

        # 4. Report the answer plus the distinct source files it drew from.
        sources = sorted({doc.metadata.get("source", "unknown") for doc in retrieved_docs})

        return {
            "answer": response.content,
            "sources": sources,
        }
