"""
Ingestion pipeline: load documents -> split into chunks -> embed -> store in ChromaDB.

Run this once to build the vector store, and again any time you add or change
files in documents/.

    python ingest.py
    python ingest.py --docs samples --db chroma_samples --chunk-size 300
"""

import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config
from embeddings import make_embeddings


def load_documents(docs_dir=config.DOCUMENTS_DIR):
    """Load every .pdf and .txt file under docs_dir into LangChain Document objects.

    Each Document keeps a `metadata["source"]` field set to its file path -
    that's what lets us tell the user which file an answer came from later.
    """
    pdf_loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    txt_loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = pdf_loader.load() + txt_loader.load()

    if not documents:
        raise RuntimeError(
            f"No .pdf or .txt files found in {docs_dir}. "
            "Add some documents and run this script again."
        )

    return documents


def split_documents(documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    """Break each document into overlapping chunks small enough to embed and retrieve well."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vectorstore(
    chunks,
    db_dir=config.CHROMA_DIR,
    embedding_model=config.EMBEDDING_MODEL_NAME,
    collection_name=config.COLLECTION_NAME,
):
    """Embed each chunk and persist the vectors + text + metadata to ChromaDB on disk."""
    embeddings = make_embeddings(embedding_model)

    # Start from an empty directory. Re-running without this would append
    # duplicate copies of the same chunks.
    db_path = Path(db_dir)
    if db_path.exists():
        shutil.rmtree(db_path)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(db_path),
    )


def ingest(
    docs_dir=config.DOCUMENTS_DIR,
    db_dir=config.CHROMA_DIR,
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    embedding_model=config.EMBEDDING_MODEL_NAME,
):
    documents = load_documents(docs_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    build_vectorstore(chunks, db_dir, embedding_model)
    return len(documents), len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector store.")
    parser.add_argument("--docs", default=str(config.DOCUMENTS_DIR))
    parser.add_argument("--db", default=str(config.CHROMA_DIR))
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL_NAME)
    args = parser.parse_args()

    print(f"Ingesting {args.docs} -> {args.db} ...")
    n_docs, n_chunks = ingest(
        args.docs, args.db, args.chunk_size, args.chunk_overlap, args.embedding_model
    )
    print(f"Done: {n_docs} document page(s), {n_chunks} chunk(s).")


if __name__ == "__main__":
    main()
