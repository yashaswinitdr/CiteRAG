"""
Ingestion pipeline: load documents -> split into chunks -> embed -> store in ChromaDB.

Run this once to build the vector store, and again any time you add or change
files in documents/.
"""

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def load_documents():
    """Load every .pdf and .txt file under documents/ into LangChain Document objects.

    Each Document keeps a `metadata["source"]` field set to its file path -
    that's what lets us tell the user which file an answer came from later.
    """
    pdf_loader = DirectoryLoader(
        str(config.DOCUMENTS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    txt_loader = DirectoryLoader(
        str(config.DOCUMENTS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = pdf_loader.load() + txt_loader.load()

    if not documents:
        raise RuntimeError(
            f"No .pdf or .txt files found in {config.DOCUMENTS_DIR}. "
            "Add some documents and run this script again."
        )

    return documents


def split_documents(documents):
    """Break each document into overlapping chunks small enough to embed and retrieve well."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Embed each chunk and persist the vectors + text + metadata to ChromaDB on disk."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    return vectorstore


def main():
    print(f"Loading documents from {config.DOCUMENTS_DIR} ...")
    documents = load_documents()
    print(f"  loaded {len(documents)} document(s)")

    print("Splitting into chunks ...")
    chunks = split_documents(documents)
    print(f"  produced {len(chunks)} chunk(s)")

    print("Embedding and writing to ChromaDB (first run downloads the embedding model) ...")
    build_vectorstore(chunks)
    print(f"Done. Vector store persisted at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
