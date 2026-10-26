"""Embedding model factory shared by ingestion and querying (they must use the same model)."""

from langchain_huggingface import HuggingFaceEmbeddings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class QueryAwareEmbeddings(HuggingFaceEmbeddings):
    """BGE models are trained to embed search queries with an instruction prefix."""

    def embed_query(self, text: str):
        if "bge" in self.model_name.lower():
            text = BGE_QUERY_PREFIX + text
        return super().embed_query(text)


def make_embeddings(model_name: str) -> QueryAwareEmbeddings:
    return QueryAwareEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
