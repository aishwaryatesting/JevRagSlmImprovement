import os

from app.implementations.lexical_retriever import LexicalRetriever
from app.implementations.deterministic_generator import DeterministicGenerator


def build_retriever():
    mode = os.getenv("RETRIEVER_MODE", "lexical").lower()
    if mode == "faiss":
        from app.implementations.faiss_retriever import FaissRetriever
        return FaissRetriever()
    if mode == "hybrid":
        from app.implementations.hybrid_retriever import HybridRetriever
        return HybridRetriever(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            float(os.getenv("HYBRID_ALPHA", "0.5")),
        )
    return LexicalRetriever()


def build_reranker():
    if os.getenv("RERANKER_MODE", "none").lower() == "cross_encoder":
        from app.reranking.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(
            os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
    return None


def build_generator():
    if os.getenv("GENERATOR_MODE", "deterministic").lower() == "hf":
        from app.implementations.hf_generator import HFGenerator
        return HFGenerator(
            os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            os.getenv("HF_ADAPTER") or None,
            int(os.getenv("MAX_NEW_TOKENS", "160")),
            float(os.getenv("TEMPERATURE", "0.0")),
        )
    return DeterministicGenerator()
