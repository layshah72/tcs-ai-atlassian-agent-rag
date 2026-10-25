"""Backward-compatible re-exports. Import from the specific modules for new code."""

from .chunker import result_to_text, split_into_chunks
from .embedder import get_embedding_model, load_embedding_model
from .retriever import FaissRetriever, RetrievedChunk
from .store import AtlassianFaissStore

__all__ = [
    "AtlassianFaissStore",
    "FaissRetriever",
    "RetrievedChunk",
    "get_embedding_model",
    "load_embedding_model",
    "result_to_text",
    "split_into_chunks",
]
