"""Retrieval-augmented generation helpers for live Atlassian results."""

from .chunker import result_to_text, split_into_chunks
from .embedder import get_embedding_model, load_embedding_model
from .ingestion import AtlassianIngestionPipeline, IngestionError, IngestionResult
from .retriever import FaissRetriever, RetrievedChunk
from .store import AtlassianFaissStore
from .tools import RETRIEVAL_TOOL_NAME, make_retrieval_tool

__all__ = [
    "AtlassianFaissStore",
    "AtlassianIngestionPipeline",
    "FaissRetriever",
    "IngestionError",
    "IngestionResult",
    "RETRIEVAL_TOOL_NAME",
    "RetrievedChunk",
    "get_embedding_model",
    "load_embedding_model",
    "make_retrieval_tool",
    "result_to_text",
    "split_into_chunks",
]
