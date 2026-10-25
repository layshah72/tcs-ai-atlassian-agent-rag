from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from atlassian_chatbot.mcp.results import ensure_successful_tool_result
from .chunker import result_to_text, split_into_chunks
from .embedder import get_embedding_model, load_embedding_model
from .retriever import FaissRetriever, RetrievedChunk


class AtlassianFaissStore:
    """Indexes MCP results and retrieves relevant chunks via FAISS + FastEmbed."""

    INDEX_VERSION = 2

    def __init__(
        self,
        *,
        chunk_words: int = 220,
        overlap_words: int = 35,
        model_name: str = "BAAI/bge-small-en-v1.5",
        embedding_model: Any | None = None,
    ) -> None:
        if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
            raise ValueError("chunking requires 0 <= overlap_words < chunk_words")
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.model_name = model_name
        self.index_version = self.INDEX_VERSION
        self._embedding_model = embedding_model
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._chunks: list[RetrievedChunk] = []

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def add_result(self, source: str, result: Any) -> int:
        """Chunk and index an MCP result; return the number of chunks added."""
        ensure_successful_tool_result(result, tool_name=source)
        text = result_to_text(result).strip()
        if not text:
            return 0
        chunks = split_into_chunks(text, self.chunk_words, self.overlap_words)
        vectors = np.asarray(list(self._model().embed(chunks)), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("FastEmbed returned an invalid document embedding matrix")
        faiss.normalize_L2(vectors)
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        elif vectors.shape[1] != self._faiss_index.d:
            raise ValueError("FastEmbed returned embeddings with inconsistent dimensions")
        self._faiss_index.add(vectors)
        self._chunks.extend(
            RetrievedChunk(text=chunk, source=source, score=0.0, chunk_number=i + 1)
            for i, chunk in enumerate(chunks)
        )
        return len(chunks)

    def search(self, query: str, *, top_k: int = 5, max_chars: int = 7000) -> list[RetrievedChunk]:
        """Return the most relevant chunks for *query* within a bounded context."""
        if self._faiss_index is None or not self._chunks or not query.strip():
            return []
        vector = np.asarray(list(self._model().query_embed(query)), dtype="float32")
        retriever = FaissRetriever(self._faiss_index, self._chunks)
        return retriever.search(vector, top_k=top_k, max_chars=max_chars)

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and chunks to *directory*."""
        if self._faiss_index is None or not self._chunks:
            return
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as fh:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "chunk_words": self.chunk_words,
                    "overlap_words": self.overlap_words,
                    "model_name": self.model_name,
                    "index_version": self.index_version,
                },
                fh,
            )

    @classmethod
    def load(cls, directory: str | Path, *, embedding_model: Any | None = None) -> "AtlassianFaissStore | None":
        """Load a previously saved store; return *None* if no valid data exists."""
        path = Path(directory)
        index_file = path / "index.faiss"
        chunks_file = path / "chunks.pkl"
        if not index_file.exists() or not chunks_file.exists():
            return None
        try:
            with open(chunks_file, "rb") as fh:
                meta = pickle.load(fh)
            if meta.get("index_version") != cls.INDEX_VERSION:
                return None
            store = cls(
                chunk_words=meta["chunk_words"],
                overlap_words=meta["overlap_words"],
                model_name=meta["model_name"],
                embedding_model=embedding_model,
            )
            store._faiss_index = faiss.read_index(str(index_file))
            store._chunks = meta["chunks"]
            return store
        except Exception:
            return None

    def _model(self) -> Any:
        if self._embedding_model is None:
            self._embedding_model = load_embedding_model(self.model_name)
        return self._embedding_model
