from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float
    chunk_number: int


class FaissRetriever:
    """Wraps a FAISS IndexFlatIP and performs bounded similarity search."""

    def __init__(self, index: faiss.IndexFlatIP, chunks: list[RetrievedChunk]) -> None:
        self._index = index
        self._chunks = chunks

    @property
    def index(self) -> faiss.IndexFlatIP:
        return self._index

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return self._chunks

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        max_chars: int = 7000,
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks within max_chars budget."""
        if not self._chunks or top_k < 1 or max_chars < 1:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[0] != 1 or query_vector.shape[1] != self._index.d:
            raise ValueError("Query embedding dimensions do not match the index")
        faiss.normalize_L2(query_vector)
        scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))
        selected: list[RetrievedChunk] = []
        used_chars = 0
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or score <= 0:
                continue
            chunk = self._chunks[int(index)]
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            text = chunk.text[:remaining]
            selected.append(
                RetrievedChunk(
                    text=text,
                    source=chunk.source,
                    score=float(score),
                    chunk_number=chunk.chunk_number,
                )
            )
            used_chars += len(text)
        return selected
