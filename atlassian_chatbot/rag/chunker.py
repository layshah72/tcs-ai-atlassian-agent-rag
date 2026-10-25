from __future__ import annotations

import json
from typing import Any


def result_to_text(value: Any) -> str:
    """Normalize MCP text blocks and structured payloads into indexable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (list, tuple)):
        parts = [result_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    item_text = getattr(value, "text", None)
    if isinstance(item_text, str):
        return item_text
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return result_to_text(model_dump())
    return str(value)


def split_into_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    return [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), step)
        if words[start : start + chunk_words]
    ]
