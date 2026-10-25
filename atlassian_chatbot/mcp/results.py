from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MCPToolResultError(RuntimeError):
    """An MCP tool returned an error payload instead of successful data."""


def ensure_successful_tool_result(result: Any, *, tool_name: str) -> None:
    """Raise for protocol error flags or recognizable validation/error responses."""
    if _has_error_flag(result):
        raise MCPToolResultError(f"{tool_name} returned an MCP error: {_result_text(result)[:1000]}")

    text = _result_text(result).strip()
    lowered = text.lower()
    error_prefixes = (
        "error:",
        "error executing tool",
        "tool execution failed",
        "failed to call tool",
    )
    if lowered.startswith(error_prefixes) or (
        "validation error" in lowered and ("required" in lowered or "missing" in lowered)
    ):
        raise MCPToolResultError(
            f"{tool_name} returned an MCP error instead of data: {text[:1000]}"
        )


def _has_error_flag(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("isError", value.get("is_error", False)):
            return True
        return any(_has_error_flag(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_error_flag(item) for item in value)
    return bool(getattr(value, "isError", getattr(value, "is_error", False)))


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return "; ".join(_result_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "; ".join(_result_text(item) for item in value)
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return str(value)
