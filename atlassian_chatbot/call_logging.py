from __future__ import annotations

import contextvars
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)
_current_prompt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "atlassian_agent_prompt", default=None
)
_current_system_prompt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "atlassian_agent_system_prompt", default=None
)
_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "authorization", "api_key", "apikey")
_SECRET_PATTERNS = (
    re.compile(r"\bATATT[A-Za-z0-9_=-]+", re.IGNORECASE),
    re.compile(r"\bgsk_[A-Za-z0-9_-]+", re.IGNORECASE),
    re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE),
)


def set_current_prompt(prompt: str) -> contextvars.Token[str | None]:
    """Associate tool calls in this async context with the user's latest prompt."""
    return _current_prompt.set(prompt)


def reset_current_prompt(token: contextvars.Token[str | None]) -> None:
    _current_prompt.reset(token)


def get_current_prompt() -> str | None:
    return _current_prompt.get()


def set_current_system_prompt(prompt: str) -> contextvars.Token[str | None]:
    """Associate tool calls in this async context with the agent system prompt."""
    return _current_system_prompt.set(prompt)


def reset_current_system_prompt(token: contextvars.Token[str | None]) -> None:
    _current_system_prompt.reset(token)


def get_current_system_prompt() -> str | None:
    return _current_system_prompt.get()


def accepted_tool_parameters(tool: Any) -> list[str]:
    """Return parameter names declared by an MCP tool's input schema."""
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, Mapping):
        properties = schema.get("properties", {})
        return list(properties) if isinstance(properties, Mapping) else []
    model_fields = getattr(schema, "model_fields", None)
    if model_fields is not None:
        return list(model_fields)
    schema_method = getattr(schema, "model_json_schema", None)
    if callable(schema_method):
        properties = schema_method().get("properties", {})
        return list(properties) if isinstance(properties, Mapping) else []
    return []


def _redact(value: Any, *, key: str = "") -> Any:
    if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): _redact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
    return value


def log_tool_call(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    phase: str = "invoke",
    prompt: str | None = None,
    system_prompt: str | None = None,
    accepted_parameters: list[str] | None = None,
) -> None:
    """Log prompts and parameters while masking credential-like values."""
    prompt = get_current_prompt() if prompt is None else prompt
    system_prompt = get_current_system_prompt() if system_prompt is None else system_prompt
    safe_prompt = _redact(prompt) if prompt is not None else "[not available]"
    safe_system_prompt = _redact(system_prompt) if system_prompt is not None else "[not available]"
    safe_arguments = _redact(arguments)
    logger.info(
        "MCP tool %s | system_prompt=%s | user_prompt=%s | tool=%s | accepted_parameters=%s | passed_parameters=%s",
        phase,
        json.dumps(safe_system_prompt, ensure_ascii=False, default=str),
        json.dumps(safe_prompt, ensure_ascii=False, default=str),
        tool_name,
        json.dumps(accepted_parameters or [], ensure_ascii=False),
        json.dumps(safe_arguments, ensure_ascii=False, default=str),
    )
