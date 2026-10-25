from __future__ import annotations

from pathlib import Path

from atlassian_chatbot.config import AppConfig

_PROMPTS_DIR = Path(__file__).parent


def _load(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


SYSTEM_PROMPT = _load("system.md")
INDEXED_READ_SYSTEM_PROMPT = _load("indexed_read.md")


def build_system_prompt(config: AppConfig) -> str:
    _ = config
    return SYSTEM_PROMPT
