from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """Send application logs to the terminal running Streamlit.

    Streamlit may configure the root logger itself, so configure the application
    namespace directly and avoid adding duplicate handlers on script reruns.
    """
    requested_level = os.getenv("ATLASSIAN_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, requested_level, None)
    if not isinstance(level, int):
        level = logging.INFO

    app_logger = logging.getLogger("atlassian_chatbot")
    app_logger.setLevel(level)
    app_logger.propagate = False

    if not any(getattr(handler, "_atlassian_terminal_handler", False) for handler in app_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler._atlassian_terminal_handler = True  # type: ignore[attr-defined]
        app_logger.addHandler(handler)
