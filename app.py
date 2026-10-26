from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from atlassian_chatbot.config import AppConfig, ConfigError
from atlassian_chatbot.config.logging_config import configure_logging
from atlassian_chatbot.ui.conversation import render_conversation
from atlassian_chatbot.ui.sidebar import render_sidebar

load_dotenv()
configure_logging()
st.set_page_config(page_title="Atlassian AI Agent", page_icon="✦", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 980px; padding-top: 2.2rem; }
      [data-testid="stChatMessage"] { border: 1px solid rgba(120,130,150,.18); border-radius: 14px; padding: .8rem 1rem; }
      .hero { padding: 1.35rem 1.5rem; border-radius: 18px; margin-bottom: 1.4rem;
              background: linear-gradient(120deg, rgba(28,91,153,.15), rgba(92,57,150,.12)); }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 1.9rem; }
      .hero p { margin: 0; opacity: .78; }
    </style>
    <div class="hero"><h1>✦ Atlassian AI Agent</h1>
    <p>Ask about Jira issues and Confluence knowledge. Changes always wait for your approval.</p></div>
    """,
    unsafe_allow_html=True,
)

try:
    app_config = AppConfig.from_environment()
except ConfigError as exc:
    st.error(str(exc))
    st.info("Copy .env.example to .env and configure the Atlassian MCP endpoint and AWS credentials.")
    st.stop()

import os
from pathlib import Path
from atlassian_chatbot.rag.store import AtlassianFaissStore

_PERSIST_DIR = Path(
    os.getenv("LOCALAPPDATA")
    or os.getenv("XDG_DATA_HOME")
    or str(Path.home() / ".cache")
) / "TCS-Confluent-chatbot" / "faiss_store"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_actions" not in st.session_state:
    st.session_state.pending_actions = []
if "rag_store" not in st.session_state:
    loaded = AtlassianFaissStore.load(_PERSIST_DIR)
    st.session_state.rag_store = loaded if loaded and getattr(loaded, "index_version", None) == AtlassianFaissStore.INDEX_VERSION else None

active_config = render_sidebar(app_config, persist_dir=_PERSIST_DIR)
render_conversation(active_config, document_store=st.session_state.rag_store, persist_dir=_PERSIST_DIR)
st.caption("Connected tools are limited to the configured Atlassian Jira and Confluence MCP tools.")
