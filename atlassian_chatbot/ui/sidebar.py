from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
from pathlib import Path

import streamlit as st

from atlassian_chatbot.config import AppConfig
from atlassian_chatbot.rag import AtlassianIngestionPipeline, IngestionError

logger = logging.getLogger(__name__)


def render_sidebar(config: AppConfig, *, persist_dir: "Path | None" = None) -> AppConfig:
    """Render connection controls and return the session-selected config."""
    with st.sidebar:
        st.subheader("Connection")
        st.caption(f"MCP transport: **{config.mcp_transport}**")
        if config.atlassian_site_url:
            st.caption(f"Atlassian site: **{config.atlassian_site_url}**")
        options = ["oauth2", "bearer", "basic"]
        default_auth = config.mcp_auth_type if config.mcp_auth_type in options else "oauth2"
        auth_type = st.selectbox(
            "Atlassian sign-in method",
            options=options,
            index=options.index(default_auth),
            format_func=lambda value: {
                "oauth2": "OAuth 2.0 (browser sign-in)",
                "bearer": "Access token (Bearer)",
                "basic": "Personal API token (Basic)",
            }[value],
            key="atlassian_auth_type",
        )
        bearer_token = config.mcp_bearer_token
        basic_email = config.mcp_basic_email
        personal_api_token = config.mcp_personal_api_token
        if auth_type == "oauth2":
            st.caption("The first Atlassian request opens a browser for sign-in and consent.")
        elif auth_type == "bearer":
            entered_token = st.text_input(
                "Atlassian MCP access token", type="password", key="atlassian_bearer_token_input",
                help="The token is kept only in this Streamlit session. Personal API tokens may require Basic auth instead.",
            ).strip()
            if entered_token:
                bearer_token = entered_token
            elif bearer_token:
                st.caption("Using the configured access token from the environment.")
            else:
                st.warning("Enter an access token to use token-based sign-in.")
        else:
            entered_email = st.text_input(
                "Atlassian account email", value=basic_email or "", key="atlassian_basic_email_input"
            ).strip()
            entered_token = st.text_input(
                "Atlassian personal API token", type="password", key="atlassian_basic_token_input",
                help="Personal API tokens use Basic auth with your Atlassian account email.",
            ).strip()
            if entered_email:
                basic_email = entered_email
            if entered_token:
                personal_api_token = entered_token
            if not basic_email or not personal_api_token:
                st.warning("Enter your Atlassian email and personal API token.")

        active_config = replace(
            config, mcp_auth_type=auth_type, mcp_bearer_token=bearer_token,
            mcp_basic_email=basic_email, mcp_personal_api_token=personal_api_token,
        )
        labels = {"oauth2": "OAuth 2.0", "bearer": "Bearer access token", "basic": "Personal API token (Basic)"}
        st.caption(f"Atlassian authentication: **{labels[auth_type]}**")
        st.caption(f"Bedrock model: **{config.bedrock_model_id}**")
        st.caption(f"AWS region: **{config.bedrock_region}**")
        st.divider()
        st.subheader("Knowledge index")
        if st.session_state.rag_store is not None:
            st.caption(f"Indexed chunks: **{st.session_state.rag_store.chunk_count}**")
            last_refresh = st.session_state.get("rag_last_refresh")
            if last_refresh:
                st.caption(f"Last refreshed: **{last_refresh}**")
        with st.form("refresh_atlassian_index"):
            jira_jql = st.text_area(
                "Daily Jira JQL",
                value="assignee = currentUser() AND updated >= -1d ORDER BY updated DESC",
                help="Choose which Jira issues to fetch into the session's vector index.",
                height=90,
            )
            confluence_query = st.text_area(
                "Daily Confluence CQL / query",
                value="type = page AND lastModified >= startOfDay() ORDER BY lastModified DESC",
                help="Enter a Confluence query accepted by your MCP search tool. Results replace the current session index.",
                height=90,
            )
            refresh_index = st.form_submit_button(
                "Refresh Jira and Confluence index",
                type="primary",
                width="stretch",
                icon=":material/refresh:",
            )
        if refresh_index:
            try:
                with st.spinner("Fetching Jira and Confluence data and rebuilding the index…"):
                    refreshed = AtlassianIngestionPipeline(active_config).refresh(
                        jira_jql, confluence_query
                    )
                st.session_state.rag_store = refreshed.store
                if persist_dir is not None:
                    refreshed.store.save(persist_dir)
                st.session_state.rag_last_refresh = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                st.success(
                    "Index refreshed: "
                    f"{refreshed.jira_chunks} Jira chunks and "
                    f"{refreshed.confluence_chunks} Confluence chunks."
                )
            except IngestionError as exc:
                st.error(str(exc))
            except Exception as exc:
                logger.exception("Unexpected Atlassian index refresh error")
                st.error(f"Could not refresh the Jira and Confluence index: {exc}")

        st.divider()
        st.markdown("**Try asking**")
        if st.button("Show my assigned Jira tickets", width="stretch"):
            st.session_state.next_prompt = (
                "Find Jira issues assigned to me. Use Jira JQL assignee = currentUser() "
                "ORDER BY updated DESC, summarize the results, and include issue keys and statuses."
            )
        if st.button("Show recent Confluence pages", width="stretch"):
            st.session_state.next_prompt = (
                "Search Confluence for pages modified today. Summarize each page title, "
                "the space it belongs to, and a brief description of its content."
            )
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.pending_actions = []
            st.rerun()
        st.divider()
        st.caption("Jira create/update operations are drafts until approved below.")
    return active_config
