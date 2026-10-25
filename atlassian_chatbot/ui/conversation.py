from __future__ import annotations

import streamlit as st

from atlassian_chatbot.agent import AgentError, AtlassianAgent
from atlassian_chatbot.config import AppConfig, ConfigError
from atlassian_chatbot.rag import AtlassianFaissStore


def render_conversation(
    config: AppConfig, *, document_store: AtlassianFaissStore | None = None
) -> None:
    """Render chat history, approval actions, and the current prompt workflow."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    _render_pending_actions(config)
    prompt = st.chat_input("Ask about your Jira tickets or Confluence…")
    if not prompt and "next_prompt" in st.session_state:
        prompt = st.session_state.pop("next_prompt")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching Atlassian…"):
                result = AtlassianAgent(config).answer(
                    prompt,
                    history=st.session_state.messages[:-1],
                    document_store=document_store,
                )
            st.markdown(result.text)
            st.session_state.pending_actions.extend(result.pending_actions)
            st.session_state.messages.append({"role": "assistant", "content": result.text})
            if result.pending_actions:
                st.rerun()
        except (AgentError, ConfigError) as exc:
            st.error(str(exc))
            if "401 Unauthorized" in str(exc):
                st.info(
                    "Atlassian rejected the configured credentials. For OAuth 2.0, retry the browser "
                    "sign-in and consent flow. For legacy static credentials, ensure the selected auth "
                    "type matches the token and that the credential is valid for this MCP endpoint."
                )


def _render_pending_actions(config: AppConfig) -> None:
    for index, action in enumerate(st.session_state.pending_actions):
        with st.container(border=True):
            st.warning(f"Approval required: `{action['name']}`")
            st.json(action["arguments"])
            approve_col, reject_col = st.columns([1, 1])
            if approve_col.button("Approve and execute", key=f"approve_{index}", type="primary"):
                try:
                    with st.spinner("Executing approved change…"):
                        result = AtlassianAgent(config).execute_approved_action(
                            action["name"], action["arguments"], prompt=action.get("prompt")
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Approved operation completed.\n\n{result}"}
                    )
                    st.session_state.pending_actions.pop(index)
                    st.rerun()
                except AgentError as exc:
                    st.error(str(exc))
            if reject_col.button("Reject", key=f"reject_{index}"):
                st.session_state.pending_actions.pop(index)
                st.info("Draft rejected; no change was made.")
                st.rerun()
