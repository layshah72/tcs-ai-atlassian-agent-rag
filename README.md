# Atlassian AI Agent

A Streamlit chat interface that queries Jira and Confluence through an Atlassian MCP server. Jira issue creation and updates are staged as drafts; the app does not execute either write operation unless a user clicks **Approve and execute**.

The chat agent uses LangChain `create_agent`, the Amazon Bedrock chat model integration, and `langchain-mcp-adapters` to discover and call Atlassian MCP tools.

## Code structure

- `app.py` is the Streamlit entry point.
- `atlassian_chatbot/config/` loads and validates environment configuration and configures terminal logging.
- `atlassian_chatbot/auth/` contains OAuth provider setup and encrypted DPAPI token storage. `basic.py` and `bearer.py` each define a dedicated credential configuration class and header builder.
- `atlassian_chatbot/mcp/` configures the authenticated MCP connection.
- `atlassian_chatbot/agent/` separates agent orchestration, tool policy, and guarded tool wrappers.
- `atlassian_chatbot/rag/` chunks live MCP results, indexes them in a per-request FAISS store, and retrieves a bounded context for answer generation.
- `atlassian_chatbot/ui/` contains sidebar and conversation rendering.
- Root-level `agent.py`, `auth.py`, `config.py`, `mcp_connection.py`, and `atlassian_agent.py` are compatibility facades; new code should import from `atlassian_chatbot`.

## Requirements

- Python 3.10+
- AWS credentials with access to Amazon Bedrock. Ensure your IAM user or role has the `bedrock:InvokeModel` permission. See the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html).
- An Atlassian MCP endpoint that exposes the Jira and Confluence tools below, with authorization configured for your Atlassian account

The app only makes available the supported Jira/Confluence operations. Atlassian's hosted MCP currently publishes names such as `searchConfluence`, `getConfluenceContent`, `getJiraIssue`, `searchJiraIssuesUsingJql`, `createJiraIssue`, and `editJiraIssue`; the app maps those to its stable tool names. Unknown and generic tools (including `executeWrite` and `executeDestructive`) are excluded. Jira create/edit stay approval-gated.

## Configure

1. Create and activate a virtual environment.
2. Install packages from `requirements.txt`.
3. Copy `.env.example` to `.env` and set `AWS_REGION`, `BEDROCK_MODEL_ID`, `ATLASSIAN_MCP_TRANSPORT`, and the MCP connection settings. `BEDROCK_MODEL_ID` defaults to `anthropic.claude-3-5-sonnet-20241022-v2:0`; change it to a model enabled in your AWS account and region. Optionally set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` if not using instance/profile credentials.
4. Configure the Atlassian site once in `.env`: set `ATLASSIAN_SITE_URL=https://<your-site-name>.atlassian.net`. At startup, the app requests `https://<your-site-name>.atlassian.net/_edge/tenant_info` and reads `cloudId`. If you already know it, set `ATLASSIAN_CLOUD_ID` directly instead and omit the site URL. The app supplies this configured site/cloud ID to MCP tools that request it, so the agent should not ask which Jira or Confluence site to use.
5. For Atlassian's hosted MCP server, set `ATLASSIAN_MCP_TRANSPORT=streamable-http`, `ATLASSIAN_MCP_URL=https://mcp.atlassian.com/v2/mcp`, and `ATLASSIAN_MCP_AUTH_TYPE=oauth2`. The sidebar also lets you switch authentication methods at runtime: OAuth 2.0 browser sign-in, a Bearer access token, or a personal API token with account email using Basic auth. Entered tokens are password-masked and retained only in the current Streamlit session; alternatively, configure `ATLASSIAN_MCP_BEARER_TOKEN`, `ATLASSIAN_MCP_EMAIL`, and `ATLASSIAN_MCP_PERSONAL_API_TOKEN` in `.env`. OAuth tokens and client registration are encrypted with Windows DPAPI for the current Windows user and stored under `%LOCALAPPDATA%\\TCS-Confluent-chatbot\\oauth`. The local app uses one persistent credential namespace and is intended for a single user.

	Legacy static options remain available if your MCP endpoint requires them: `bearer` with `ATLASSIAN_MCP_BEARER_TOKEN`, or `basic` with `ATLASSIAN_MCP_EMAIL` and `ATLASSIAN_MCP_PERSONAL_API_TOKEN`. These are not interchangeable. Keep any static credential in `.env`; never commit it. OAuth browser callbacks are intended for a locally running, single-user Streamlit app; a hosted or multi-user deployment needs a public HTTPS callback and per-user secure token storage.

	When running in Docker or on Linux, OAuth tokens are encrypted with Fernet instead of Windows DPAPI. Set `OAUTH_STORAGE_SECRET` in `.env` to a stable random secret so tokens survive container restarts. Generate one with:

	```powershell
	python -c "import secrets; print(secrets.token_hex(32))"
	```

	If `OAUTH_STORAGE_SECRET` is not set, an ephemeral key is used and tokens are lost on restart. Keep this value secret and do not change it after tokens are stored; changing it requires re-authentication.
6. Alternatively, configure `ATLASSIAN_MCP_TRANSPORT=stdio`, `ATLASSIAN_MCP_COMMAND`, and JSON arrays/objects in `ATLASSIAN_MCP_ARGS` and `ATLASSIAN_MCP_ENV`. This is useful for an MCP server command that handles its own authentication. Keep secret values out of committed files.

The MCP server must expose the supported Jira/Confluence tools. Tool input schemas are discovered at runtime from the connected server. The site URL must be the HTTPS origin only (no path or query). If tenant-info lookup is blocked, configure `ATLASSIAN_CLOUD_ID` directly.

## Live retrieval (RAG)

The sidebar's **Knowledge index** form refreshes data through the configured Atlassian MCP endpoint. By default, it fetches the signed-in user's Jira issues updated in the last day and Confluence pages modified today; both JQL and the Confluence CQL/search query are editable. Required `cloudId`/site parameters are filled from app configuration before tool schema validation. MCP error responses are rejected and never indexed. The two successful searches are chunked and written to a replacement FAISS index for the current Streamlit session. If either search fails or both return no indexable results, the previous index remains active. The index is reused by subsequent chat turns in that browser session and is replaced at the next successful refresh; it is not a scheduled background job and is not shared across sessions or persisted to disk. Older in-memory indexes are discarded after the index validation version changes; refresh the index after restarting the updated app.

FastEmbed generates local neural embeddings using `BAAI/bge-small-en-v1.5`; the model weights are downloaded on first use and cached under `%LOCALAPPDATA%\TCS-Confluent-chatbot\models\fastembed` (override with `FASTEMBED_CACHE_DIR`). When the session index has data, ordinary read-only questions use only the FAISS retrieval tool—the agent does not connect to MCP or invoke remote read/search tools. Write-intent prompts may discover MCP tools but expose only approval-gated write tools alongside retrieval. Before an index exists, the agent may search Jira/Confluence live; those results are also indexed, and the full raw result is withheld from model context. Retrieval returns only a bounded set of relevant chunks. Embeddings run locally; document text is not sent to an embedding API.

If Atlassian returns `401 Unauthorized`, retry the OAuth sign-in/consent flow or verify that a configured legacy auth type matches its credential. Also ensure the Atlassian account has access to the hosted MCP service.

## Run

Start the app with Streamlit:

```powershell
streamlit run app.py
```

Ask a question in the chat. Use **Show my assigned Jira tickets** to prompt a Jira search using `assignee = currentUser()`. That JQL identifies the user represented by the configured Atlassian MCP connection, which may differ from the person using the UI if the server uses a shared account.

When a create/update request is detected, review the exact tool arguments in the approval panel. **Approve and execute** calls the Jira MCP tool; **Reject** discards the draft. No Jira write is performed before approval.

## Docker

The Docker image runs on Amazon Linux 2023. OAuth browser flow is not available in a container — use `bearer` or `basic` auth instead.

**1. Configure `.env`**

Ensure `.env` has static credentials and an `OAUTH_STORAGE_SECRET`:

```
ATLASSIAN_MCP_AUTH_TYPE=basic
ATLASSIAN_MCP_EMAIL=you@example.com
ATLASSIAN_MCP_PERSONAL_API_TOKEN=your-personal-api-token
OAUTH_STORAGE_SECRET=your-generated-secret
```

Generate `OAUTH_STORAGE_SECRET` with:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

**2. Build the image**

```powershell
docker build -t atlassian-chatbot .
```

**3. Run the container**

```powershell
docker run -d `
  --name atlassian-chatbot `
  -p 8501:8501 `
  -v atlassian-chatbot-data:/data `
  --env-file .env `
  atlassian-chatbot
```

Open `http://localhost:8501` in your browser.

**Useful commands**

```powershell
# View logs
docker logs -f atlassian-chatbot

# Stop
docker stop atlassian-chatbot

# Restart (keeps /data volume intact)
docker start atlassian-chatbot

# Delete container only (volume preserved)
docker rm atlassian-chatbot

# Delete container and wipe all persisted data
docker stop atlassian-chatbot
docker rm atlassian-chatbot
docker volume rm atlassian-chatbot-data
```

**Persisted data**

The named volume `atlassian-chatbot-data` is mounted at `/data` inside the container and survives container restarts and deletions. It stores:

- FAISS vector index — `/data/TCS-Confluent-chatbot/faiss_store/`
- OAuth tokens — `/data/TCS-Confluent-chatbot/oauth/`
- FastEmbed model weights — `/data/models/fastembed/`


## Tests

Run `pytest` from the project root. Tests cover configuration validation, OAuth wiring/token storage, tool allowlisting, approval gating, and the FAISS/FastEmbed retrieval pipeline.

## Security notes

- Run this app in a trusted environment; chat and tool results are sent to the configured Groq model.
- Tool-call logs include the agent system prompt, originating user prompt, and tool parameters in the Streamlit terminal. Credential-like values are redacted, but prompts and issue content may still contain sensitive information; protect terminal output and avoid sharing it.
- Provide only least-privilege Atlassian credentials to the MCP server.
- Approval is per operation and applies to the displayed tool arguments. Verify the target project, issue key, and field values before approving.
- Do not expose the Streamlit service publicly without adding authentication and appropriate network protections.
- Treat access and refresh tokens as secrets; they grant access as the Atlassian account that authorized the app.
-- cloud id can be ftech via https://your-site/_edge/tenant_info
