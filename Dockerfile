FROM amazonlinux:2023

# ── system deps ──────────────────────────────────────────────────────────────
RUN dnf install -y python3.11 python3.11-pip shadow-utils && \
    dnf clean all

# ── non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# ── application code ──────────────────────────────────────────────────────────
COPY . .

# ── runtime dirs owned by appuser ─────────────────────────────────────────────
# /data is the mount point for persisted FAISS index and OAuth tokens
RUN mkdir -p /data/faiss_store /data/oauth && \
    chown -R appuser:appuser /data /app

USER appuser

# ── environment ───────────────────────────────────────────────────────────────
# Override XDG_DATA_HOME so FileOAuthTokenStorage and FAISS persist under /data
ENV XDG_DATA_HOME=/data \
    FASTEMBED_CACHE_DIR=/data/models/fastembed \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["python3.11", "-m", "streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true"]
