FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src \
    HF_HOME=/app/hf PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first (cached layer), CPU torch comes from pyproject
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# bake the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# code + data (chunks.json and the Chroma index)
COPY src ./src
COPY data/processed/chunks.json ./data/processed/chunks.json
COPY data/chroma ./data/chroma

EXPOSE 8000
CMD ["uvicorn", "arabic_legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
