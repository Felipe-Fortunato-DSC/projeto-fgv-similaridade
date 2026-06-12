# syntax=docker/dockerfile:1
# Imagem da aplicação de consulta por similaridade (Streamlit + SBERT + Snowflake).
# Build CPU-only: torch é instalado do índice CPU para evitar pacotes CUDA (~GBs).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Caches de modelos/NLTK dentro da imagem (pré-baixados no build)
    HF_HOME=/opt/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sbert \
    NLTK_DATA=/opt/nltk_data \
    # Streamlit headless
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1: runtime do OpenMP exigido pelo torch CPU. curl: usado no HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch CPU primeiro — assim o resolver não puxa a variante CUDA (nvidia-*).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) demais dependências (sentence-transformers já encontra o torch CPU instalado).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3) Pré-download de artefatos pesados para acelerar o cold start e não depender
#    de rede no boot: corpora NLTK + modelo SBERT multilíngue.
RUN python -m nltk.downloader -d "$NLTK_DATA" punkt_tab stopwords \
    && python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# 4) Código da aplicação (segredos ficam de fora via .dockerignore).
COPY . .

# Usuário não-root com posse de /app e dos caches de modelo.
RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app /opt/models /opt/nltk_data
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
