# =====================================================================
# Imagem de produção — Streamlit (Consulta por Similaridade FGV v2)
# Destino: Amazon ECR -> execução em Amazon ECS (Fargate ou EC2).
#
# Estratégia:
#  - Base slim + torch CPU-only (evita ~2 GB de wheels CUDA).
#  - Modelo SBERT e corpora NLTK são PRÉ-BAIXADOS no build => startup rápido
#    e sem dependência de rede externa em runtime.
#  - Os dados NÃO entram na imagem: tabulares vêm do Athena e os vetores do
#    S3 Vectors (busca k-NN gerenciada). Credenciais AWS via task role (ECS).
# =====================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models/hf \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app

# curl é usado pelo HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU-only ANTES do requirements para não puxar a build CUDA.
RUN pip install --no-cache-dir torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Dependências da aplicação (camada cacheável).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pré-download do modelo SBERT + corpora NLTK para dentro da imagem.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" \
    && python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data'); nltk.download('stopwords', download_dir='/opt/nltk_data')"

# Código da aplicação.
COPY . .

# Config AWS (defaults; sobrescreva no ECS/local conforme necessário).
# As CREDENCIAIS não vão aqui: no ECS vêm da task role; localmente do AWS CLI.
ENV AWS_REGION=us-east-1 \
    ATHENA_DATABASE=db_spdo_apps \
    S3_DATA_BASE=s3://ibre-spdo-coleta-bronze/apps/consulta_similaridade \
    S3_VECTORS_BUCKET=spdo-embeddings \
    S3_VECTORS_INDEX=insumos-sbert-384

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Healthcheck nativo do Streamlit.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
