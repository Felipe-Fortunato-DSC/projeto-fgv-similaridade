"""Configuração central — paths locais, parâmetros do modelo e ambiente AWS.

A fonte de verdade dos dados é a AWS:
  - **Athena/Iceberg** (database ``db_spdo_apps``) para dados tabulares.
  - **S3 Vectors** (bucket ``spdo-embeddings``) para os embeddings/busca k-NN.

Todos os parâmetros de ambiente podem ser sobrescritos por variáveis de
ambiente (útil no ECS/local). Em ECS, as credenciais AWS vêm da *task role*;
localmente, do perfil padrão do AWS CLI (``aws configure``).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------- Paths locais ----------------------
# Usados pelo script de seed (lê os artefatos já gerados) e pelo buffer de
# feedback. O app NÃO depende mais de cache local de embeddings.
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
STAGING_DIR = DATA_DIR / "staging"
OUTPUT_DIR = DATA_DIR / "output"
TRAINING_DIR = DATA_DIR / "training"
EVAL_DIR = DATA_DIR / "eval"

# Artefatos locais consumidos pelo seed (scripts/seed_aws.py).
EMBEDDINGS_PARQUET = STAGING_DIR / "embeddings_bp.parquet"
DF_PAD_CSV = STAGING_DIR / "df_pad.csv"
MEDIDA_CORRELACAO_CSV = STAGING_DIR / "medida_correlacao.csv"

FEEDBACK_JSONL = TRAINING_DIR / "feedback.jsonl"
EVAL_GOLD_CSV = EVAL_DIR / "gold_standard.csv"
EVAL_TEMPLATE_CSV = EVAL_DIR / "gold_standard_template.csv"

# ---------------------- Modelo SBERT ----------------------
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # dimensão do paraphrase-multilingual-MiniLM-L12-v2

# ---------------------- Ambiente AWS ----------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Athena / Glue
ATHENA_DATABASE = os.getenv("ATHENA_DATABASE", "db_spdo_apps")
S3_DATA_BASE = os.getenv(
    "S3_DATA_BASE",
    "s3://ibre-spdo-coleta-bronze/apps/consulta_similaridade",
).rstrip("/")
# Local dos resultados de query do Athena (workgroup pode já definir isto).
ATHENA_RESULTS_LOCATION = os.getenv(
    "ATHENA_RESULTS_LOCATION", f"{S3_DATA_BASE}/_athena_results/"
)

# S3 Vectors
S3_VECTORS_BUCKET = os.getenv("S3_VECTORS_BUCKET", "spdo-embeddings")
S3_VECTORS_INDEX = os.getenv("S3_VECTORS_INDEX", "insumos-sbert-384")

# ---------------------- Scoring ----------------------
# Pesos default da combinação linear do score final.
# Σ deve ser 1 (caso contrário o app renormaliza). Racional:
#   sbert  → captura semântica da descrição (sinônimos, paráfrases) — sinal principal
#   desc   → 0 por default: SBERT já cobre semântica de descrição.
#   marca  → atributo curto, alta precisão quando informado
#   medida → comparação numérica com tolerância, alta precisão quando informado
DEFAULT_WEIGHTS = {
    "sbert": 0.5,
    "desc": 0.0,
    "marca": 0.25,
    "medida": 0.25,
}
DEFAULT_THRESHOLD = 50.0

# Top-K default da busca vetorial (o S3 Vectors limita o teto por consulta).
DEFAULT_TOP_K = 30

APP_VERSION = "v3-aws"

for _d in (INPUT_DIR, STAGING_DIR, OUTPUT_DIR, TRAINING_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
