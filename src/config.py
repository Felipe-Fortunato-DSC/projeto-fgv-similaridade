from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
STAGING_DIR = DATA_DIR / "staging"
OUTPUT_DIR = DATA_DIR / "output"
TRAINING_DIR = DATA_DIR / "training"
EVAL_DIR = DATA_DIR / "eval"

CONSULTA_BP_CSV = INPUT_DIR / "consulta_bp.csv"
MEDIDA_CORRELACAO_CSV = STAGING_DIR / "medida_correlacao.csv"
DF_EMBEDDINGS_CSV = STAGING_DIR / "df_embeddings.csv"
DF_PAD_CSV = STAGING_DIR / "df_pad.csv"
EMBEDDINGS_PARQUET = STAGING_DIR / "embeddings_bp.parquet"

FEEDBACK_JSONL = TRAINING_DIR / "feedback.jsonl"
EVAL_GOLD_CSV = EVAL_DIR / "gold_standard.csv"
EVAL_TEMPLATE_CSV = EVAL_DIR / "gold_standard_template.csv"

SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Pesos default da combinação linear do score final.
# Σ deve ser 1 (caso contrário o app renormaliza). Racional:
#   sbert  → captura semântica da descrição (sinônimos, paráfrases) — sinal principal
#   desc   → 0 por default: SBERT já cobre semântica de descrição; sim_token agrega
#            pouco e tende a penalizar paráfrases legítimas. Pode subir se necessário.
#   marca  → atributo curto, alta precisão quando informado
#   medida → comparação numérica com tolerância, alta precisão quando informado
DEFAULT_WEIGHTS = {
    "sbert": 0.5,
    "desc": 0.0,
    "marca": 0.25,
    "medida": 0.25,
}
DEFAULT_THRESHOLD = 50.0

APP_VERSION = "v2"

for _d in (INPUT_DIR, STAGING_DIR, OUTPUT_DIR, TRAINING_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
