from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
STAGING_DIR = DATA_DIR / "staging"
OUTPUT_DIR = DATA_DIR / "output"

CONSULTA_BP_CSV = INPUT_DIR / "consulta_bp.csv"
MEDIDA_CORRELACAO_CSV = STAGING_DIR / "medida_correlacao.csv"
DF_EMBEDDINGS_CSV = STAGING_DIR / "df_embeddings.csv"
DF_PAD_CSV = STAGING_DIR / "df_pad.csv"
EMBEDDINGS_PARQUET = STAGING_DIR / "embeddings_bp.parquet"

SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

for _d in (INPUT_DIR, STAGING_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
