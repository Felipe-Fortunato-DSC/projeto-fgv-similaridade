"""Bootstrap one-time: sobe o parquet local de embeddings para o Snowflake.

Roda esse script **uma vez** após criar o schema/tabelas para reaproveitar
os embeddings já gerados localmente. A partir daí, a sincronização
incremental (botão na sidebar do app) cuida de adicionar apenas itens novos.

Uso:
    python scripts/seed_snowflake.py

O script é idempotente:
- Se um ``CD_INSUMO`` do parquet já estiver em ``TBL_INSUMOS_EMBEDDINGS``,
  é pulado.
- Pode ser rerodado a qualquer momento sem duplicar registros.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import snowflake_io as sf
from src.config import (
    DF_PAD_CSV,
    EMBEDDINGS_PARQUET,
    MEDIDA_CORRELACAO_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("seed_snowflake")


def main() -> int:
    if not EMBEDDINGS_PARQUET.exists():
        logger.error("Parquet local não encontrado em %s", EMBEDDINGS_PARQUET)
        logger.error("Esse script aproveita embeddings já gerados localmente. "
                     "Sem o parquet, use o botão 'Sincronizar base' no app.")
        return 1

    print(f"📦 Carregando parquet local: {EMBEDDINGS_PARQUET}")
    df_emb = pd.read_parquet(EMBEDDINGS_PARQUET)
    print(f"   → {len(df_emb):,} embeddings encontrados".replace(",", "."))

    df_pad = None
    if DF_PAD_CSV.exists():
        df_pad = pd.read_csv(DF_PAD_CSV).drop(columns=['Unnamed: 0'], errors='ignore')
        print(f"   → {len(df_pad):,} registros em df_pad.csv".replace(",", "."))

    df_medida = None
    if MEDIDA_CORRELACAO_CSV.exists():
        df_medida = pd.read_csv(MEDIDA_CORRELACAO_CSV)
        print(f"   → {len(df_medida):,} registros em medida_correlacao.csv".replace(",", "."))

    print("\n🔌 Conectando ao Snowflake...")
    conn = sf.conectar()
    try:
        print("📐 Garantindo tabelas (DDL idempotente)...")
        sf.garantir_tabelas(conn)

        cd_em_sf = sf.ler_cd_insumos_codificados(conn)
        cd_no_parquet = set(df_emb['CD_INSUMO'].astype(int).tolist())
        cd_novos = sorted(cd_no_parquet - cd_em_sf)

        print(
            f"\n📊 Estado atual:"
            f"\n   • Já em SF:       {len(cd_em_sf):,}".replace(",", ".")
            + f"\n   • No parquet:     {len(cd_no_parquet):,}".replace(",", ".")
            + f"\n   • A inserir:      {len(cd_novos):,}".replace(",", ".")
        )

        if not cd_novos:
            print("\n✅ Snowflake já tem todos os embeddings do parquet local. Nada a fazer.")
            return 0

        df_emb_novos = df_emb[df_emb['CD_INSUMO'].astype(int).isin(cd_novos)].copy()

        # 1) Embeddings
        print(f"\n⬆️  Subindo {len(df_emb_novos):,} embeddings...".replace(",", "."))
        n = sf.insert_embeddings_novos(conn, df_emb_novos[['CD_INSUMO', 'bert_vectors']])
        print(f"   → {n} linhas inseridas em TBL_INSUMOS_EMBEDDINGS")

        # 2) Preprocessados (mesmas colunas exceto bert_vectors)
        print("⬆️  Subindo preprocessados...")
        df_pre = df_emb_novos.drop(columns=['bert_vectors'])
        n = sf.insert_preprocessados_novos(conn, df_pre)
        print(f"   → {n} linhas inseridas em TBL_INSUMOS_PREPROCESSADOS")

        # 3) Padronizados (somente os CD_INSUMO novos)
        if df_pad is not None:
            print("⬆️  Subindo padronizados...")
            df_pad_novos = df_pad[df_pad['CD_INSUMO'].astype(int).isin(cd_novos)].copy()
            n = sf.insert_padronizados_novos(conn, df_pad_novos)
            print(f"   → {n} linhas inseridas em TBL_INSUMOS_PADRONIZADOS")
        else:
            print("⚠️  df_pad.csv ausente — TBL_INSUMOS_PADRONIZADOS não foi populada.")
            print("    Rode 'Sincronizar base' no app para preenchê-la a partir de TBL_INSUMOS.")

        # 4) Medidas correlação (substituição total)
        if df_medida is not None:
            print("⬆️  Regravando TBL_MEDIDAS_CORRELACAO (truncate + insert)...")
            n = sf.regravar_medida_correlacao(conn, df_medida)
            print(f"   → {n} linhas em TBL_MEDIDAS_CORRELACAO")

        print("\n✅ Bootstrap concluído.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
