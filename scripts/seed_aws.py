"""Seed one-time: artefatos locais → Athena (Iceberg) + S3 Vectors.

Aproveita os embeddings/CSVs já gerados localmente (em ``data/staging``) para
popular a AWS, permitindo testar a aplicação localmente contra Athena + S3
Vectors sem reprocessar tudo.

Popula:
  - S3 Vectors (índice de embeddings) + manifesto ``tbl_insumos_embeddings``
  - ``tbl_insumos_preprocessados`` (a partir do parquet)
  - ``tbl_insumos_padronizados``    (a partir de df_pad.csv)
  - ``tbl_medidas_correlacao``      (a partir de medida_correlacao.csv)

NÃO popula ``tbl_insumos`` (fonte bruta) — essa é a migração upstream, feita à
parte. Sem ela, o botão "Sincronizar base" do app reporta "base já atualizada"
(esperado), mas a CONSULTA já funciona com o que foi semeado aqui.

Pré-requisitos:
  - Tabelas Iceberg criadas (deploy/athena/schema.sql) — ou serão criadas pelo
    awswrangler na primeira escrita.
  - Índice no S3 Vectors (deploy/athena/s3vectors_setup.md) — este script tenta
    criá-lo se faltar.
  - Credenciais AWS no ambiente (aws configure / variáveis).

Uso:
    python scripts/seed_aws.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import aws_io
from src.config import (
    DF_PAD_CSV,
    EMBEDDING_DIM,
    EMBEDDINGS_PARQUET,
    MEDIDA_CORRELACAO_CSV,
    S3_VECTORS_BUCKET,
    S3_VECTORS_INDEX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("seed_aws")


def _garantir_indice() -> None:
    """Cria o índice no S3 Vectors se ainda não existir (idempotente)."""
    client = aws_io._s3vectors()
    try:
        client.create_index(
            vectorBucketName=S3_VECTORS_BUCKET,
            indexName=S3_VECTORS_INDEX,
            dataType="float32",
            dimension=EMBEDDING_DIM,
            distanceMetric="cosine",
        )
        print(f"   → índice '{S3_VECTORS_INDEX}' criado em '{S3_VECTORS_BUCKET}'")
    except Exception as e:
        # Já existe (ConflictException) ou erro de permissão — segue e deixa
        # o put_vectors falhar depois se for problema real.
        print(f"   → índice não criado agora ({type(e).__name__}); assumindo que já existe")


def main() -> int:
    if not EMBEDDINGS_PARQUET.exists():
        logger.error("Parquet local não encontrado: %s", EMBEDDINGS_PARQUET)
        return 1

    print(f"📦 Lendo parquet local: {EMBEDDINGS_PARQUET}")
    df_emb = pd.read_parquet(EMBEDDINGS_PARQUET)
    df_emb.columns = [c.upper() if c != "bert_vectors" else c for c in df_emb.columns]
    print(f"   → {len(df_emb):,} embeddings".replace(",", "."))

    print("\n🔌 Verificando S3 Vectors...")
    _garantir_indice()

    print("📐 Lendo manifesto atual (idempotência)...")
    cd_existentes = aws_io.ler_cds_codificados()
    cd_no_parquet = set(df_emb["CD_INSUMO"].astype(int).tolist())
    cd_novos = sorted(cd_no_parquet - cd_existentes)
    print(
        f"   • Já na AWS:  {len(cd_existentes):,}".replace(",", ".")
        + f"\n   • No parquet: {len(cd_no_parquet):,}".replace(",", ".")
        + f"\n   • A inserir:  {len(cd_novos):,}".replace(",", ".")
    )

    if not cd_novos:
        print("\n✅ Nada novo a semear. S3 Vectors já está em dia com o parquet.")
    else:
        df_novos = df_emb[df_emb["CD_INSUMO"].astype(int).isin(cd_novos)].copy()

        print(f"\n⬆️  Gravando {len(df_novos):,} vetores no S3 Vectors...".replace(",", "."))
        n = aws_io.put_vectors(df_novos[["CD_INSUMO", "bert_vectors"]])
        print(f"   → {n} vetores gravados (+ manifesto no Athena)")

        print("⬆️  Gravando tbl_insumos_preprocessados...")
        n = aws_io.insert_preprocessados_novos(df_novos.drop(columns=["bert_vectors"]))
        print(f"   → {n} linhas")

        if DF_PAD_CSV.exists():
            print("⬆️  Gravando tbl_insumos_padronizados...")
            df_pad = pd.read_csv(DF_PAD_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
            df_pad.columns = [c.upper() for c in df_pad.columns]
            df_pad_novos = df_pad[df_pad["CD_INSUMO"].astype(int).isin(cd_novos)].copy()
            n = aws_io.insert_padronizados_novos(df_pad_novos)
            print(f"   → {n} linhas")
        else:
            print("⚠️  df_pad.csv ausente — tbl_insumos_padronizados não populada.")

    if MEDIDA_CORRELACAO_CSV.exists():
        print("⬆️  Regravando tbl_medidas_correlacao...")
        df_med = pd.read_csv(MEDIDA_CORRELACAO_CSV)
        df_med.columns = [c.upper() for c in df_med.columns]
        n = aws_io.regravar_medidas(df_med)
        print(f"   → {n} linhas")

    print("\n✅ Seed concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
