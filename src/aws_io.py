"""Camada de dados AWS — Athena (Iceberg) + S3 Vectors.

Substitui o antigo ``snowflake_io``. Responsabilidades:

- **Athena/Iceberg** (``awswrangler``): dados tabulares (padronizados,
  preprocessados, medidas, feedback) e o *manifesto* de embeddings
  (``tbl_insumos_embeddings`` — só metadados, sem o vetor).
- **S3 Vectors** (``boto3`` client ``s3vectors``): armazenamento e busca k-NN
  dos vetores SBERT. A chave de cada vetor é o ``cd_insumo``.

Credenciais: *task role* no ECS; perfil padrão do AWS CLI localmente.

Convenção de colunas: o Athena devolve nomes em minúsculo; aqui as leituras
são normalizadas para MAIÚSCULAS (CD_INSUMO, INSUMO_DESCRICAO, ...) para o
restante do app não precisar mudar.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Iterable

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

from .config import (
    ATHENA_DATABASE,
    ATHENA_RESULTS_LOCATION,
    AWS_REGION,
    EMBEDDING_DIM,
    S3_DATA_BASE,
    S3_VECTORS_BUCKET,
    S3_VECTORS_INDEX,
    SBERT_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# ---------------------- Nomes de tabela (Athena, minúsculo) ----------------------
TBL_INSUMOS = "tbl_insumos"
TBL_PADRONIZADOS = "tbl_insumos_padronizados"
TBL_PREPROCESSADOS = "tbl_insumos_preprocessados"
TBL_EMBEDDINGS = "tbl_insumos_embeddings"  # manifesto (sem vetor)
TBL_MEDIDAS = "tbl_medidas_correlacao"
TBL_FEEDBACK = "tbl_feedback_validacoes"

_PUT_BATCH = 200  # vetores por chamada put_vectors (verificar teto do serviço)

_session = boto3.Session(region_name=AWS_REGION)


def _now() -> datetime:
    """UTC naive — casa com o tipo ``timestamp`` (sem tz) do Iceberg."""
    return datetime.utcnow()


def _table_location(table: str) -> str:
    return f"{S3_DATA_BASE}/{table}/"


def _temp_path(table: str) -> str:
    return f"{S3_DATA_BASE}/_tmp/{table}/"


def _s3vectors():
    return _session.client("s3vectors")


# ---------------------- Leitura Athena ----------------------

def _read_sql(sql: str) -> pd.DataFrame:
    df = wr.athena.read_sql_query(
        sql,
        database=ATHENA_DATABASE,
        s3_output=ATHENA_RESULTS_LOCATION,
        ctas_approach=False,
        boto3_session=_session,
    )
    df.columns = [c.upper() for c in df.columns]
    return df


def contar_totais() -> tuple[int, int]:
    """(n_em_tbl_insumos, n_no_manifesto_de_embeddings)."""
    sql = f"""
        SELECT
            (SELECT count(*) FROM {TBL_INSUMOS})    AS n_brutos,
            (SELECT count(*) FROM {TBL_EMBEDDINGS})  AS n_codificados
    """
    df = _read_sql(sql)
    return int(df.iloc[0]["N_BRUTOS"]), int(df.iloc[0]["N_CODIFICADOS"])


def contar_codificados() -> int:
    df = _read_sql(f"SELECT count(*) AS n FROM {TBL_EMBEDDINGS}")
    return int(df.iloc[0]["N"])


def ler_cds_codificados() -> set[int]:
    """Conjunto de cd_insumo já presentes no manifesto (para idempotência)."""
    try:
        df = _read_sql(f"SELECT cd_insumo FROM {TBL_EMBEDDINGS}")
        if df.empty:
            return set()
        return set(df["CD_INSUMO"].astype(int).tolist())
    except Exception as e:
        logger.warning("Manifesto provavelmente vazio/ausente: %s", e)
        return set()


def ler_insumos_brutos_novos() -> pd.DataFrame:
    """Diff server-side: cd_insumo de tbl_insumos ainda não codificados."""
    sql = f"""
        SELECT i.*
        FROM {TBL_INSUMOS} i
        LEFT JOIN {TBL_EMBEDDINGS} e ON i.cd_insumo = e.cd_insumo
        WHERE e.cd_insumo IS NULL
    """
    return _read_sql(sql)


def ler_padronizados() -> pd.DataFrame:
    """tbl_insumos_padronizados (df_pad usado no display)."""
    df = _read_sql(f"SELECT * FROM {TBL_PADRONIZADOS}")
    return df.drop(columns=["UPDATED_AT"], errors="ignore")


def ler_preprocessados() -> pd.DataFrame:
    """tbl_insumos_preprocessados (campos para a penalização; sem vetor)."""
    df = _read_sql(f"SELECT * FROM {TBL_PREPROCESSADOS}")
    return df.drop(columns=["UPDATED_AT"], errors="ignore")


def ler_medidas() -> pd.DataFrame:
    return _read_sql(f"SELECT cd_medida, medida FROM {TBL_MEDIDAS}")


def ler_medidas_distintas() -> pd.DataFrame:
    return _read_sql(f"SELECT DISTINCT cd_medida, medida FROM {TBL_INSUMOS}")


def ha_diferenca_medidas() -> bool:
    sql = f"""
        SELECT
            (SELECT count(DISTINCT cd_medida || '|' || medida) FROM {TBL_INSUMOS})
          - (SELECT count(*) FROM {TBL_MEDIDAS}) AS diff
    """
    df = _read_sql(sql)
    return int(df.iloc[0]["DIFF"]) != 0


# ---------------------- Escrita Athena (Iceberg) ----------------------

def _to_iceberg(df: pd.DataFrame, table: str, mode: str = "append") -> int:
    if df.empty:
        return 0
    df_out = df.copy()
    df_out.columns = [c.lower() for c in df_out.columns]
    wr.athena.to_iceberg(
        df=df_out,
        database=ATHENA_DATABASE,
        table=table,
        table_location=_table_location(table),
        temp_path=_temp_path(table),
        mode=mode,
        boto3_session=_session,
    )
    return len(df_out)


def insert_padronizados_novos(df: pd.DataFrame) -> int:
    cols = ["GRP_INSUMO", "CD_INSUMO", "INSUMO_DESCRICAO", "MARCA", "MEDIDA"]
    if "STATUS" in df.columns:
        cols.append("STATUS")
    out = df[cols].copy()
    out["UPDATED_AT"] = _now()
    return _to_iceberg(out, TBL_PADRONIZADOS, mode="append")


def insert_preprocessados_novos(df: pd.DataFrame) -> int:
    cols = ["GRP_INSUMO", "CD_INSUMO", "INSUMO_DESCRICAO", "MARCA", "MEDIDA"]
    out = df[cols].copy()
    out["UPDATED_AT"] = _now()
    return _to_iceberg(out, TBL_PREPROCESSADOS, mode="append")


def regravar_medidas(df: pd.DataFrame) -> int:
    """Substituição total (overwrite) da tabela de medidas."""
    return _to_iceberg(df[["CD_MEDIDA", "MEDIDA"]], TBL_MEDIDAS, mode="overwrite")


def _registrar_embeddings(cd_insumos: Iterable[int]) -> int:
    """Insere o manifesto (sem vetor) no Athena para o diff incremental."""
    cds = [int(c) for c in cd_insumos]
    if not cds:
        return 0
    reg = pd.DataFrame({
        "CD_INSUMO": cds,
        "MODEL_NAME": SBERT_MODEL_NAME,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "VECTOR_KEY": [str(c) for c in cds],
        "CREATED_AT": _now(),
    })
    return _to_iceberg(reg, TBL_EMBEDDINGS, mode="append")


# ---------------------- S3 Vectors ----------------------

def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def put_vectors(df: pd.DataFrame) -> int:
    """Grava vetores no S3 Vectors e registra o manifesto no Athena.

    df precisa ter ``CD_INSUMO`` + ``bert_vectors`` (list/np.array de floats).
    """
    if df.empty:
        return 0

    invalid = df["bert_vectors"].apply(lambda v: len(v) != EMBEDDING_DIM)
    if bool(invalid.any()):
        bad = df.loc[invalid, "CD_INSUMO"].tolist()
        raise ValueError(
            f"{len(bad)} embeddings com dimensão != {EMBEDDING_DIM}. "
            f"CD_INSUMO (até 5): {bad[:5]}"
        )

    vectors = [
        {
            "key": str(int(cd)),
            "data": {"float32": [float(x) for x in vec]},
        }
        for cd, vec in zip(df["CD_INSUMO"], df["bert_vectors"])
    ]

    client = _s3vectors()
    n = 0
    for chunk in _chunks(vectors, _PUT_BATCH):
        client.put_vectors(
            vectorBucketName=S3_VECTORS_BUCKET,
            indexName=S3_VECTORS_INDEX,
            vectors=chunk,
        )
        n += len(chunk)

    _registrar_embeddings(df["CD_INSUMO"].tolist())
    return n


def query_vectors(embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Busca k-NN no S3 Vectors. Retorna [(cd_insumo, distancia_cosseno), ...].

    distancia_cosseno = 1 - cos_sim (menor = mais similar).
    """
    client = _s3vectors()
    resp = client.query_vectors(
        vectorBucketName=S3_VECTORS_BUCKET,
        indexName=S3_VECTORS_INDEX,
        queryVector={"float32": [float(x) for x in np.asarray(embedding).ravel()]},
        topK=int(top_k),
        returnDistance=True,
        returnMetadata=False,
    )
    out: list[tuple[int, float]] = []
    for v in resp.get("vectors", []):
        out.append((int(v["key"]), float(v.get("distance", 0.0))))
    return out


# ---------------------- Feedback ----------------------

def insert_feedback(registros: list[dict]) -> int:
    if not registros:
        return 0
    df = pd.DataFrame([
        {
            "FEEDBACK_ID": r.get("feedback_id"),
            "TIMESTAMP_UTC": pd.to_datetime(r.get("timestamp"), utc=True).tz_convert(None),
            "SESSION_ID": r.get("session_id"),
            "USER_DESCRICAO": (r.get("user_input") or {}).get("descricao"),
            "USER_MARCA": (r.get("user_input") or {}).get("marca"),
            "USER_MEDIDA": (r.get("user_input") or {}).get("medida"),
            "MATCH_CD_INSUMO": (r.get("match") or {}).get("cd_insumo"),
            "MATCH_GRP_INSUMO": (r.get("match") or {}).get("grp_insumo"),
            "MATCH_DESCRICAO": (r.get("match") or {}).get("descricao"),
            "MATCH_MARCA": (r.get("match") or {}).get("marca"),
            "MATCH_MEDIDA": (r.get("match") or {}).get("medida"),
            "MATCH_STATUS": (r.get("match") or {}).get("status"),
            "SCORE_SBERT": (r.get("scores") or {}).get("sbert"),
            "SCORE_DESC_TOKENS": (r.get("scores") or {}).get("desc_tokens"),
            "SCORE_MARCA_TOKENS": (r.get("scores") or {}).get("marca_tokens"),
            "SCORE_MEDIDA_NUM": (r.get("scores") or {}).get("medida_numeric"),
            "SCORE_FINAL": (r.get("scores") or {}).get("final"),
            "RANK_POSICAO": r.get("rank"),
            "LABEL": int(r.get("label")) if r.get("label") is not None else None,
            "APP_VERSION": r.get("app_version"),
            "KNN_K": r.get("knn_k"),
            "WEIGHTS_SNAPSHOT": json.dumps(r.get("weights_snapshot") or {}),
        }
        for r in registros
    ])
    return _to_iceberg(df, TBL_FEEDBACK, mode="append")


def ler_feedback() -> pd.DataFrame:
    df = _read_sql(f"SELECT * FROM {TBL_FEEDBACK} ORDER BY timestamp_utc")
    if df.empty:
        return df
    return df.rename(columns={
        "FEEDBACK_ID": "feedback_id",
        "TIMESTAMP_UTC": "timestamp",
        "SESSION_ID": "session_id",
        "USER_DESCRICAO": "user_input.descricao",
        "USER_MARCA": "user_input.marca",
        "USER_MEDIDA": "user_input.medida",
        "MATCH_CD_INSUMO": "match.cd_insumo",
        "MATCH_GRP_INSUMO": "match.grp_insumo",
        "MATCH_DESCRICAO": "match.descricao",
        "MATCH_MARCA": "match.marca",
        "MATCH_MEDIDA": "match.medida",
        "MATCH_STATUS": "match.status",
        "SCORE_SBERT": "scores.sbert",
        "SCORE_DESC_TOKENS": "scores.desc_tokens",
        "SCORE_MARCA_TOKENS": "scores.marca_tokens",
        "SCORE_MEDIDA_NUM": "scores.medida_numeric",
        "SCORE_FINAL": "scores.final",
        "RANK_POSICAO": "rank",
        "LABEL": "label",
        "APP_VERSION": "app_version",
        "KNN_K": "knn_k",
        "WEIGHTS_SNAPSHOT": "weights_snapshot",
    })
