"""Integração com Snowflake — fonte de verdade da base de insumos.

Credenciais lidas de ``.streamlit/secrets.toml`` (seção ``[snowflake]``).
Dentro do Streamlit, usa ``st.secrets``; em scripts standalone, lê o TOML
diretamente.

Tabelas no schema configurado (esperado: BASES_SPDO.DB_GESTAO_BANCO_PRECO_APP_CONSULTA):

- ``TBL_INSUMOS``                  — fonte bruta (leitura)
- ``TBL_INSUMOS_PADRONIZADOS``     — equivalente a df_pad.csv
- ``TBL_INSUMOS_PREPROCESSADOS``   — equivalente a df_embeddings.csv
- ``TBL_INSUMOS_EMBEDDINGS``       — equivalente a embeddings_bp.parquet (ARRAY)
- ``TBL_MEDIDAS_CORRELACAO``       — equivalente a medida_correlacao.csv
"""
from __future__ import annotations

import json
import logging
import time
import tomllib

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, SBERT_MODEL_NAME

logger = logging.getLogger(__name__)

SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"

TBL_INSUMOS = "TBL_INSUMOS"
TBL_INSUMOS_PADRONIZADOS = "TBL_INSUMOS_PADRONIZADOS"
TBL_INSUMOS_PREPROCESSADOS = "TBL_INSUMOS_PREPROCESSADOS"
TBL_INSUMOS_EMBEDDINGS = "TBL_INSUMOS_EMBEDDINGS"
TBL_MEDIDAS_CORRELACAO = "TBL_MEDIDAS_CORRELACAO"
TBL_FEEDBACK_VALIDACOES = "TBL_FEEDBACK_VALIDACOES"

EMBEDDING_DIM = 384  # SBERT paraphrase-multilingual-MiniLM-L12-v2


# ---------------------- Credenciais & conexão ----------------------

def _config_do_ambiente() -> dict | None:
    """Lê credenciais de variáveis de ambiente (SNOWFLAKE_*).

    Mecanismo usado no deploy em container (ECS/Docker), onde os segredos são
    injetados em runtime — sem arquivo no repositório nem na imagem. Retorna
    ``None`` se as variáveis obrigatórias (account/user/password) não estiverem
    todas presentes, para que o caller faça fallback ao arquivo TOML.
    """
    import os
    mapa = {
        "account": "SNOWFLAKE_ACCOUNT",
        "user": "SNOWFLAKE_USER",
        "password": "SNOWFLAKE_PASSWORD",
        "role": "SNOWFLAKE_ROLE",
        "warehouse": "SNOWFLAKE_WAREHOUSE",
        "database": "SNOWFLAKE_DATABASE",
        "schema": "SNOWFLAKE_SCHEMA",
    }
    cfg = {k: os.environ[v] for k, v in mapa.items() if os.environ.get(v)}
    if not all(cfg.get(k) for k in ("account", "user", "password")):
        return None
    return cfg


def load_config() -> dict:
    """Carrega config Snowflake.

    Ordem de prioridade:
    1. ``st.secrets`` (Streamlit Community Cloud / .streamlit/secrets.toml),
    2. variáveis de ambiente ``SNOWFLAKE_*`` (deploy em container),
    3. arquivo ``.streamlit/secrets.toml`` (dev local / mount em runtime).
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "snowflake" in st.secrets:
            return dict(st.secrets["snowflake"])
    except Exception:
        pass

    cfg_env = _config_do_ambiente()
    if cfg_env is not None:
        return cfg_env

    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Credenciais Snowflake não encontradas. Forneça uma das opções: "
            f"variáveis de ambiente SNOWFLAKE_ACCOUNT/USER/PASSWORD/... ou o "
            f"arquivo {SECRETS_PATH} com seção [snowflake]."
        )
    with SECRETS_PATH.open("rb") as f:
        data = tomllib.load(f)
    cfg = data.get("snowflake") or data.get("connections", {}).get("snowflake")
    if not cfg:
        raise KeyError("Seção [snowflake] ausente em secrets.toml")
    return dict(cfg)


def conectar():
    """Cria conexão Snowflake. Caller responsável por fechar."""
    from snowflake import connector
    cfg = load_config()
    return connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        password=cfg["password"],
        warehouse=cfg.get("warehouse"),
        database=cfg.get("database"),
        schema=cfg.get("schema"),
        role=cfg.get("role"),
    )


# ---------------------- DDL ----------------------

_DDL: dict[str, str] = {
    TBL_INSUMOS_PADRONIZADOS: f"""
        CREATE TABLE IF NOT EXISTS {TBL_INSUMOS_PADRONIZADOS} (
            GRP_INSUMO VARCHAR,
            CD_INSUMO NUMBER,
            INSUMO_DESCRICAO VARCHAR,
            MARCA VARCHAR,
            MEDIDA VARCHAR,
            STATUS VARCHAR,
            UPDATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """,
    TBL_INSUMOS_PREPROCESSADOS: f"""
        CREATE TABLE IF NOT EXISTS {TBL_INSUMOS_PREPROCESSADOS} (
            GRP_INSUMO VARCHAR,
            CD_INSUMO NUMBER,
            INSUMO_DESCRICAO VARCHAR,
            MARCA VARCHAR,
            MEDIDA VARCHAR,
            UPDATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """,
    TBL_INSUMOS_EMBEDDINGS: f"""
        CREATE TABLE IF NOT EXISTS {TBL_INSUMOS_EMBEDDINGS} (
            CD_INSUMO NUMBER PRIMARY KEY,
            BERT_VECTOR ARRAY,
            MODEL_NAME VARCHAR,
            EMBEDDING_DIM NUMBER,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """,
    TBL_MEDIDAS_CORRELACAO: f"""
        CREATE TABLE IF NOT EXISTS {TBL_MEDIDAS_CORRELACAO} (
            CD_MEDIDA VARCHAR,
            MEDIDA VARCHAR
        )
    """,
    TBL_FEEDBACK_VALIDACOES: f"""
        CREATE TABLE IF NOT EXISTS {TBL_FEEDBACK_VALIDACOES} (
            FEEDBACK_ID VARCHAR PRIMARY KEY,
            TIMESTAMP_UTC TIMESTAMP_LTZ,
            SESSION_ID VARCHAR,
            USER_DESCRICAO VARCHAR,
            USER_MARCA VARCHAR,
            USER_MEDIDA VARCHAR,
            MATCH_CD_INSUMO NUMBER,
            MATCH_GRP_INSUMO VARCHAR,
            MATCH_DESCRICAO VARCHAR,
            MATCH_MARCA VARCHAR,
            MATCH_MEDIDA VARCHAR,
            MATCH_STATUS VARCHAR,
            SCORE_SBERT FLOAT,
            SCORE_DESC_TOKENS FLOAT,
            SCORE_MARCA_TOKENS FLOAT,
            SCORE_MEDIDA_NUM FLOAT,
            SCORE_FINAL FLOAT,
            RANK_POSICAO NUMBER,
            LABEL NUMBER,
            APP_VERSION VARCHAR,
            KNN_K NUMBER,
            WEIGHTS_SNAPSHOT VARIANT
        )
    """,
}


def garantir_tabelas(conn) -> None:
    """DDL idempotente — cria as 4 tabelas se ainda não existirem."""
    with conn.cursor() as cur:
        for nome, ddl in _DDL.items():
            cur.execute(ddl)
            logger.info("Tabela %s pronta", nome)


# ---------------------- Leitura ----------------------

def _fetch_df(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetch_pandas_all()


def ler_insumos_brutos(conn) -> pd.DataFrame:
    """Carrega TBL_INSUMOS completa (fonte para padronização).

    Uso preferencial: ``ler_insumos_brutos_novos`` para fast-path incremental.
    Esta função permanece para casos especiais (reprocessamento full).
    """
    return _fetch_df(conn, f"SELECT * FROM {TBL_INSUMOS}")


def ler_insumos_brutos_novos(conn) -> pd.DataFrame:
    """Server-side diff: retorna apenas CD_INSUMO ainda não codificados.

    Usa LEFT JOIN com TBL_INSUMOS_EMBEDDINGS. Quando tudo já está
    sincronizado retorna DataFrame **vazio sem transferir dados** — base
    do fast-path da sincronização incremental.
    """
    sql = f"""
        SELECT i.*
        FROM {TBL_INSUMOS} i
        LEFT JOIN {TBL_INSUMOS_EMBEDDINGS} e ON i.CD_INSUMO = e.CD_INSUMO
        WHERE e.CD_INSUMO IS NULL
    """
    return _fetch_df(conn, sql)


def contar_totais(conn) -> tuple[int, int]:
    """Retorna (n_em_TBL_INSUMOS, n_em_TBL_INSUMOS_EMBEDDINGS) em uma única ida."""
    sql = f"""
        SELECT
            (SELECT COUNT(*) FROM {TBL_INSUMOS}) AS n_brutos,
            (SELECT COUNT(*) FROM {TBL_INSUMOS_EMBEDDINGS}) AS n_codificados
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        n_brutos, n_codificados = cur.fetchone()
    return int(n_brutos), int(n_codificados)


def ler_medidas_distintas(conn) -> pd.DataFrame:
    """``SELECT DISTINCT CD_MEDIDA, MEDIDA FROM TBL_INSUMOS`` (157 rows ~ instantâneo)."""
    return _fetch_df(conn, f"SELECT DISTINCT CD_MEDIDA, MEDIDA FROM {TBL_INSUMOS}")


def ha_diferenca_medidas(conn) -> bool:
    """True se há diferença entre ``DISTINCT CD_MEDIDA/MEDIDA`` em TBL_INSUMOS
    e o que está em TBL_MEDIDAS_CORRELACAO. Operação barata (157 rows)."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT (SELECT COUNT(DISTINCT CD_MEDIDA||'|'||MEDIDA) FROM {TBL_INSUMOS}) "
            f"- (SELECT COUNT(*) FROM {TBL_MEDIDAS_CORRELACAO})"
        )
        diff = cur.fetchone()[0]
    return int(diff) != 0


def ler_cd_insumos_codificados(conn) -> set[int]:
    """Retorna conjunto de CD_INSUMO já presentes em TBL_INSUMOS_EMBEDDINGS."""
    try:
        df = _fetch_df(conn, f"SELECT CD_INSUMO FROM {TBL_INSUMOS_EMBEDDINGS}")
        if df.empty:
            return set()
        return set(df["CD_INSUMO"].astype(int).tolist())
    except Exception as e:
        logger.warning("Falha ao ler %s (provavelmente vazia): %s", TBL_INSUMOS_EMBEDDINGS, e)
        return set()


def ler_padronizados(conn) -> pd.DataFrame:
    """Carrega TBL_INSUMOS_PADRONIZADOS (usado como df_pad pela app)."""
    df = _fetch_df(conn, f"SELECT * FROM {TBL_INSUMOS_PADRONIZADOS}")
    # Coluna UPDATED_AT é metadado interno — descarta para a app não lidar com timestamps.
    return df.drop(columns=["UPDATED_AT"], errors="ignore")


def ler_embeddings_por_cd(conn, cd_list: list[int]) -> pd.DataFrame:
    """Carrega preprocessados + embeddings para CD_INSUMO específicos."""
    if not cd_list:
        return pd.DataFrame(columns=["GRP_INSUMO", "CD_INSUMO", "INSUMO_DESCRICAO",
                                     "MARCA", "MEDIDA", "bert_vectors"])
    cd_list_str = ",".join(str(int(c)) for c in cd_list)
    sql = f"""
        SELECT
            p.GRP_INSUMO, p.CD_INSUMO, p.INSUMO_DESCRICAO,
            p.MARCA, p.MEDIDA, e.BERT_VECTOR
        FROM {TBL_INSUMOS_PREPROCESSADOS} p
        JOIN {TBL_INSUMOS_EMBEDDINGS} e ON p.CD_INSUMO = e.CD_INSUMO
        WHERE p.CD_INSUMO IN ({cd_list_str})
    """
    df = _fetch_df(conn, sql)
    df = df.rename(columns={"BERT_VECTOR": "bert_vectors"})
    df["bert_vectors"] = df["bert_vectors"].apply(
        lambda v: np.array(json.loads(v) if isinstance(v, str) else v, dtype=np.float32)
    )
    return df


def ler_todos_embeddings(conn) -> pd.DataFrame:
    """Carrega a base completa preprocessados+embeddings (uso: rehidratar cache local)."""
    sql = f"""
        SELECT
            p.GRP_INSUMO, p.CD_INSUMO, p.INSUMO_DESCRICAO,
            p.MARCA, p.MEDIDA, e.BERT_VECTOR
        FROM {TBL_INSUMOS_PREPROCESSADOS} p
        JOIN {TBL_INSUMOS_EMBEDDINGS} e ON p.CD_INSUMO = e.CD_INSUMO
    """
    df = _fetch_df(conn, sql)
    df = df.rename(columns={"BERT_VECTOR": "bert_vectors"})
    df["bert_vectors"] = df["bert_vectors"].apply(
        lambda v: np.array(json.loads(v) if isinstance(v, str) else v, dtype=np.float32)
    )
    return df


# ---------------------- Escrita ----------------------

def _write_pandas_tabela(conn, df: pd.DataFrame, table: str) -> int:
    """Wrapper de write_pandas com normalização de nomes de colunas (uppercase)."""
    if df.empty:
        return 0
    from snowflake.connector.pandas_tools import write_pandas
    df_out = df.copy()
    df_out.columns = [c.upper() for c in df_out.columns]
    success, _, nrows, _ = write_pandas(
        conn=conn,
        df=df_out,
        table_name=table,
        auto_create_table=False,
        overwrite=False,
        quote_identifiers=False,
    )
    if not success:
        raise RuntimeError(f"Falha em write_pandas para {table}")
    return nrows


def insert_padronizados_novos(conn, df: pd.DataFrame) -> int:
    """INSERT em TBL_INSUMOS_PADRONIZADOS (df deve conter apenas registros novos)."""
    cols = ["GRP_INSUMO", "CD_INSUMO", "INSUMO_DESCRICAO", "MARCA", "MEDIDA"]
    if "STATUS" in df.columns:
        cols.append("STATUS")
    return _write_pandas_tabela(conn, df[cols], TBL_INSUMOS_PADRONIZADOS)


def insert_preprocessados_novos(conn, df: pd.DataFrame) -> int:
    """INSERT em TBL_INSUMOS_PREPROCESSADOS (df deve conter apenas registros novos)."""
    cols = ["GRP_INSUMO", "CD_INSUMO", "INSUMO_DESCRICAO", "MARCA", "MEDIDA"]
    return _write_pandas_tabela(conn, df[cols], TBL_INSUMOS_PREPROCESSADOS)


def insert_embeddings_novos(conn, df: pd.DataFrame) -> int:
    """INSERT em TBL_INSUMOS_EMBEDDINGS.

    df precisa ter ``CD_INSUMO`` + ``bert_vectors`` (list ou np.array de floats).
    Faz validação de dimensão antes do INSERT (substitui o que VECTOR daria).
    """
    if df.empty:
        return 0

    invalid_idx = df["bert_vectors"].apply(lambda v: len(v) != EMBEDDING_DIM)
    if bool(invalid_idx.any()):
        bad = df.loc[invalid_idx, "CD_INSUMO"].tolist()
        raise ValueError(
            f"{len(bad)} embeddings com dimensão != {EMBEDDING_DIM}. "
            f"CD_INSUMO afetados (até 5): {bad[:5]}"
        )

    # Stage via temp table com BERT_VECTOR_JSON (VARCHAR), depois PARSE_JSON::ARRAY no INSERT.
    df_stage = pd.DataFrame({
        "CD_INSUMO": df["CD_INSUMO"].astype(int).values,
        "BERT_VECTOR_JSON": [
            json.dumps([float(x) for x in v]) for v in df["bert_vectors"]
        ],
        "MODEL_NAME": [SBERT_MODEL_NAME] * len(df),
        "EMBEDDING_DIM": [EMBEDDING_DIM] * len(df),
    })

    temp_name = f"_TMP_EMB_{int(time.time() * 1000)}"
    from snowflake.connector.pandas_tools import write_pandas
    success, _, _, _ = write_pandas(
        conn=conn,
        df=df_stage,
        table_name=temp_name,
        auto_create_table=True,
        overwrite=True,
        quote_identifiers=False,
    )
    if not success:
        raise RuntimeError(f"Falha em write_pandas para temp {temp_name}")

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {TBL_INSUMOS_EMBEDDINGS} (CD_INSUMO, BERT_VECTOR, MODEL_NAME, EMBEDDING_DIM)
            SELECT CD_INSUMO, PARSE_JSON(BERT_VECTOR_JSON)::ARRAY, MODEL_NAME, EMBEDDING_DIM
            FROM {temp_name}
        """)
        n_inserted = cur.rowcount
        cur.execute(f"DROP TABLE IF EXISTS {temp_name}")
    return int(n_inserted)


def regravar_medida_correlacao(conn, df: pd.DataFrame) -> int:
    """TRUNCATE + INSERT (tabela pequena, substituição total é segura)."""
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE IF EXISTS {TBL_MEDIDAS_CORRELACAO}")
    return _write_pandas_tabela(conn, df[["CD_MEDIDA", "MEDIDA"]], TBL_MEDIDAS_CORRELACAO)


# ---------------------- Feedback de validações ----------------------

def insert_feedback(conn, registros: list[dict]) -> int:
    """Insere registros de feedback em TBL_FEEDBACK_VALIDACOES.

    Cada registro deve estar no formato produzido por
    ``feedback.montar_registro`` (nested dicts: user_input, match, scores,
    weights_snapshot). ``WEIGHTS_SNAPSHOT`` vira VARIANT via PARSE_JSON.
    """
    if not registros:
        return 0

    df_stage = pd.DataFrame([
        {
            "FEEDBACK_ID": r.get("feedback_id"),
            "TIMESTAMP_UTC": r.get("timestamp"),
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
            "WEIGHTS_SNAPSHOT_JSON": json.dumps(r.get("weights_snapshot") or {}),
        }
        for r in registros
    ])

    temp_name = f"_TMP_FB_{int(time.time() * 1000)}"
    from snowflake.connector.pandas_tools import write_pandas
    success, _, _, _ = write_pandas(
        conn=conn,
        df=df_stage,
        table_name=temp_name,
        auto_create_table=True,
        overwrite=True,
        quote_identifiers=False,
    )
    if not success:
        raise RuntimeError(f"Falha em write_pandas para temp {temp_name}")

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {TBL_FEEDBACK_VALIDACOES} (
                FEEDBACK_ID, TIMESTAMP_UTC, SESSION_ID,
                USER_DESCRICAO, USER_MARCA, USER_MEDIDA,
                MATCH_CD_INSUMO, MATCH_GRP_INSUMO, MATCH_DESCRICAO,
                MATCH_MARCA, MATCH_MEDIDA, MATCH_STATUS,
                SCORE_SBERT, SCORE_DESC_TOKENS, SCORE_MARCA_TOKENS,
                SCORE_MEDIDA_NUM, SCORE_FINAL, RANK_POSICAO,
                LABEL, APP_VERSION, KNN_K, WEIGHTS_SNAPSHOT
            )
            SELECT
                FEEDBACK_ID, TIMESTAMP_UTC, SESSION_ID,
                USER_DESCRICAO, USER_MARCA, USER_MEDIDA,
                MATCH_CD_INSUMO, MATCH_GRP_INSUMO, MATCH_DESCRICAO,
                MATCH_MARCA, MATCH_MEDIDA, MATCH_STATUS,
                SCORE_SBERT, SCORE_DESC_TOKENS, SCORE_MARCA_TOKENS,
                SCORE_MEDIDA_NUM, SCORE_FINAL, RANK_POSICAO,
                LABEL, APP_VERSION, KNN_K, PARSE_JSON(WEIGHTS_SNAPSHOT_JSON)
            FROM {temp_name}
        """)
        n = cur.rowcount
        cur.execute(f"DROP TABLE IF EXISTS {temp_name}")
    return int(n)


def ler_feedback(conn) -> pd.DataFrame:
    """Lê TBL_FEEDBACK_VALIDACOES e devolve DataFrame com colunas achatadas
    no mesmo formato esperado pelo restante do app (``user_input.descricao``,
    ``scores.sbert``, etc.)."""
    df = _fetch_df(conn, f"SELECT * FROM {TBL_FEEDBACK_VALIDACOES} ORDER BY TIMESTAMP_UTC")
    if df.empty:
        return df

    df = df.rename(columns={
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
    return df
