"""Wrappers cacheados para Streamlit (modelos pesados, embeddings e KNN)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import EMBEDDINGS_PARQUET, DF_PAD_CSV


def parquet_mtime() -> float:
    """Modtime do parquet — usado como chave de invalidação do cache."""
    return EMBEDDINGS_PARQUET.stat().st_mtime if EMBEDDINGS_PARQUET.exists() else 0.0


def base_existe() -> bool:
    return EMBEDDINGS_PARQUET.exists()


def total_codificados() -> int:
    if not EMBEDDINGS_PARQUET.exists():
        return 0
    return len(pd.read_parquet(EMBEDDINGS_PARQUET, columns=['CD_INSUMO']))


@st.cache_resource(show_spinner="Carregando modelo SBERT (primeira execução pode levar 1–2 min)...")
def carregar_modulos():
    """Importa e cacheia módulos pesados (SBERT é carregado em src.similarity)."""
    from src import similarity, knowledge_base
    return similarity, knowledge_base


@st.cache_resource(show_spinner="Indexando base no KNN...")
def carregar_indice(mtime: float, n_neighbors: int):
    """Cacheia (df_embeddings, df_pad, knn). Reconstrói se mtime ou k mudar."""
    similarity, _ = carregar_modulos()
    df_embeddings = pd.read_parquet(EMBEDDINGS_PARQUET)
    df_pad = pd.read_csv(DF_PAD_CSV).drop(columns=['Unnamed: 0'], errors='ignore')
    knn = similarity.treinar_knn(df_embeddings, metric='cosine', n_neighbors=n_neighbors)
    return df_embeddings, df_pad, knn


def invalidar_cache_indice() -> None:
    carregar_indice.clear()


# ---------------------- Feedback (cacheado para evitar SF query a cada rerun) ----------------------

@st.cache_data(ttl=30, show_spinner=False)
def estatisticas_feedback_cached() -> dict:
    from src import feedback
    return feedback.estatisticas_feedback()


@st.cache_data(ttl=30, show_spinner=False)
def estatisticas_detalhadas_cached() -> dict:
    from src import feedback
    return feedback.estatisticas_detalhadas()


@st.cache_data(ttl=30, show_spinner="Carregando feedback do Snowflake...")
def carregar_feedback_cached() -> pd.DataFrame:
    from src import feedback
    return feedback.carregar_feedback()


def invalidar_cache_feedback() -> None:
    """Chamar após salvar feedback novo para atualizar dashboards no próximo render."""
    estatisticas_feedback_cached.clear()
    estatisticas_detalhadas_cached.clear()
    carregar_feedback_cached.clear()
