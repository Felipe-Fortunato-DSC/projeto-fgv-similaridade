"""Wrappers cacheados para Streamlit (modelo SBERT, metadados e feedback).

A base de busca vive no S3 Vectors; aqui carregamos apenas os METADADOS
(padronizados + preprocessados + medidas) do Athena para hidratar candidatos
e aplicar a penalização. Sem parquet local, sem KNN em memória.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(ttl=60, show_spinner=False)
def total_codificados() -> int:
    """Nº de itens já codificados (manifesto no Athena). Cacheado (ttl 60s)."""
    from src import aws_io
    try:
        return aws_io.contar_codificados()
    except Exception:
        return 0


def base_existe() -> bool:
    return total_codificados() > 0


@st.cache_resource(show_spinner="Carregando modelo SBERT (primeira execução pode levar 1–2 min)...")
def carregar_modulos():
    """Importa e cacheia módulos pesados (SBERT é carregado em src.similarity)."""
    from src import similarity, knowledge_base
    return similarity, knowledge_base


@st.cache_resource(show_spinner="Carregando base de metadados do Athena...")
def carregar_base():
    """Carrega e cacheia (df_embeddings, df_pad, df_medida) do Athena.

    df_embeddings e df_pad ficam indexados por CD_INSUMO para lookup O(1) dos
    candidatos retornados pelo S3 Vectors.
    """
    from src import aws_io
    df_pad = aws_io.ler_padronizados()
    df_embeddings = aws_io.ler_preprocessados()
    df_medida = aws_io.ler_medidas()

    df_pad = df_pad.set_index("CD_INSUMO", drop=False)
    df_embeddings = df_embeddings.set_index("CD_INSUMO", drop=False)
    return df_embeddings, df_pad, df_medida


def invalidar_cache_base() -> None:
    carregar_base.clear()
    total_codificados.clear()


# ---------------------- Feedback (cacheado para evitar query a cada rerun) ----------------------

@st.cache_data(ttl=30, show_spinner=False)
def estatisticas_feedback_cached() -> dict:
    from src import feedback
    return feedback.estatisticas_feedback()


@st.cache_data(ttl=30, show_spinner=False)
def estatisticas_detalhadas_cached() -> dict:
    from src import feedback
    return feedback.estatisticas_detalhadas()


@st.cache_data(ttl=30, show_spinner="Carregando feedback do Athena...")
def carregar_feedback_cached() -> pd.DataFrame:
    from src import feedback
    return feedback.carregar_feedback()


def invalidar_cache_feedback() -> None:
    """Chamar após salvar feedback novo para atualizar dashboards no próximo render."""
    estatisticas_feedback_cached.clear()
    estatisticas_detalhadas_cached.clear()
    carregar_feedback_cached.clear()
