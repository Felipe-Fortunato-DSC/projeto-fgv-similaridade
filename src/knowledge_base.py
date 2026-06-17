"""Sincronização incremental: tbl_insumos (Athena) → S3 Vectors + tabelas derivadas.

AWS é a fonte de verdade. Não há mais cache local de embeddings: a busca roda
direto no S3 Vectors e os metadados são lidos do Athena.

Fluxo:

1. Conta totais (Athena): tbl_insumos vs manifesto de embeddings.
2. **Diff server-side**: cd_insumo em tbl_insumos sem embedding ainda.
3. Se nada novo → fast path (retorna na hora).
4. Se há novos: padroniza só esses, gera embeddings em lote, grava no
   S3 Vectors (+ manifesto no Athena) e nas tabelas padronizados/preprocessados.
"""
from __future__ import annotations

import logging

import pandas as pd

from .data_process import (
    padronizar_medida,
    preprocess_text,
    remove_stopwords,
    remover_palavras_duplicadas,
)
from . import aws_io
from . import similarity

logger = logging.getLogger(__name__)

_BATCH_SBERT = 64


# ---------------------- Padronização (mesma lógica do notebook 0) ----------------------

def _padronizar_brutos(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica a padronização do notebook 0. Retorna (df_emb_pre, df_pad)."""
    df = df_raw.copy()
    df['DESCRICAO'] = df['DESCRICAO'].fillna('')
    df['MARCA'] = df['MARCA'].fillna('SEM MARCA')
    df['EMBALAGEM'] = df['EMBALAGEM'].fillna('')
    df['QTD_MEDIDA'] = df['QTD_MEDIDA'].fillna('')

    df['GRP_INSUMO'] = df['GRP_INSUMO'].astype(str).str.replace('.0', '')
    df['MEDIDA_PAD'] = (
        df['QTD_MEDIDA'].astype(str).str.replace('.0', '') + df['MEDIDA'].astype(str)
    )
    df['INSUMO_DESCRICAO'] = (
        df['INSUMO'] + ' ' + df['DESCRICAO'].astype(str)
        + ' (' + df['EMBALAGEM'].astype(str) + ')'
    )
    df['MEDIDA_ABV'] = (
        df['QTD_MEDIDA'].astype(str).str.replace('.0', '')
        + ' ' + df['CD_MEDIDA'].astype(str)
    )

    df_emb = df[['GRP_INSUMO', 'CD_INSUMO', 'INSUMO_DESCRICAO', 'MARCA', 'MEDIDA_PAD']].copy()
    df_emb = remover_palavras_duplicadas(df_emb, ['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA_PAD'])
    df_emb = df_emb.rename(columns={'MEDIDA_PAD': 'MEDIDA'})

    pad_cols = ['GRP_INSUMO', 'CD_INSUMO', 'INSUMO_DESCRICAO', 'MARCA', 'MEDIDA_ABV']
    if 'STATUS' in df.columns:
        pad_cols.append('STATUS')
    df_pad = df[pad_cols].copy()
    df_pad = remover_palavras_duplicadas(df_pad, ['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA_ABV'])
    df_pad = df_pad.rename(columns={'MEDIDA_ABV': 'MEDIDA'})

    df_emb['MEDIDA'] = df_emb['MEDIDA'].apply(padronizar_medida).astype(str).str.replace('.0', '')
    for col in ['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA']:
        df_emb[col] = df_emb[col].apply(preprocess_text).apply(remove_stopwords)

    return df_emb, df_pad


# ---------------------- Encoding em lote ----------------------

def _encodar_em_lote(textos: list[str], progress_cb=None) -> list:
    n_total = len(textos)
    vetores: list = []
    for i in range(0, n_total, _BATCH_SBERT):
        batch = textos[i:i + _BATCH_SBERT]
        vetores.extend(similarity.encode_textos(batch, batch_size=_BATCH_SBERT))
        if progress_cb is not None:
            progress_cb(min(i + _BATCH_SBERT, n_total), n_total)
    return vetores


# ---------------------- Sincronização ----------------------

def sincronizar_base(progress_cb=None) -> dict:
    """Sincronização incremental. Retorna ``{'novos', 'total', 'primeira_carga'}``."""
    n_brutos, n_codificados = aws_io.contar_totais()
    primeira_carga = n_codificados == 0
    logger.info("Athena: %s brutos, %s codificados", n_brutos, n_codificados)

    df_raw_novos = aws_io.ler_insumos_brutos_novos()
    logger.info("Linhas novas em tbl_insumos: %s", len(df_raw_novos))

    if df_raw_novos.empty:
        return {"novos": 0, "total": n_codificados, "primeira_carga": False}

    # Padroniza só os novos.
    df_emb_novos, df_pad_novos = _padronizar_brutos(df_raw_novos)
    logger.info("Padronização aplicada em %s registros", len(df_emb_novos))

    # Embeddings em lote.
    textos = (
        df_emb_novos[['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA']]
        .astype(str).agg(' '.join, axis=1).tolist()
    )
    df_emb_novos['bert_vectors'] = _encodar_em_lote(textos, progress_cb=progress_cb)

    # Grava vetores no S3 Vectors (+ manifesto no Athena).
    aws_io.put_vectors(df_emb_novos[['CD_INSUMO', 'bert_vectors']])

    # Tabelas derivadas no Athena.
    aws_io.insert_padronizados_novos(df_pad_novos)
    aws_io.insert_preprocessados_novos(df_emb_novos.drop(columns=['bert_vectors']))

    # Medidas — só regrava se houve mudança real.
    if aws_io.ha_diferenca_medidas():
        aws_io.regravar_medidas(aws_io.ler_medidas_distintas())

    return {
        "novos": len(df_emb_novos),
        "total": n_codificados + len(df_emb_novos),
        "primeira_carga": primeira_carga,
    }
