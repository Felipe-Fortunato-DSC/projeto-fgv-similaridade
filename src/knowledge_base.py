"""Sincronização incremental da base de embeddings.

Identifica novos CD_INSUMO presentes em ``data/input/consulta_bp.csv`` que
ainda não foram codificados em ``data/staging/embeddings_bp.parquet`` e
processa apenas esses novos itens (padronização + embedding), anexando-os
ao parquet existente. Na primeira execução, gera embeddings para toda a
base.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import (
    CONSULTA_BP_CSV,
    MEDIDA_CORRELACAO_CSV,
    DF_PAD_CSV,
    DF_EMBEDDINGS_CSV,
    EMBEDDINGS_PARQUET,
)
from .data_process import (
    padronizar_medida,
    preprocess_text,
    remove_stopwords,
    remover_palavras_duplicadas,
)


def _padronizar_brutos(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica a padronização do notebook 0 sobre o df bruto."""
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


def _salvar_medida_correlacao(df_raw: pd.DataFrame) -> None:
    df_medida = df_raw[['CD_MEDIDA', 'MEDIDA']].drop_duplicates()
    df_medida.to_csv(MEDIDA_CORRELACAO_CSV, index=False)


def carregar_parquet_existente() -> pd.DataFrame | None:
    if EMBEDDINGS_PARQUET.exists():
        return pd.read_parquet(EMBEDDINGS_PARQUET)
    return None


def sincronizar_base(progress_cb=None) -> dict:
    """Sincronização incremental da base de embeddings.

    - Lê ``data/input/consulta_bp.csv``.
    - Compara CD_INSUMO com os já presentes em ``embeddings_bp.parquet``.
    - Para CD_INSUMO novos: padroniza + gera embeddings SBERT.
    - Anexa ao parquet; reescreve ``df_pad.csv`` e ``df_embeddings.csv``.

    ``progress_cb(atual, total)`` é chamado por item quando fornecido.
    Retorna ``{'novos': N, 'total': M, 'primeira_carga': bool}``.
    """
    from .similarity import sbert_model

    if not CONSULTA_BP_CSV.exists():
        raise FileNotFoundError(
            f'Arquivo de entrada não encontrado: {CONSULTA_BP_CSV}. '
            'Coloque o consulta_bp.csv em data/input/.'
        )

    df_raw = pd.read_csv(CONSULTA_BP_CSV, engine='python')
    _salvar_medida_correlacao(df_raw)

    df_emb_full, df_pad_full = _padronizar_brutos(df_raw)
    df_pad_full.to_csv(DF_PAD_CSV)

    df_existente = carregar_parquet_existente()
    primeira_carga = df_existente is None

    if primeira_carga:
        df_novos = df_emb_full.copy()
    else:
        ja_codificados = set(df_existente['CD_INSUMO'].tolist())
        df_novos = df_emb_full[~df_emb_full['CD_INSUMO'].isin(ja_codificados)].copy()

    n_novos = len(df_novos)
    if n_novos == 0:
        return {
            'novos': 0,
            'total': 0 if df_existente is None else len(df_existente),
            'primeira_carga': False,
        }

    textos = df_novos[['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA']].astype(str).agg(' '.join, axis=1).tolist()
    vetores: list[np.ndarray] = []
    if progress_cb is None:
        for texto in tqdm(textos, desc='Embeddings (novos)', total=n_novos):
            vetores.append(sbert_model.encode(texto, convert_to_numpy=True))
    else:
        for i, texto in enumerate(textos):
            vetores.append(sbert_model.encode(texto, convert_to_numpy=True))
            progress_cb(i + 1, n_novos)
    df_novos['bert_vectors'] = vetores

    if df_existente is not None:
        df_final = pd.concat([df_existente, df_novos], ignore_index=True)
    else:
        df_final = df_novos.reset_index(drop=True)

    df_final.to_parquet(EMBEDDINGS_PARQUET)
    df_final.drop(columns=['bert_vectors']).to_csv(DF_EMBEDDINGS_CSV)

    return {
        'novos': n_novos,
        'total': len(df_final),
        'primeira_carga': primeira_carga,
    }
