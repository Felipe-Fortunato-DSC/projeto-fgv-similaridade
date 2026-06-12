"""Sincronização incremental Snowflake ↔ cache local — otimizada.

Estratégia "fast-path": no caso comum (base estável, sem novos itens), a
sincronização retorna em poucos segundos sem:
- baixar 505k rows de TBL_INSUMOS,
- carregar 750MB de parquet,
- carregar o modelo SBERT,
- reescrever caches locais.

Snowflake é a fonte de verdade. O parquet local é cache de runtime para o
KNN em memória (sklearn).

Fluxo:

1. Conta totais via SF (1 query barata).
2. **Server-side diff**: ``SELECT ... LEFT JOIN ... WHERE e.CD_INSUMO IS NULL``
   retorna apenas as linhas novas (zero rows na maioria dos casos).
3. Lê só a coluna ``CD_INSUMO`` do parquet local (não os vetores).
4. Decide:
   - **Fast path** — nada novo, parquet em dia: retorna imediatamente.
   - **Rehidratação** — parquet ausente/incompleto: baixa embeddings da SF.
   - **Append** — há novos: padroniza só os novos, gera embeddings em batch,
     INSERT em SF, anexa ao parquet/CSV locais.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    DF_EMBEDDINGS_CSV,
    DF_PAD_CSV,
    EMBEDDINGS_PARQUET,
    MEDIDA_CORRELACAO_CSV,
)
from .data_process import (
    padronizar_medida,
    preprocess_text,
    remove_stopwords,
    remover_palavras_duplicadas,
)
from . import snowflake_io as sf

logger = logging.getLogger(__name__)

_BATCH_SBERT = 64  # tamanho do batch para encode em lote


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


# ---------------------- Helpers de cache local ----------------------

def carregar_parquet_existente() -> pd.DataFrame | None:
    if EMBEDDINGS_PARQUET.exists():
        return pd.read_parquet(EMBEDDINGS_PARQUET)
    return None


def _cd_insumos_no_parquet() -> set[int]:
    """Lê só a coluna CD_INSUMO do parquet — ~5MB em vez de 750MB."""
    if not EMBEDDINGS_PARQUET.exists():
        return set()
    df_ids = pd.read_parquet(EMBEDDINGS_PARQUET, columns=['CD_INSUMO'])
    return set(df_ids['CD_INSUMO'].astype(int).tolist())


def _anexar_ao_parquet(df_novos_com_vetores: pd.DataFrame) -> None:
    """Append-only: concatena com o parquet existente sem reler vetores se possível."""
    if EMBEDDINGS_PARQUET.exists():
        df_existente = pd.read_parquet(EMBEDDINGS_PARQUET)
        df_final = pd.concat([df_existente, df_novos_com_vetores], ignore_index=True)
    else:
        df_final = df_novos_com_vetores.reset_index(drop=True)
    df_final.to_parquet(EMBEDDINGS_PARQUET)


def _anexar_df_pad(df_pad_novos: pd.DataFrame) -> None:
    """Append no CSV local de df_pad."""
    if DF_PAD_CSV.exists():
        df_existente = pd.read_csv(DF_PAD_CSV).drop(columns=['Unnamed: 0'], errors='ignore')
        df_final = pd.concat([df_existente, df_pad_novos], ignore_index=True)
    else:
        df_final = df_pad_novos.reset_index(drop=True)
    df_final.to_csv(DF_PAD_CSV)


# ---------------------- Encoding em batch ----------------------

def _encodar_em_lote(textos: list[str], progress_cb=None) -> list[np.ndarray]:
    """Gera embeddings em batches (mais rápido que encode linha-a-linha)."""
    # Import lazy — só carrega SBERT quando realmente há o que codificar.
    from .similarity import sbert_model

    n_total = len(textos)
    vetores: list[np.ndarray] = []
    for i in range(0, n_total, _BATCH_SBERT):
        batch = textos[i:i + _BATCH_SBERT]
        batch_vecs = sbert_model.encode(
            batch, convert_to_numpy=True, show_progress_bar=False
        )
        vetores.extend(batch_vecs)
        if progress_cb is not None:
            progress_cb(min(i + _BATCH_SBERT, n_total), n_total)
    return vetores


# ---------------------- Sincronização ----------------------

def sincronizar_base(progress_cb=None) -> dict:
    """Sincronização incremental otimizada.

    Retorna ``{'novos': N, 'total': M, 'primeira_carga': bool, 'rehidratados': R}``.
    """
    conn = sf.conectar()
    try:
        sf.garantir_tabelas(conn)

        # 1. Contagens rápidas (1 ida ao SF)
        n_brutos, n_codificados = sf.contar_totais(conn)
        primeira_carga = n_codificados == 0
        logger.info("Snowflake: %s brutos, %s codificados", n_brutos, n_codificados)

        # 2. CDs já no cache local (leitura barata: só 1 coluna)
        cd_local = _cd_insumos_no_parquet()
        logger.info("Cache local: %s CD_INSUMO", len(cd_local))

        # 3. Server-side diff: traz só linhas novas (0 rows no caso comum)
        df_raw_novos = sf.ler_insumos_brutos_novos(conn)
        logger.info("Linhas novas em TBL_INSUMOS: %s", len(df_raw_novos))

        # 4. Rehidratação: SF tem itens que o cache local não tem
        rehidratados = 0
        if n_codificados > len(cd_local):
            faltam_local = n_codificados - len(cd_local)
            logger.info("Rehidratando cache local: %s embeddings faltam", faltam_local)
            if not cd_local:
                # Cache vazio — download completo
                df_baixados = sf.ler_todos_embeddings(conn)
            else:
                # Cache parcial — busca apenas CDs faltantes
                from .snowflake_io import (
                    TBL_INSUMOS_PREPROCESSADOS as TPRE,
                    TBL_INSUMOS_EMBEDDINGS as TEMB,
                )
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT CD_INSUMO FROM {TEMB} "
                        f"WHERE CD_INSUMO NOT IN ({','.join(str(c) for c in cd_local)})"
                    )
                    cds = [r[0] for r in cur.fetchall()]
                df_baixados = sf.ler_embeddings_por_cd(conn, cds)

            df_local_existente = carregar_parquet_existente()
            if df_local_existente is None or df_local_existente.empty:
                df_local_atualizado = df_baixados
            else:
                df_local_atualizado = pd.concat(
                    [df_local_existente, df_baixados], ignore_index=True
                )
            df_local_atualizado.to_parquet(EMBEDDINGS_PARQUET)
            rehidratados = len(df_baixados)
            cd_local = set(df_local_atualizado['CD_INSUMO'].astype(int).tolist())

        # 5. Fast path — nada novo: pula medida_correlacao também.
        # Justificativa: como medidas são derivadas de TBL_INSUMOS e nada novo
        # foi adicionado, o conjunto distinto não muda. Edge case (UPDATE em
        # registro existente alterando CD_MEDIDA) é raro o suficiente para
        # aceitar que seja capturado só na próxima sync com novos itens.
        if df_raw_novos.empty:
            return {
                "novos": 0,
                "total": n_codificados,
                "primeira_carga": False,
                "rehidratados": rehidratados,
            }

        # 6. Há novos — padroniza só esses (não 505k!)
        df_emb_novos, df_pad_novos = _padronizar_brutos(df_raw_novos)
        logger.info("Padronização aplicada em %s registros", len(df_emb_novos))

        # 7. Embeddings em batch (defer SBERT import)
        textos = (
            df_emb_novos[['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA']]
            .astype(str).agg(' '.join, axis=1).tolist()
        )
        vetores = _encodar_em_lote(textos, progress_cb=progress_cb)
        df_emb_novos['bert_vectors'] = vetores

        # 8. INSERT em SF (3 tabelas)
        sf.insert_padronizados_novos(conn, df_pad_novos)
        sf.insert_preprocessados_novos(conn, df_emb_novos.drop(columns=['bert_vectors']))
        sf.insert_embeddings_novos(conn, df_emb_novos[['CD_INSUMO', 'bert_vectors']])

        # 9. Append nos caches locais (sem reescrita full)
        _anexar_ao_parquet(df_emb_novos)
        _anexar_df_pad(df_pad_novos)

        # 10. medida_correlacao — só regrava se houve mudança real
        if sf.ha_diferenca_medidas(conn):
            df_medida = sf.ler_medidas_distintas(conn)
            sf.regravar_medida_correlacao(conn, df_medida)
            df_medida.to_csv(MEDIDA_CORRELACAO_CSV, index=False)

        # 11. df_embeddings.csv (cache complementar)
        if EMBEDDINGS_PARQUET.exists():
            df_full = pd.read_parquet(EMBEDDINGS_PARQUET, columns=[
                'GRP_INSUMO', 'CD_INSUMO', 'INSUMO_DESCRICAO', 'MARCA', 'MEDIDA',
            ])
            df_full.to_csv(DF_EMBEDDINGS_CSV)

        return {
            "novos": len(df_emb_novos),
            "total": n_codificados + len(df_emb_novos),
            "primeira_carga": primeira_carga,
            "rehidratados": rehidratados,
        }
    finally:
        conn.close()
