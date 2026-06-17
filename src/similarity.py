"""Busca por similaridade — SBERT (encode) + S3 Vectors (k-NN) + penalização v2.

A busca k-NN deixou de rodar em memória (sklearn). Agora:

1. A query é codificada pelo SBERT (no processo).
2. O S3 Vectors retorna o top-K ``cd_insumo`` + distância cosseno.
3. Os candidatos são hidratados a partir de DataFrames em memória
   (``df_pad`` para display, ``df_embeddings`` preprocessado para a penalização)
   e reordenados pela combinação linear ponderada de ``penalty``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_THRESHOLD, DEFAULT_TOP_K, DEFAULT_WEIGHTS, SBERT_MODEL_NAME
from .data_process import converter_medida, padronizar_medida, preprocess_text, remove_stopwords
from . import aws_io, penalty

logger = logging.getLogger(__name__)

# Modelo SBERT multilíngue (PT-BR) carregado uma vez por processo.
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)


def encode_textos(textos: list[str], batch_size: int = 64) -> list[np.ndarray]:
    """Encode em lote (usado pelo seed e pela sincronização incremental)."""
    return list(
        sbert_model.encode(
            textos, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
    )


def _processar_query(descricao: str, marca: str, medida: str, df_medida: pd.DataFrame) -> dict:
    """Aplica a mesma pipeline de padronização que df_embeddings.MEDIDA/MARCA/DESC."""
    desc_p = remove_stopwords(preprocess_text(descricao))
    marca_p = remove_stopwords(preprocess_text(marca or ""))
    if medida and medida.strip():
        med_canon = padronizar_medida(converter_medida(medida, df_medida))
        med_p = remove_stopwords(preprocess_text(str(med_canon).replace(".0", "")))
    else:
        med_p = ""
    return {"descricao": desc_p, "marca": marca_p, "medida": med_p}


def consultar_item(
    descricao: str,
    medida: str,
    df_pad: pd.DataFrame,
    df_embeddings: pd.DataFrame,
    df_medida: pd.DataFrame,
    marca: str = "SEM MARCA",
    *,
    weights: dict | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
):
    """Busca top-K vizinhos no S3 Vectors e reordena pela penalização v2.

    ``df_pad``/``df_embeddings`` devem estar indexados por CD_INSUMO (ver
    ``streamlit_app.services``). Retorna DataFrame com colunas:
      CD_INSUMO, GRP_INSUMO, INSUMO_DESCRICAO, MARCA, MEDIDA, STATUS (se houver),
      CONSULTA, SCORE_SBERT, SIM_DESC, SIM_MARCA, SIM_MEDIDA, SCORE, RANK.
    """
    weights = weights or DEFAULT_WEIGHTS
    query_proc = _processar_query(descricao, marca, medida, df_medida)
    query_text = f"{query_proc['descricao']} {query_proc['marca']} {query_proc['medida']}".strip()
    logger.info("Consulta processada: %r", query_text)

    emb = sbert_model.encode(query_text, convert_to_numpy=True)
    vizinhos = aws_io.query_vectors(emb, top_k=top_k)  # [(cd_insumo, distancia), ...]

    cols_saida = ["CD_INSUMO", "GRP_INSUMO", "INSUMO_DESCRICAO", "MARCA", "MEDIDA"]
    if "STATUS" in df_pad.columns:
        cols_saida.append("STATUS")
    if not vizinhos:
        return pd.DataFrame(columns=cols_saida + [
            "CONSULTA", "SCORE_SBERT", "SIM_DESC", "SIM_MARCA", "SIM_MEDIDA", "SCORE", "RANK",
        ])

    linhas = []
    for cd, dist in vizinhos:
        if cd not in df_pad.index or cd not in df_embeddings.index:
            continue
        disp = df_pad.loc[cd]
        proc = df_embeddings.loc[cd]
        sbert_sim = float(np.clip(1.0 - dist, 0.0, 1.0))
        doc = {
            "descricao": str(proc["INSUMO_DESCRICAO"]),
            "marca": str(proc["MARCA"]),
            "medida": str(proc["MEDIDA"]),
        }
        sims = penalty.calcular_similaridades(query_proc, doc)
        final, _w = penalty.combinar_score(sbert_sim, sims, query_proc, weights)
        linha = {
            "CD_INSUMO": cd,
            "GRP_INSUMO": disp["GRP_INSUMO"],
            "INSUMO_DESCRICAO": disp["INSUMO_DESCRICAO"],
            "MARCA": disp["MARCA"],
            "MEDIDA": disp["MEDIDA"],
            "CONSULTA": descricao,
            "SCORE_SBERT": sbert_sim,
            "SIM_DESC": sims["desc"],
            "SIM_MARCA": sims["marca"],
            "SIM_MEDIDA": sims["medida"],
            "SCORE": final * 100,
        }
        if "STATUS" in df_pad.columns:
            linha["STATUS"] = disp["STATUS"]
        linhas.append(linha)

    resultados = pd.DataFrame(linhas)
    if resultados.empty:
        return resultados
    resultados = resultados.sort_values("SCORE", ascending=False)
    resultados = resultados[resultados["SCORE"] >= float(threshold)]
    resultados = resultados.reset_index(drop=True)
    resultados["RANK"] = range(1, len(resultados) + 1)
    return resultados
