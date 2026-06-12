"""Pipeline de embeddings SBERT + KNN + consulta com penalização v2."""
from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

from .config import SBERT_MODEL_NAME, DEFAULT_WEIGHTS, DEFAULT_THRESHOLD
from .data_process import padronizar_medida, preprocess_text, remove_stopwords, converter_medida
from . import penalty

logger = logging.getLogger(__name__)

# Modelo SBERT multilíngue (PT-BR) carregado uma vez por processo.
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)


def gerar_embeddings(df, colunas):
    df['bert_vectors'] = [
        sbert_model.encode(row, convert_to_numpy=True)
        for row in tqdm(
            df[colunas].astype(str).agg(' '.join, axis=1),
            desc="Processando embeddings",
        )
    ]
    return df


def treinar_knn(df, metric='cosine', n_neighbors=5):
    X_train = np.array(df['bert_vectors'].tolist())
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric=metric)
    knn_model.fit(X_train)
    return knn_model


def _processar_query(descricao: str, marca: str, medida: str) -> dict:
    """Aplica a mesma pipeline de padronização que df_embeddings.MEDIDA/MARCA/DESC."""
    desc_p = remove_stopwords(preprocess_text(descricao))
    marca_p = remove_stopwords(preprocess_text(marca or ""))
    if medida and medida.strip():
        med_canon = padronizar_medida(converter_medida(medida))
        med_p = remove_stopwords(preprocess_text(str(med_canon).replace(".0", "")))
    else:
        med_p = ""
    return {"descricao": desc_p, "marca": marca_p, "medida": med_p}


def consultar_item(
    descricao: str,
    medida: str,
    knn_model,
    df_pad,
    df_embeddings,
    marca: str = "SEM MARCA",
    *,
    weights: dict | None = None,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Busca os ``k`` vizinhos mais próximos via SBERT+KNN e reordena pela
    combinação linear ponderada definida em ``penalty``.

    Retorna um DataFrame com colunas:
      CD_INSUMO, GRP_INSUMO, INSUMO_DESCRICAO, MARCA, MEDIDA, STATUS (se houver),
      CONSULTA, SCORE_SBERT, SIM_DESC, SIM_MARCA, SIM_MEDIDA, SCORE, RANK.
    """
    weights = weights or DEFAULT_WEIGHTS
    query_proc = _processar_query(descricao, marca, medida)
    query_text = f"{query_proc['descricao']} {query_proc['marca']} {query_proc['medida']}".strip()
    logger.info("Consulta processada: %r", query_text)

    consulta_vector = sbert_model.encode(query_text, convert_to_numpy=True).reshape(1, -1)
    distances, indices = knn_model.kneighbors(consulta_vector)
    idx_flat = indices.flatten()

    # Cópia para display (formato legível, preserva STATUS).
    resultados = df_pad.iloc[idx_flat].copy()
    resultados = resultados.loc[:, ~resultados.columns.str.contains('^Unnamed')]
    resultados['CONSULTA'] = descricao

    # Cosseno → similaridade [0, 1] (KNN devolve 1 - cos_sim como distância).
    sbert_sims = np.clip(1.0 - distances.flatten(), 0.0, 1.0)
    resultados['SCORE_SBERT'] = sbert_sims

    # Versões pré-processadas dos candidatos (usadas para cálculo de sims).
    df_proc = df_embeddings.iloc[idx_flat][['INSUMO_DESCRICAO', 'MARCA', 'MEDIDA']]

    sims_desc, sims_marca, sims_medida, scores_final = [], [], [], []
    for (_, doc_row), sbert_sim in zip(df_proc.iterrows(), sbert_sims):
        doc = {
            "descricao": str(doc_row['INSUMO_DESCRICAO']),
            "marca": str(doc_row['MARCA']),
            "medida": str(doc_row['MEDIDA']),
        }
        sims = penalty.calcular_similaridades(query_proc, doc)
        final, _w_used = penalty.combinar_score(sbert_sim, sims, query_proc, weights)
        sims_desc.append(sims['desc'])
        sims_marca.append(sims['marca'])
        sims_medida.append(sims['medida'])
        scores_final.append(final)

    resultados['SIM_DESC'] = sims_desc
    resultados['SIM_MARCA'] = sims_marca
    resultados['SIM_MEDIDA'] = sims_medida
    resultados['SCORE'] = [v * 100 for v in scores_final]

    # Ordenação e filtro acontecem UMA vez, fora de qualquer loop.
    resultados = resultados.sort_values('SCORE', ascending=False)
    resultados = resultados[resultados['SCORE'] >= float(threshold)]
    resultados = resultados.reset_index(drop=True)
    resultados['RANK'] = range(1, len(resultados) + 1)

    return resultados
