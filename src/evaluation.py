"""Esqueleto de avaliação contra um gold standard.

Espera CSV em ``data/eval/gold_standard.csv`` com colunas:

    query_descricao, query_marca, query_medida, expected_cd_insumo

Cada linha = uma consulta rotulada (CD_INSUMO esperado no top-1).
Calibração real dos pesos depende deste conjunto estar preenchido.
"""
from __future__ import annotations

import pandas as pd

from .config import EVAL_GOLD_CSV


def carregar_gold_standard() -> pd.DataFrame:
    if not EVAL_GOLD_CSV.exists():
        raise FileNotFoundError(
            f"Gold standard não encontrado: {EVAL_GOLD_CSV}. "
            "Preencha o template em data/eval/gold_standard_template.csv "
            "e renomeie para gold_standard.csv."
        )
    return pd.read_csv(EVAL_GOLD_CSV)


def recall_at_k(rankings: list[list[int]], esperados: list[int], k: int = 5) -> float:
    """Fração de consultas em que o esperado aparece no top-k."""
    if not rankings:
        return 0.0
    hits = sum(1 for r, e in zip(rankings, esperados) if e in r[:k])
    return hits / len(rankings)


def mrr(rankings: list[list[int]], esperados: list[int]) -> float:
    """Mean Reciprocal Rank — métrica clássica de ranking."""
    if not rankings:
        return 0.0
    rrs: list[float] = []
    for r, e in zip(rankings, esperados):
        try:
            pos = r.index(e) + 1
            rrs.append(1.0 / pos)
        except ValueError:
            rrs.append(0.0)
    return sum(rrs) / len(rrs)
