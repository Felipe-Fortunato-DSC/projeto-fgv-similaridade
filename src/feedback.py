"""Persistência de feedback de validação para fine-tuning futuro.

Armazena cada validação como uma linha JSON em
``data/training/feedback.jsonl``. Formato append-only e schema-evolution
friendly (campos podem ser adicionados sem migrar histórico).

A leitura via ``carregar_feedback`` retorna um DataFrame com colunas
achatadas (``user_input.descricao``, ``scores.sbert``, etc.) para análise.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from .config import APP_VERSION, FEEDBACK_JSONL


def gerar_session_id() -> str:
    return str(uuid.uuid4())


def montar_registro(
    *,
    session_id: str,
    user_input: dict,
    match: dict,
    scores: dict,
    rank: int,
    label: int,
    weights_snapshot: dict,
    knn_k: int,
) -> dict:
    """Monta o registro com timestamp e versão do app preenchidos."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_input": user_input,
        "match": match,
        "scores": scores,
        "rank": rank,
        "label": int(label),
        "app_version": APP_VERSION,
        "knn_k": int(knn_k),
        "weights_snapshot": weights_snapshot,
    }


def salvar_feedback(rows: list[dict]) -> int:
    """Append de registros no JSONL. Retorna a quantidade salva."""
    if not rows:
        return 0
    FEEDBACK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_JSONL.open('a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return len(rows)


def carregar_feedback() -> pd.DataFrame:
    """Lê todos os registros como DataFrame achatado."""
    if not FEEDBACK_JSONL.exists():
        return pd.DataFrame()
    records: list[dict] = []
    with FEEDBACK_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def estatisticas_feedback() -> dict:
    """Conta total, aprovados e reprovados (rápido — itera apenas a coluna label)."""
    df = carregar_feedback()
    if df.empty or 'label' not in df.columns:
        return {"total": 0, "aprovados": 0, "reprovados": 0}
    total = len(df)
    aprovados = int((df['label'] == 1).sum())
    return {
        "total": total,
        "aprovados": aprovados,
        "reprovados": total - aprovados,
    }


# Metas para iniciar fine-tuning (ver docs/ROADMAP_FINETUNING.txt)
META_VALIDACOES = 1000
META_QUERIES_UNICAS = 300
META_MIN_VAL_POR_QUERY = 3


def estatisticas_detalhadas() -> dict:
    """Métricas completas para a aba de feedback + indicador de prontidão."""
    df = carregar_feedback()

    base = {
        "total": 0,
        "aprovados": 0,
        "reprovados": 0,
        "queries_unicas": 0,
        "queries_3mais": 0,
        "balanco_pct": (0.0, 0.0),
        "desbalanceado": False,
        "ranks_dist": {},
        "top_queries": [],
        "pronto_treino": False,
        "faltam_validacoes": META_VALIDACOES,
        "faltam_queries": META_QUERIES_UNICAS,
    }
    if df.empty or 'label' not in df.columns:
        return base

    total = len(df)
    aprovados = int((df['label'] == 1).sum())
    reprovados = total - aprovados
    ap_pct = round(100 * aprovados / total, 1) if total else 0.0
    rp_pct = round(100 - ap_pct, 1)

    # Query key: descrição + marca + medida (uppercase, trimmed)
    def _q_key(row):
        d = str(row.get('user_input.descricao') or '').strip().upper()
        m = str(row.get('user_input.marca') or '').strip().upper()
        med = str(row.get('user_input.medida') or '').strip().upper()
        return f"{d}|{m}|{med}"

    df['_qkey'] = df.apply(_q_key, axis=1)
    q_counts = df['_qkey'].value_counts()
    queries_unicas = int(len(q_counts))
    queries_3mais = int((q_counts >= META_MIN_VAL_POR_QUERY).sum())

    # Top 10 queries (descrição apenas, para readability)
    top10 = []
    for qkey, n in q_counts.head(10).items():
        d, m, med = qkey.split('|')
        partes = [d or "(sem descrição)"]
        if m:
            partes.append(f"marca={m}")
        if med:
            partes.append(f"medida={med}")
        top10.append({"query": " · ".join(partes), "validacoes": int(n)})

    # Distribuição por rank
    ranks_dist = {}
    if 'rank' in df.columns:
        rc = df['rank'].value_counts().sort_index()
        ranks_dist = {int(k): int(v) for k, v in rc.items()}

    desbalanceado = total >= 50 and abs(ap_pct - 50) > 25  # +/-25pp do equilíbrio
    pronto = total >= META_VALIDACOES and queries_unicas >= META_QUERIES_UNICAS

    return {
        "total": total,
        "aprovados": aprovados,
        "reprovados": reprovados,
        "queries_unicas": queries_unicas,
        "queries_3mais": queries_3mais,
        "balanco_pct": (ap_pct, rp_pct),
        "desbalanceado": desbalanceado,
        "ranks_dist": ranks_dist,
        "top_queries": top10,
        "pronto_treino": pronto,
        "faltam_validacoes": max(0, META_VALIDACOES - total),
        "faltam_queries": max(0, META_QUERIES_UNICAS - queries_unicas),
    }
