"""Penalização v2 — combinação linear de sinais heterogêneos.

Substitui a abordagem anterior (Levenshtein de caractere em descrição/marca/
medida + penalização multiplicativa) por:

- **descrição** e **marca**: similaridade de tokens (rapidfuzz token_set_ratio),
  robusta a ordem de palavras, prefixos e tokens extras.
- **medida**: parser numérico ``(valor, unidade)`` com comparação por erro
  relativo quando unidades batem; fallback para token similarity.
- **combinação linear ponderada** com renormalização de pesos para campos
  ausentes na consulta.

Bugs corrigidos vs versão anterior:

- B1: divisão por zero em strings vazias.
- B2: soma de pesos > 1 levando a score negativo.
- B3: ``"SEM MARCA"`` agora é tratado como ausência de marca.
"""
from __future__ import annotations

import logging
import re

from rapidfuzz.fuzz import token_set_ratio

logger = logging.getLogger(__name__)

# Strings que representam "marca não informada"
VAZIO_MARCAS: set[str] = {"", "sem marca"}


def _is_vazio(s: str | None, *, marcas_vazias: set[str] | None = None) -> bool:
    if s is None:
        return True
    s_norm = s.strip().lower()
    if not s_norm:
        return True
    if marcas_vazias is not None and s_norm in marcas_vazias:
        return True
    return False


_MEDIDA_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Zçãáéíóú\s]*)$")


def parse_medida_canonica(s: str | None) -> tuple[float | None, str]:
    """Decompõe ``"1000grama"`` em ``(1000.0, "grama")``.

    Retorna ``(None, unidade)`` quando não há parte numérica e
    ``(None, "")`` quando a string é vazia/None.
    """
    if not s:
        return None, ""
    s_norm = s.strip().lower()
    m = _MEDIDA_RE.match(s_norm)
    if m:
        val = float(m.group(1).replace(",", "."))
        return val, m.group(2).strip()
    return None, s_norm


def sim_texto_tokens(query: str, doc: str) -> float:
    """Similaridade de tokens em [0, 1]. Vazio em qualquer lado → 0."""
    if not query or not doc:
        return 0.0
    return token_set_ratio(query, doc) / 100.0


def sim_medida(query_med: str, doc_med: str) -> float:
    """Similaridade de medida.

    - Consulta vazia → 1.0 (sem penalidade; caller deve zerar o peso).
    - Doc vazio mas consulta não → 0.0 (mismatch claro).
    - Ambos com valor numérico + unidades compatíveis → ``1 - |Δ|/max``.
    - Unidades diferentes → 0.0.
    - Sem parte numérica → fallback ``token_set_ratio``.
    """
    if _is_vazio(query_med):
        return 1.0
    if _is_vazio(doc_med):
        return 0.0

    q_val, q_unit = parse_medida_canonica(query_med)
    d_val, d_unit = parse_medida_canonica(doc_med)

    units_compatible = (q_unit == d_unit) or (not q_unit) or (not d_unit)

    if q_val is not None and d_val is not None:
        if not units_compatible:
            return 0.0
        denom = max(abs(q_val), abs(d_val), 1.0)
        rel_err = abs(q_val - d_val) / denom
        return max(0.0, 1.0 - rel_err)

    return token_set_ratio(query_med, doc_med) / 100.0


def calcular_similaridades(query: dict, doc: dict) -> dict:
    """Calcula as três similaridades de atributo da consulta vs documento.

    Ambos são dicts com chaves ``descricao``, ``marca``, ``medida`` (strings
    já pré-processadas — lowercase, sem stopwords).
    """
    return {
        "desc": sim_texto_tokens(query.get("descricao", ""), doc.get("descricao", "")),
        "marca": sim_texto_tokens(
            "" if _is_vazio(query.get("marca"), marcas_vazias=VAZIO_MARCAS) else query["marca"],
            doc.get("marca", ""),
        ),
        "medida": sim_medida(query.get("medida", ""), doc.get("medida", "")),
    }


def _ajusta_pesos(weights: dict, query: dict) -> dict:
    """Zera pesos de campos ausentes na consulta e renormaliza."""
    w = dict(weights)
    if _is_vazio(query.get("marca"), marcas_vazias=VAZIO_MARCAS):
        w["marca"] = 0.0
    if _is_vazio(query.get("medida")):
        w["medida"] = 0.0
    if _is_vazio(query.get("descricao")):
        w["desc"] = 0.0

    total = sum(w.values())
    if total <= 0:
        # fallback — todos zerados: cai num caso degenerado, mantém originais
        return dict(weights)
    return {k: v / total for k, v in w.items()}


def combinar_score(
    sbert_sim: float,
    sims: dict,
    query: dict,
    weights: dict,
) -> tuple[float, dict]:
    """Combinação linear ponderada. Retorna (score em [0,1], pesos efetivos)."""
    w = _ajusta_pesos(weights, query)
    score = (
        w["sbert"] * sbert_sim
        + w["desc"] * sims["desc"]
        + w["marca"] * sims["marca"]
        + w["medida"] * sims["medida"]
    )
    return max(0.0, min(1.0, score)), w
