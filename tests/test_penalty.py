"""Testes unitários para src.penalty (v2)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import penalty
from src.config import DEFAULT_WEIGHTS


# ---- parse_medida_canonica ----

def test_parse_valor_unidade_simples():
    assert penalty.parse_medida_canonica("1000grama") == (1000.0, "grama")


def test_parse_unidade_apenas():
    val, unit = penalty.parse_medida_canonica("pacote")
    assert val is None
    assert unit == "pacote"


def test_parse_vazio():
    assert penalty.parse_medida_canonica("") == (None, "")
    assert penalty.parse_medida_canonica(None) == (None, "")


def test_parse_aceita_virgula_decimal():
    val, unit = penalty.parse_medida_canonica("2,5kg")
    assert val == 2.5
    assert unit == "kg"


# ---- sim_medida ----

def test_sim_medida_unidades_iguais_valores_iguais():
    assert penalty.sim_medida("85grama", "85grama") == 1.0


def test_sim_medida_unidades_diferentes_zero():
    assert penalty.sim_medida("1000grama", "1000mililitro") == 0.0


def test_sim_medida_erro_relativo():
    # 85g vs 100g → erro relativo 15/100 = 0.15 → sim ~0.85
    sim = penalty.sim_medida("85grama", "100grama")
    assert math.isclose(sim, 0.85, abs_tol=1e-6)


def test_sim_medida_consulta_vazia_retorna_um():
    # Consulta vazia → não penaliza (caller deve zerar peso, mas valor neutro = 1)
    assert penalty.sim_medida("", "1000grama") == 1.0


def test_sim_medida_doc_vazio_mas_consulta_nao_retorna_zero():
    assert penalty.sim_medida("1000grama", "") == 0.0


# ---- sim_texto_tokens ----

def test_sim_texto_tokens_strings_vazias_zero():
    # B1: nenhuma divisão por zero deve ocorrer
    assert penalty.sim_texto_tokens("", "") == 0.0
    assert penalty.sim_texto_tokens("tubo", "") == 0.0
    assert penalty.sim_texto_tokens("", "tubo") == 0.0


def test_sim_texto_tokens_tokens_iguais_em_ordem_diferente():
    # token_set_ratio é invariante a ordem
    s1 = penalty.sim_texto_tokens("tubo pvc", "pvc tubo")
    assert s1 == 1.0


def test_sim_texto_tokens_prefixo_marca():
    # "TIGRE" vs "TIGRE DO BRASIL" deve ser alto (token_set robusto a tokens extras)
    sim = penalty.sim_texto_tokens("tigre", "tigre do brasil")
    assert sim > 0.7  # com Levenshtein de char ficaria ~0.33


# ---- combinar_score / _ajusta_pesos ----

def test_combinar_score_dentro_do_intervalo():
    """B2: garante que score nunca sai de [0, 1]."""
    sims = {"desc": 0.0, "marca": 0.0, "medida": 0.0}
    query = {"descricao": "x", "marca": "y", "medida": "z"}
    score, _ = penalty.combinar_score(0.0, sims, query, DEFAULT_WEIGHTS)
    assert 0.0 <= score <= 1.0

    score, _ = penalty.combinar_score(1.0, {"desc": 1.0, "marca": 1.0, "medida": 1.0}, query, DEFAULT_WEIGHTS)
    assert score == 1.0


def test_combinar_score_redistribui_pesos_marca_vazia():
    """B3: marca vazia/SEM MARCA → peso de marca redistribuído."""
    sims = {"desc": 1.0, "marca": 0.0, "medida": 1.0}
    query = {"descricao": "x", "marca": "SEM MARCA", "medida": "1g"}
    score, pesos = penalty.combinar_score(1.0, sims, query, DEFAULT_WEIGHTS)
    assert pesos["marca"] == 0.0
    # Soma dos pesos sempre normalizada
    assert math.isclose(sum(pesos.values()), 1.0, abs_tol=1e-6)
    # marca não influencia o score
    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_combinar_score_marca_string_vazia_tratada_como_ausente():
    """B3: marca = '' deve ser tratada igual a 'SEM MARCA'."""
    sims = {"desc": 0.5, "marca": 0.0, "medida": 0.5}
    query_vazia = {"descricao": "x", "marca": "", "medida": "1g"}
    query_sem_marca = {"descricao": "x", "marca": "SEM MARCA", "medida": "1g"}

    score_vazia, _ = penalty.combinar_score(0.5, sims, query_vazia, DEFAULT_WEIGHTS)
    score_sem_marca, _ = penalty.combinar_score(0.5, sims, query_sem_marca, DEFAULT_WEIGHTS)
    assert math.isclose(score_vazia, score_sem_marca, abs_tol=1e-9)


def test_combinar_score_todos_campos_vazios_fallback():
    """Caso degenerado: query totalmente vazia — não deve dividir por zero."""
    sims = {"desc": 0.0, "marca": 0.0, "medida": 0.0}
    query = {"descricao": "", "marca": "", "medida": ""}
    # SBERT sim ainda existe → score baseado só nele com pesos originais
    score, pesos = penalty.combinar_score(0.8, sims, query, DEFAULT_WEIGHTS)
    assert 0.0 <= score <= 1.0


def test_calcular_similaridades_retorna_estrutura_esperada():
    query = {"descricao": "tubo pvc", "marca": "tigre", "medida": "75mm"}
    doc = {"descricao": "tubo pvc esgoto", "marca": "tigre", "medida": "75mm"}
    sims = penalty.calcular_similaridades(query, doc)
    assert set(sims.keys()) == {"desc", "marca", "medida"}
    assert all(0.0 <= v <= 1.0 for v in sims.values())
    assert sims["marca"] == 1.0
    assert sims["medida"] == 1.0
