"""Frontend Streamlit para consulta de insumos por similaridade.

Executar: ``streamlit run app.py`` a partir da raiz do projeto.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from src.config import EMBEDDINGS_PARQUET
from streamlit_app.services import (
    base_existe,
    carregar_indice,
    carregar_modulos,
    invalidar_cache_indice,
    parquet_mtime,
    total_codificados,
)


st.set_page_config(
    page_title="Consulta por Similaridade — FGV",
    page_icon="🔎",
    layout="wide",
)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Base de conhecimento")

    if base_existe():
        st.metric("Itens codificados", f"{total_codificados():,}".replace(",", "."))
    else:
        st.warning("Nenhum embedding gerado ainda. Sincronize a base antes da consulta.")

    if st.button("🔄 Sincronizar base", use_container_width=True):
        _, knowledge_base = carregar_modulos()
        progress = st.progress(0.0, text="Iniciando sincronização...")

        def cb(atual: int, total: int) -> None:
            progress.progress(atual / total, text=f"Gerando embedding {atual}/{total}")

        try:
            stats = knowledge_base.sincronizar_base(progress_cb=cb)
        except FileNotFoundError as e:
            progress.empty()
            st.error(str(e))
            stats = None

        if stats is not None:
            progress.empty()
            if stats['novos'] == 0:
                st.info(f"Base já atualizada ({stats['total']} itens).")
            else:
                tipo = "Primeira carga" if stats['primeira_carga'] else "Atualização incremental"
                st.success(f"{tipo}: {stats['novos']} novos itens. Total: {stats['total']}.")
            invalidar_cache_indice()
            st.rerun()

    st.divider()
    st.header("Configuração")
    n_neighbors = st.slider("Vizinhos (k)", min_value=5, max_value=50, value=30)


# ---------------- Conteúdo principal ----------------
st.title("🔎 Consulta por Similaridade")
st.caption(
    "Busca de insumos semelhantes via embeddings SBERT + KNN com penalização Levenshtein."
)

tab_consulta, tab_dados = st.tabs(["Consulta", "Dados"])

with tab_consulta:
    if not base_existe():
        st.info(
            "Sincronize a base de conhecimento no menu lateral antes da primeira consulta. "
            "Na primeira execução, o processo pode levar vários minutos para gerar embeddings "
            "de toda a base. Acessos subsequentes processarão apenas itens novos."
        )
    else:
        with st.form("formulario_consulta"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                descricao = st.text_input(
                    "Descrição do insumo *",
                    placeholder="Ex: TUBO DE PVC 75MM",
                )
            with col2:
                marca = st.text_input("Marca (opcional)", placeholder="Ex: TIGRE")
            with col3:
                medida = st.text_input("Medida (opcional)", placeholder="Ex: 12M, 1KG, 500ML")
            submitted = st.form_submit_button(
                "Consultar", type="primary", use_container_width=True
            )

        if submitted:
            if not descricao.strip():
                st.error("A descrição é obrigatória.")
            else:
                similarity, _ = carregar_modulos()
                df_embeddings, df_pad, knn = carregar_indice(parquet_mtime(), n_neighbors)
                marca_str = marca.strip() if marca.strip() else "SEM MARCA"
                with st.spinner("Buscando insumos similares..."):
                    resultados = similarity.consultar_item(
                        descricao.strip(),
                        medida.strip(),
                        knn,
                        df_pad,
                        df_embeddings,
                        marca_str,
                    )

                if resultados is None or resultados.empty:
                    st.warning(
                        "Nenhum resultado com score acima de 50% para essa consulta. "
                        "Tente refinar descrição, marca ou medida."
                    )
                else:
                    st.success(f"{len(resultados)} resultado(s) encontrado(s).")
                    st.dataframe(resultados, use_container_width=True, hide_index=True)
                    safe_name = "".join(c if c.isalnum() else "_" for c in descricao.strip())[:50]
                    csv_bytes = resultados.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Baixar resultados (CSV)",
                        data=csv_bytes,
                        file_name=f"resultados_{safe_name}.csv",
                        mime="text/csv",
                    )

with tab_dados:
    if not base_existe():
        st.info("Sem dados para exibir.")
    else:
        st.subheader("Amostra da base de embeddings")
        df_amostra = pd.read_parquet(EMBEDDINGS_PARQUET)
        n_amostra = min(50, len(df_amostra))
        st.dataframe(
            df_amostra.drop(columns=['bert_vectors']).sample(n_amostra),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Exibindo {n_amostra} de {len(df_amostra):,} itens.".replace(",", "."))
