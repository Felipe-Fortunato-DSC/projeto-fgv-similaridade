"""Frontend Streamlit v2 — consulta por similaridade com human-in-the-loop.

Diferenças vs v1:
- Pesos da combinação linear configuráveis na sidebar (com renormalização).
- Threshold configurável (slider).
- Tabela de resultados editável com coluna "Avaliação" (Aprovado/Reprovado).
- Validações persistidas em ``data/training/feedback.jsonl`` para fine-tuning.
- Aba de feedback registrado para auditar e exportar a base de treinamento.

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

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

import pandas as pd
import streamlit as st

from src.config import (
    APP_VERSION,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    FEEDBACK_JSONL,
)
from src import feedback
from streamlit_app.services import (
    base_existe,
    carregar_indice,
    carregar_modulos,
    invalidar_cache_indice,
    parquet_mtime,
    total_codificados,
)


st.set_page_config(
    page_title=f"Consulta por Similaridade — FGV ({APP_VERSION})",
    page_icon="🔎",
    layout="wide",
)


# ---------------- Inicialização de session_state ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = feedback.gerar_session_id()
if "resultados" not in st.session_state:
    st.session_state.resultados = None
if "ultima_query" not in st.session_state:
    st.session_state.ultima_query = None
if "weights_snapshot" not in st.session_state:
    st.session_state.weights_snapshot = dict(DEFAULT_WEIGHTS)
if "evaluated_rows" not in st.session_state:
    # rank (int) -> 1 (aprovado) | 0 (reprovado)
    st.session_state.evaluated_rows = {}
if "knn_k" not in st.session_state:
    st.session_state.knn_k = 30
if "auto_sync_done" not in st.session_state:
    st.session_state.auto_sync_done = False
# NÃO usar setdefault para chaves ligadas a widgets — Streamlit pode ignorar
# o valor no primeiro render do widget. Inicialização é feita via `value=` no
# próprio slider abaixo.


# ---------------- Feedback helpers ----------------
def _save_single_feedback(row_dict: dict, label: int) -> None:
    """Persiste uma única validação no JSONL."""
    registro = feedback.montar_registro(
        session_id=st.session_state.session_id,
        user_input=st.session_state.ultima_query or {},
        match={
            "cd_insumo": int(row_dict["CD_INSUMO"]),
            "grp_insumo": str(row_dict["GRP_INSUMO"]),
            "descricao": str(row_dict["INSUMO_DESCRICAO"]),
            "marca": str(row_dict["MARCA"]),
            "medida": str(row_dict["MEDIDA"]),
            "status": str(row_dict.get("STATUS", "")),
        },
        scores={
            "sbert": float(row_dict["SCORE_SBERT"]),
            "desc_tokens": float(row_dict["SIM_DESC"]),
            "marca_tokens": float(row_dict["SIM_MARCA"]),
            "medida_numeric": float(row_dict["SIM_MEDIDA"]),
            "final": float(row_dict["SCORE"]),
        },
        rank=int(row_dict.get("RANK", 0)),
        label=label,
        weights_snapshot=st.session_state.weights_snapshot,
        knn_k=st.session_state.knn_k,
    )
    feedback.salvar_feedback([registro])


@st.dialog("Confirmar avaliação")
def _dialog_confirmar(row_dict: dict, label: int) -> None:
    if label == 1:
        st.markdown("### ✅ Aprovar este match?")
    else:
        st.markdown("### ❌ Reprovar este match?")

    st.markdown(f"**Item consultado:** {(st.session_state.ultima_query or {}).get('descricao', '')}")
    st.divider()
    st.markdown(f"**Match #{int(row_dict.get('RANK', 0))}** — {row_dict['INSUMO_DESCRICAO']}")
    st.caption(
        f"CD: {row_dict['CD_INSUMO']} · Marca: {row_dict['MARCA']} · "
        f"Medida: {row_dict['MEDIDA']} · Score: {float(row_dict['SCORE']):.2f}"
    )
    st.caption(
        "Esta validação será gravada em `data/training/feedback.jsonl` para "
        "compor a base de fine-tuning."
    )
    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("Confirmar", type="primary", use_container_width=True):
        _save_single_feedback(row_dict, label)
        st.session_state.evaluated_rows[int(row_dict.get("RANK", 0))] = label
        st.rerun()
    if col2.button("Cancelar", use_container_width=True):
        st.rerun()


@st.dialog("Sincronização automática", width="large")
def _dialog_auto_sync() -> None:
    """Modal exibido na abertura da app — sincroniza a base com Snowflake."""
    st.markdown("### 🔄 Atualizando base de insumos")
    st.caption(
        "Verificando se há novos itens no Snowflake "
        "(`BASES_SPDO.DB_GESTAO_BANCO_PRECO_APP_CONSULTA.TBL_INSUMOS`). "
        "A sincronização é **incremental** — apenas itens ausentes do cache "
        "local geram embeddings novos."
    )

    progress = st.progress(0.0, text="Conectando ao Snowflake…")

    def _cb(atual: int, total: int) -> None:
        progress.progress(
            atual / total,
            text=f"Gerando embedding {atual:,} de {total:,}…".replace(",", "."),
        )

    stats = None
    erro: str | None = None
    try:
        _, knowledge_base = carregar_modulos()
        stats = knowledge_base.sincronizar_base(progress_cb=_cb)
        progress.empty()

        # Pre-warm do KNN: treina sobre a base atualizada para que a 1ª consulta
        # seja instantânea. Sem isso, o usuário esperaria 10-20s na 1ª busca.
        if base_existe():
            warm = st.empty()
            warm.info("⚙️ Preparando motor de busca (treinando KNN sobre a base)...")
            carregar_indice(parquet_mtime(), st.session_state.knn_k)
            warm.empty()
    except Exception as e:
        progress.empty()
        erro = str(e)

    if erro:
        st.error(f"Não foi possível sincronizar com o Snowflake.\n\n{erro}")
        st.caption(
            "Você pode continuar usando o cache local — mas ele pode estar desatualizado."
        )
    elif stats:
        if stats["novos"] == 0 and stats.get("rehidratados", 0) == 0:
            total_fmt = f"{stats['total']:,}".replace(",", ".")
            st.success(f"✅ Base atualizada — **{total_fmt}** itens disponíveis.")
        else:
            if stats.get("rehidratados", 0) > 0:
                r_fmt = f"{stats['rehidratados']:,}".replace(",", ".")
                st.info(f"☁️ Cache local rehidratado com **{r_fmt}** embeddings da SF.")
            if stats["novos"] > 0:
                tipo = "Primeira carga" if stats["primeira_carga"] else "Atualização incremental"
                n_fmt = f"{stats['novos']:,}".replace(",", ".")
                st.success(f"🆕 {tipo}: **{n_fmt}** novos itens processados.")
            total_fmt = f"{stats['total']:,}".replace(",", ".")
            st.caption(f"Total na base de consulta: {total_fmt} itens.")

    st.divider()
    if st.button("Continuar →", type="primary", use_container_width=True):
        st.session_state.auto_sync_done = True
        invalidar_cache_indice()
        st.rerun()


# ---------------- Sincronização automática (uma vez por sessão) ----------------
if not st.session_state.auto_sync_done:
    _dialog_auto_sync()
    st.stop()


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Base de conhecimento")

    if base_existe():
        st.metric("Itens codificados", f"{total_codificados():,}".replace(",", "."))
    else:
        st.warning("Sincronize a base antes da consulta.")

    if st.button("🔄 Sincronizar base", use_container_width=True):
        _, knowledge_base = carregar_modulos()
        progress = st.progress(0.0, text="Iniciando sincronização...")

        def _cb(atual: int, total: int) -> None:
            progress.progress(atual / total, text=f"Embedding {atual}/{total}")

        try:
            stats = knowledge_base.sincronizar_base(progress_cb=_cb)
        except FileNotFoundError as e:
            progress.empty()
            st.error(str(e))
            stats = None
        except Exception as e:
            progress.empty()
            st.error(f"Erro ao sincronizar com Snowflake: {e}")
            stats = None

        if stats is not None:
            progress.empty()
            partes = []
            if stats.get('rehidratados', 0) > 0:
                partes.append(f"Cache local rehidratado: {stats['rehidratados']} embeddings da SF.")
            if stats['novos'] == 0:
                partes.append(f"Base já atualizada ({stats['total']} itens).")
                msg = " ".join(partes)
                st.info(msg)
            else:
                tipo = "Primeira carga" if stats['primeira_carga'] else "Atualização incremental"
                partes.append(f"{tipo}: {stats['novos']} novos. Total: {stats['total']}.")
                st.success(" ".join(partes))
            invalidar_cache_indice()
            st.rerun()

    st.divider()
    st.header("Configuração")
    n_neighbors = st.slider("Vizinhos (k)", 5, 50, 30)
    threshold = st.slider("Score mínimo", 0, 100, int(DEFAULT_THRESHOLD))

    with st.expander("⚖️ Pesos do score", expanded=False):
        if st.button("Restaurar default", use_container_width=True):
            for k, v in DEFAULT_WEIGHTS.items():
                st.session_state[f"w_{k}"] = v
            st.rerun()

        st.slider("SBERT (semântica)", 0.0, 1.0,
                  value=float(DEFAULT_WEIGHTS["sbert"]), step=0.05, key="w_sbert")
        st.slider("Descrição (tokens)", 0.0, 1.0,
                  value=float(DEFAULT_WEIGHTS["desc"]), step=0.05, key="w_desc")
        st.slider("Marca (tokens)", 0.0, 1.0,
                  value=float(DEFAULT_WEIGHTS["marca"]), step=0.05, key="w_marca")
        st.slider("Medida (numérica)", 0.0, 1.0,
                  value=float(DEFAULT_WEIGHTS["medida"]), step=0.05, key="w_medida")

        soma_pesos = (
            st.session_state.w_sbert
            + st.session_state.w_desc
            + st.session_state.w_marca
            + st.session_state.w_medida
        )
        if abs(soma_pesos - 1.0) > 0.01:
            st.caption(f"⚠️ Σ = {soma_pesos:.2f} (será renormalizada para 1.0)")
        else:
            st.caption(f"Σ = {soma_pesos:.2f}")

    st.divider()
    st.header("Feedback registrado")
    stats_fb = feedback.estatisticas_feedback()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", stats_fb['total'])
    c2.metric("✅", stats_fb['aprovados'])
    c3.metric("❌", stats_fb['reprovados'])


# ---------------- Main ----------------
st.title("🔎 Consulta por Similaridade — v2")
st.caption(
    "Combinação linear de SBERT + similaridade de tokens + comparação numérica. "
    "Marque os resultados como ✅ Aprovado ou ❌ Reprovado para alimentar a base "
    "de treinamento para fine-tuning."
)

tab_consulta, tab_feedback = st.tabs(["Consulta", "Feedback registrado"])


with tab_consulta:
    if not base_existe():
        st.info(
            "Sincronize a base de conhecimento no menu lateral antes da primeira consulta. "
            "A sincronização lê `TBL_INSUMOS` do Snowflake e processa apenas registros "
            "novos. Na primeira execução em uma máquina sem cache local, o app baixa "
            "embeddings já presentes na SF (rápido) e gera apenas os ausentes (mais lento)."
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
                medida = st.text_input("Medida (opcional)", placeholder="Ex: 12M, 1KG")
            submitted = st.form_submit_button(
                "Consultar", type="primary", use_container_width=True
            )

        if submitted:
            if not descricao.strip():
                st.error("A descrição é obrigatória.")
            else:
                similarity, _ = carregar_modulos()
                df_embeddings, df_pad, knn = carregar_indice(parquet_mtime(), n_neighbors)

                weights = {
                    "sbert": st.session_state.get("w_sbert", DEFAULT_WEIGHTS["sbert"]),
                    "desc": st.session_state.get("w_desc", DEFAULT_WEIGHTS["desc"]),
                    "marca": st.session_state.get("w_marca", DEFAULT_WEIGHTS["marca"]),
                    "medida": st.session_state.get("w_medida", DEFAULT_WEIGHTS["medida"]),
                }
                soma = sum(weights.values())
                if soma <= 0:
                    # Rede de segurança: todos os pesos zerados → fallback nos defaults.
                    # Sem isso, todos os scores ficariam 0 e nada passaria do threshold.
                    weights = dict(DEFAULT_WEIGHTS)
                    pesos_str = ", ".join(f"{k}={v}" for k, v in DEFAULT_WEIGHTS.items())
                    st.warning(f"Pesos estavam todos zerados — usando defaults ({pesos_str}).")
                else:
                    weights = {k: v / soma for k, v in weights.items()}

                with st.spinner("Buscando insumos similares..."):
                    resultados = similarity.consultar_item(
                        descricao=descricao.strip(),
                        medida=medida.strip(),
                        knn_model=knn,
                        df_pad=df_pad,
                        df_embeddings=df_embeddings,
                        marca=marca.strip(),
                        weights=weights,
                        threshold=float(threshold),
                    )
                st.session_state.resultados = resultados
                st.session_state.ultima_query = {
                    "descricao": descricao.strip(),
                    "marca": marca.strip(),
                    "medida": medida.strip(),
                }
                st.session_state.weights_snapshot = weights
                st.session_state.knn_k = int(n_neighbors)
                st.session_state.evaluated_rows = {}

        # Exibição/edição dos resultados (persiste entre reruns)
        if st.session_state.resultados is not None:
            df_res = st.session_state.resultados
            if df_res.empty:
                st.warning(
                    "Nenhum resultado acima do threshold. "
                    "Tente refinar descrição/marca/medida ou abaixar o threshold."
                )
            else:
                n_avaliados = len(st.session_state.evaluated_rows)
                col_h1, col_h2 = st.columns([3, 1])
                col_h1.success(
                    f"{len(df_res)} resultado(s) encontrado(s). "
                    f"Avaliações registradas nesta consulta: **{n_avaliados}**"
                )
                if col_h2.button("🗑️ Limpar resultados", use_container_width=True):
                    st.session_state.resultados = None
                    st.session_state.evaluated_rows = {}
                    st.rerun()

                filtrar_at = st.checkbox(
                    "Mostrar apenas insumos ativos (STATUS = AT)",
                    value=True,
                    key="filtrar_status_at",
                )

                if filtrar_at and "STATUS" in df_res.columns:
                    df_view = df_res[
                        df_res["STATUS"].astype(str).str.upper() == "AT"
                    ].copy()
                    st.caption(
                        f"Filtro ativo: exibindo **{len(df_view)}** de {len(df_res)} resultados (STATUS = AT)."
                    )
                else:
                    df_view = df_res

                if df_view.empty:
                    st.warning(
                        "Nenhum insumo com STATUS=AT entre os resultados. "
                        "Desligue o filtro acima para visualizar itens descontinuados."
                    )

                lista = st.container(height=620, border=False)
                with lista:
                    for _, row in df_view.iterrows():
                        rank = int(row["RANK"])
                        with st.container(border=True):
                            col_info, col_score, col_actions = st.columns([6, 1.5, 2.5])

                            with col_info:
                                st.markdown(
                                    f"**#{rank}** &nbsp; {row['INSUMO_DESCRICAO']}",
                                    unsafe_allow_html=True,
                                )
                                meta = (
                                    f"CD: `{row['CD_INSUMO']}` · GRP: `{row['GRP_INSUMO']}` · "
                                    f"Marca: **{row['MARCA']}** · Medida: **{row['MEDIDA']}**"
                                )
                                if 'STATUS' in row and pd.notna(row['STATUS']):
                                    meta += f" · Status: `{row['STATUS']}`"
                                st.caption(meta)
                                st.caption(
                                    f"SBERT: {float(row['SCORE_SBERT']):.3f} · "
                                    f"Desc: {float(row['SIM_DESC']):.3f} · "
                                    f"Marca: {float(row['SIM_MARCA']):.3f} · "
                                    f"Medida: {float(row['SIM_MEDIDA']):.3f}"
                                )

                            with col_score:
                                st.metric(
                                    "Score", f"{float(row['SCORE']):.1f}",
                                    label_visibility="collapsed",
                                )

                            with col_actions:
                                avaliado = st.session_state.evaluated_rows.get(rank)
                                if avaliado == 1:
                                    st.success("✅ Aprovado", icon="✅")
                                elif avaliado == 0:
                                    st.error("❌ Reprovado", icon="❌")
                                else:
                                    cb1, cb2 = st.columns(2)
                                    if cb1.button(
                                        "✅ Aprovar",
                                        key=f"ap_{rank}",
                                        use_container_width=True,
                                    ):
                                        _dialog_confirmar(row.to_dict(), 1)
                                    if cb2.button(
                                        "❌ Reprovar",
                                        key=f"rp_{rank}",
                                        use_container_width=True,
                                    ):
                                        _dialog_confirmar(row.to_dict(), 0)

                safe_name = "".join(
                    c if c.isalnum() else "_"
                    for c in (st.session_state.ultima_query or {}).get("descricao", "consulta")
                )[:50]
                csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Baixar resultados (CSV)",
                    data=csv_bytes,
                    file_name=f"resultados_{safe_name}.csv",
                    mime="text/csv",
                )


with tab_feedback:
    stats = feedback.estatisticas_detalhadas()

    # ---------- Indicador de prontidão (verde / X) ----------
    if stats["pronto_treino"]:
        st.success(
            "✅ **META ATINGIDA — base pronta para iniciar fine-tuning.** "
            f"Veja `docs/ROADMAP_FINETUNING.txt` para os próximos passos."
        )
    elif stats["total"] == 0:
        st.error(
            "❌ **Nenhuma validação ainda.** "
            f"Meta: {feedback.META_VALIDACOES} validações e "
            f"{feedback.META_QUERIES_UNICAS} queries únicas."
        )
    else:
        pct_val = min(100, int(100 * stats["total"] / feedback.META_VALIDACOES))
        pct_q = min(100, int(100 * stats["queries_unicas"] / feedback.META_QUERIES_UNICAS))
        st.error(
            f"❌ **Coletando dados** — faltam **{stats['faltam_validacoes']}** "
            f"validações ({pct_val}% da meta) e **{stats['faltam_queries']}** "
            f"queries únicas ({pct_q}% da meta) para liberar o fine-tuning."
        )

    # ---------- Métricas principais ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Validações",
        f"{stats['total']} / {feedback.META_VALIDACOES}",
        delta=(f"+{stats['total'] - feedback.META_VALIDACOES}"
               if stats['total'] >= feedback.META_VALIDACOES else None),
    )
    c2.metric(
        "Queries únicas",
        f"{stats['queries_unicas']} / {feedback.META_QUERIES_UNICAS}",
        delta=(f"+{stats['queries_unicas'] - feedback.META_QUERIES_UNICAS}"
               if stats['queries_unicas'] >= feedback.META_QUERIES_UNICAS else None),
    )
    c3.metric("✅ Aprovadas", stats["aprovados"])
    c4.metric("❌ Reprovadas", stats["reprovados"])

    # ---------- Barras de progresso ----------
    p_val = min(1.0, stats["total"] / feedback.META_VALIDACOES) if feedback.META_VALIDACOES else 0
    p_q = (
        min(1.0, stats["queries_unicas"] / feedback.META_QUERIES_UNICAS)
        if feedback.META_QUERIES_UNICAS else 0
    )
    st.progress(p_val, text=f"Validações: {int(p_val * 100)}% da meta")
    st.progress(p_q, text=f"Queries únicas: {int(p_q * 100)}% da meta")

    # ---------- Diagnóstico de balanço ----------
    if stats["total"] > 0:
        ap_pct, rp_pct = stats["balanco_pct"]
        if stats["desbalanceado"]:
            st.warning(
                f"⚠️ Distribuição desbalanceada: **{ap_pct}%** aprovadas vs "
                f"**{rp_pct}%** reprovadas. Recomendado coletar mais negativos "
                "para reduzir viés (ver Fase 1.4 do roadmap)."
            )
        else:
            st.caption(
                f"Distribuição: {ap_pct}% aprovadas · {rp_pct}% reprovadas — "
                f"{stats['queries_3mais']} queries com 3+ validações."
            )

    st.divider()

    if stats["total"] == 0:
        st.info(
            "Marque resultados na aba **Consulta** com ✅ Aprovar / ❌ Reprovar "
            "para começar a alimentar a base de treino."
        )
    else:
        # ---------- Detalhamento ----------
        with st.expander("📊 Distribuição por rank", expanded=False):
            if stats["ranks_dist"]:
                df_ranks = pd.DataFrame({
                    "Rank": list(stats["ranks_dist"].keys()),
                    "Validações": list(stats["ranks_dist"].values()),
                }).sort_values("Rank")
                st.bar_chart(df_ranks, x="Rank", y="Validações")
            else:
                st.caption("Sem dados de rank.")

        with st.expander("🏆 Top 10 queries por volume", expanded=False):
            if stats["top_queries"]:
                st.dataframe(
                    pd.DataFrame(stats["top_queries"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("Sem queries.")

        with st.expander("📋 Registros brutos (JSONL → tabela)", expanded=False):
            df_fb = feedback.carregar_feedback()
            st.dataframe(df_fb, use_container_width=True, hide_index=True)
            if FEEDBACK_JSONL.exists():
                st.download_button(
                    "⬇️ Baixar feedback.jsonl",
                    data=FEEDBACK_JSONL.read_bytes(),
                    file_name="feedback.jsonl",
                    mime="application/x-ndjson",
                )
