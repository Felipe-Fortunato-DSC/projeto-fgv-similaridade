# Projeto FGV — Consulta por Similaridade (SPDO)

Sistema de busca de insumos similares a partir de descrição livre, marca e medida.
Combina **embeddings semânticos SBERT** (multilíngue PT-BR) com **busca KNN**
(cosseno) e uma camada de **scoring composto** (similaridade de tokens + comparação
numérica de medida). Inclui captura de feedback de validação para fine-tuning
futuro do modelo de embedding.

## Stack

- **Python 3.11+**
- **Streamlit** (frontend)
- **Snowflake** (fonte de verdade dos dados)
- **sentence-transformers** (SBERT)
- **scikit-learn** (KNN)
- **rapidfuzz** (similaridade de tokens)
- **pandas / numpy / pyarrow**

## Estrutura do projeto

```
projeto_fgv_similaridade/
├── app.py                          # Entry point do Streamlit
├── requirements.txt
├── pyproject.toml                  # Metadata + config de tooling (pytest, ruff)
├── Makefile                        # Comandos comuns (run, test, seed, lint)
├── README.md
├── .gitignore
│
├── .streamlit/
│   ├── config.toml                 # Theme + page config
│   ├── secrets.toml                # Credenciais Snowflake (gitignored)
│   └── secrets.toml.example        # Template para onboarding
│
├── src/                            # Código de domínio (pacote Python)
│   ├── config.py                   # Paths, pesos default, versão
│   ├── data_process.py             # Padronização de texto/medidas, stopwords
│   ├── penalty.py                  # Scoring composto v2 (token + numérica + linear)
│   ├── similarity.py               # SBERT + KNN + consulta
│   ├── knowledge_base.py           # Sincronização incremental Snowflake↔local
│   ├── snowflake_io.py             # Conexão SF, DDL idempotente, read/write
│   ├── feedback.py                 # Persistência de validações em SF + JSONL backup
│   └── evaluation.py               # Esqueleto Recall@k, MRR (uso futuro)
│
├── streamlit_app/
│   └── services.py                 # Wrappers cacheados (modelo, KNN, feedback)
│
├── scripts/
│   └── seed_snowflake.py           # Bootstrap one-time: parquet local → SF
│
├── tests/
│   └── test_penalty.py             # 17 testes do módulo de scoring
│
├── notebooks/                      # Pipeline original (referência histórica)
│   ├── README.md
│   ├── 0.padronizar_dados.ipynb
│   ├── 1.embeddings.ipynb
│   └── 2.similarity_search.ipynb
│
├── docs/
│   └── ROADMAP_FINETUNING.txt      # Plano de fine-tuning pós-meta de feedback
│
└── data/                           # Tudo gitignored (cache local + input)
    ├── input/                      # CSV bruto (backup/dev)
    ├── staging/                    # Cache do KNN (mirror do SF)
    ├── training/feedback.jsonl     # Buffer best-effort do feedback
    ├── eval/gold_standard_template.csv
    └── output/                     # CSVs exportados de consulta
```

## Arquitetura de dados

```
Snowflake (fonte de verdade)                Local (cache de runtime)
─────────────────────────────                ─────────────────────────
TBL_INSUMOS                  ── leitura ──►  (raw)
TBL_INSUMOS_PADRONIZADOS     ◄── escrita ─── df_pad
TBL_INSUMOS_PREPROCESSADOS   ◄── escrita ─── df_embeddings
TBL_INSUMOS_EMBEDDINGS       ◄── escrita ─── embeddings_bp.parquet
                              (ARRAY)              │
TBL_MEDIDAS_CORRELACAO       ◄── escrita ─── medida_correlacao.csv
TBL_FEEDBACK_VALIDACOES      ◄── escrita ─── feedback.jsonl (backup)
                                                   ▼
                                          sklearn NearestNeighbors
                                          (KNN em memória)
```

- **SF é durável e centralizada** — múltiplas máquinas/usuários compartilham
  a mesma base de embeddings e o mesmo histórico de feedback.
- **Parquet local é cache** — necessário para o KNN em memória rodar rápido.
- **Sincronização incremental** detecta `CD_INSUMO` ausentes em SF e gera
  embeddings só para esses. No idle (nada novo) o sync termina em ~8s.
- **Rehidratação automática** — se o cache local estiver vazio em uma máquina
  nova, o app baixa embeddings da SF (rápido, não regera).
- **Feedback de validação** vai para SF como fonte de verdade; JSONL local é
  buffer best-effort (útil em dev, irrelevante no Streamlit Cloud).

## Setup local

```powershell
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar credenciais Snowflake
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# editar .streamlit\secrets.toml com as credenciais reais

# 3. Bootstrap (se já tem embeddings_bp.parquet local — evita regerar)
python scripts/seed_snowflake.py

# 4. Rodar a app
streamlit run app.py
```

## Setup no Streamlit Community Cloud

1. Push do repositório para o GitHub.
2. Em https://share.streamlit.io, conectar o repo e apontar para `app.py`.
3. Em **App settings → Secrets**, colar o conteúdo do `secrets.toml` (com
   credenciais reais). Streamlit Cloud expõe via `st.secrets`.
4. Primeiro acesso → modal de sincronização baixa embeddings do SF.

## Fluxo no app

1. **Auto-sync** ao abrir — modal mostra progresso; cache local de
   KNN é pré-aquecido.
2. **Consulta** — preencher descrição (obrigatório), marca e medida (opcionais),
   ajustar pesos e threshold na sidebar.
3. **Filtro STATUS=AT** ligado por default — esconde insumos descontinuados.
4. **Validar/Reprovar matches** — botões por linha com popup de confirmação.
   Cada validação grava um registro em `TBL_FEEDBACK_VALIDACOES` no SF.
5. **Aba "Feedback registrado"** — dashboard de prontidão para fine-tuning
   (meta: 1.000 validações + 300 queries únicas).

## Comandos rápidos

```bash
make install   # pip install -r requirements.txt
make run       # streamlit run app.py
make test      # pytest
make seed      # bootstrap one-time
make lint      # ruff check
make format    # ruff format
```

## Roadmap de fine-tuning

Plano completo em **`docs/ROADMAP_FINETUNING.txt`** — 7 fases após a meta de
coleta:

1. Preparação dos dados (EDA, dedup, split estratificado por query)
2. Hard negative mining
3. Loop de treino com OnlineContrastiveLoss
4. Avaliação contra baseline
5. Versionamento e deploy
6. A/B test no app
7. Re-treino contínuo

## Testes

```bash
pytest
# 17 testes do penalty (edge cases dos bugs corrigidos + comportamentos novos)
```
