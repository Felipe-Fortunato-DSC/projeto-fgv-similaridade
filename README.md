# Projeto FGV — Consulta por Similaridade (SPDO)

Sistema de busca de insumos similares a partir de descrição livre, marca e medida.
Combina **embeddings semânticos SBERT** (multilíngue PT-BR) com **busca k-NN no
S3 Vectors** e uma camada de **scoring composto** (similaridade de tokens +
comparação numérica de medida). Captura feedback de validação para fine-tuning
futuro do modelo de embedding.

## Stack

- **Python 3.11+**
- **Streamlit** (frontend)
- **AWS Athena / Iceberg** (dados tabulares — fonte de verdade)
- **AWS S3 Vectors** (armazenamento e busca k-NN dos embeddings)
- **sentence-transformers** (SBERT)
- **rapidfuzz** (similaridade de tokens)
- **awswrangler / boto3** (integração AWS)
- Deploy: **Docker → ECR → ECS (Fargate)**

## Arquitetura

```
┌───────────────────────────── AWS (fonte de verdade) ─────────────────────────────┐
│  Athena / Glue  (db_spdo_apps, Iceberg)            S3 Vectors (spdo-embeddings)   │
│  ───────────────────────────────────────          ─────────────────────────────  │
│  tbl_insumos                 (fonte bruta)         índice insumos-sbert-384       │
│  tbl_insumos_padronizados    (display)             (vetores 384d, cosine)         │
│  tbl_insumos_preprocessados  (penalização)               ▲                        │
│  tbl_insumos_embeddings      (manifesto)  ──── cd_insumo ─┘                        │
│  tbl_medidas_correlacao                                                            │
│  tbl_feedback_validacoes                                                           │
└───────────────────────────────────────────────────────────────────────────────────┘
            ▲ metadados (cache em memória)        ▲ encode query → query_vectors(top-K)
            │                                      │
        ┌───┴──────────────────────────────────────┴───┐
        │  App Streamlit (local ou ECS/Fargate)         │
        │  SBERT encode → S3 Vectors → penalização v2   │
        └───────────────────────────────────────────────┘
```

- **Busca**: a query é codificada pelo SBERT; o S3 Vectors retorna o top-K
  `cd_insumo` por distância cosseno; os candidatos são hidratados a partir dos
  metadados (cacheados em memória) e reordenados pela penalização v2.
- **Sem cache local de vetores** — sem parquet de 764 MB, sem KNN em memória.
- **Sincronização incremental** detecta `cd_insumo` em `tbl_insumos` ainda sem
  embedding, codifica só esses e grava no S3 Vectors + manifesto no Athena.
- **Feedback** vai para `tbl_feedback_validacoes` (Athena); JSONL local é buffer
  best-effort.

## Estrutura do projeto

```
projeto_fgv_similaridade/
├── app.py                          # Entry point do Streamlit
├── requirements.txt
├── pyproject.toml                  # Metadata + config de tooling (pytest, ruff)
├── Makefile                        # run / test / seed / lint / format
├── Dockerfile / .dockerignore      # Imagem de produção (ECR/ECS)
├── README.md
│
├── .streamlit/config.toml          # Theme + page config
│
├── src/                            # Código de domínio (pacote Python)
│   ├── config.py                   # Paths, parâmetros do modelo, config AWS
│   ├── data_process.py             # Padronização de texto/medidas, stopwords
│   ├── penalty.py                  # Scoring composto v2 (token + numérica + linear)
│   ├── similarity.py               # SBERT encode + S3 Vectors + penalização
│   ├── knowledge_base.py           # Sincronização incremental (Athena → S3 Vectors)
│   ├── aws_io.py                   # Athena (awswrangler) + S3 Vectors (boto3)
│   ├── feedback.py                 # Persistência de validações (Athena + JSONL)
│   └── evaluation.py               # Esqueleto Recall@k, MRR (uso futuro)
│
├── streamlit_app/services.py       # Wrappers cacheados (modelo, metadados, feedback)
│
├── scripts/seed_aws.py             # Seed one-time: artefatos locais → Athena + S3 Vectors
│
├── tests/test_penalty.py           # Testes do módulo de scoring
│
├── deploy/                         # Tudo de infra/deploy
│   ├── athena/schema.sql           # DDL Iceberg (estrutura das tabelas)
│   ├── athena/query_patterns.sql   # Padrões de query (referência)
│   ├── athena/s3vectors_setup.md   # Criação do índice S3 Vectors
│   ├── build_and_push.ps1          # Build + push para o ECR
│   ├── ecs-task-definition.json    # Task definition Fargate (template)
│   ├── DEPLOY_ECS.md               # Guia de deploy
│   └── .env.example                # Config AWS para teste local
│
├── legacy/                         # Material histórico (notebooks + roadmap)
│
└── data/                           # Gitignored (artefatos do seed + buffer de feedback)
```

## Setup local (apontando para a AWS real)

```powershell
# 1. Dependências
pip install -r requirements.txt

# 2. Credenciais AWS (region us-east-1)
aws configure

# 3. Criar a estrutura na AWS (uma vez)
#    - Athena: rodar deploy/athena/schema.sql
#    - S3 Vectors: deploy/athena/s3vectors_setup.md (create-index)

# 4. Semear dados a partir dos artefatos locais (data/staging/*)
python scripts/seed_aws.py

# 5. Rodar a app
streamlit run app.py
```

## Fluxo no app

1. **Auto-sync** ao abrir — verifica novos itens em `tbl_insumos`, codifica e
   grava no S3 Vectors; pré-carrega metadados.
2. **Consulta** — descrição (obrigatória), marca e medida (opcionais); pesos e
   threshold ajustáveis na sidebar.
3. **Filtro STATUS=AT** ligado por default.
4. **Validar/Reprovar matches** — grava em `tbl_feedback_validacoes`.
5. **Aba "Feedback registrado"** — dashboard de prontidão para fine-tuning.

## Comandos rápidos

```bash
make install   # pip install -r requirements.txt
make run       # streamlit run app.py
make test      # pytest
make seed      # seed AWS (one-time)
make lint      # ruff check
make format    # ruff format
```

## Deploy (ECR + ECS)

Ver **`deploy/DEPLOY_ECS.md`**. Resumo: `deploy/build_and_push.ps1 -RepoName <repo>`
e registrar `deploy/ecs-task-definition.json`. Acesso à AWS via **task role**
(sem secrets).

## Testes

```bash
pytest
```
