# Projeto FGV — Consulta por Similaridade (SPDO)

Sistema de busca de insumos similares a partir de descrição livre, marca e medida.
Combina **embeddings semânticos SBERT** (modelo multilíngue PT-BR) com **busca KNN**
(distância cosseno) e uma camada de **penalização Levenshtein** sobre marca, medida e
descrição padronizadas.

## Estrutura do projeto

```
projeto_fgv_similaridade/
├── app.py                          # Entry point do Streamlit
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml                # Credenciais Snowflake (NUNCA commitar)
│
├── src/                            # Código de domínio
│   ├── config.py                   # Paths, pesos default, versão
│   ├── data_process.py             # Padronização de texto/medidas, stopwords
│   ├── penalty.py                  # Penalização v2 (token + numérica + linear)
│   ├── similarity.py               # SBERT + KNN + consulta
│   ├── knowledge_base.py           # Sincronização incremental Snowflake↔local
│   ├── snowflake_io.py             # Conexão SF, DDL idempotente, read/write
│   ├── feedback.py                 # Persistência de validações (JSONL)
│   └── evaluation.py               # Esqueleto de avaliação (recall@k, MRR)
│
├── streamlit_app/                  # Camada de apresentação
│   └── services.py                 # Wrappers cacheados (modelo, KNN, embeddings)
│
├── scripts/
│   └── seed_snowflake.py           # Bootstrap one-time do parquet → SF
│
├── tests/
│   └── test_penalty.py             # 17 testes do módulo de penalização
│
├── notebooks/                      # Pipeline original (referência)
│   ├── 0.padronizar_dados.ipynb
│   ├── 1.embeddings.ipynb
│   └── 2.similarity_search.ipynb
│
└── data/
    ├── staging/                    # Cache local (mirror do SF, regenerável)
    │   ├── medida_correlacao.csv
    │   ├── df_pad.csv
    │   └── embeddings_bp.parquet   # cache do KNN em runtime
    ├── training/
    │   └── feedback.jsonl          # validações dos usuários (fine-tuning)
    ├── eval/
    │   └── gold_standard_template.csv
    └── output/                     # CSVs de resultado exportados
```

## Arquitetura de dados

```
Snowflake (fonte de verdade)                   Local (runtime)
────────────────────────────                   ───────────────
TBL_INSUMOS                  ── leitura ─────► (raw)
TBL_INSUMOS_PADRONIZADOS     ◄── escrita ──── df_pad
TBL_INSUMOS_PREPROCESSADOS   ◄── escrita ──── df_embeddings
TBL_INSUMOS_EMBEDDINGS       ◄── escrita ──── embeddings_bp.parquet
                              (ARRAY)              │
TBL_MEDIDAS_CORRELACAO       ◄── escrita ──── medida_correlacao.csv
                                                  ▼
                                          sklearn NearestNeighbors
                                          (KNN em memória)
```

- **SF é durável** — múltiplas máquinas/desenvolvedores compartilham a mesma base de embeddings.
- **Parquet local é cache** — necessário para o KNN em memória rodar rápido.
- **Sincronização incremental** detecta `CD_INSUMO` ausentes em SF e gera embeddings só para esses.
- **Rehidratação automática** — se o cache local estiver vazio em uma máquina nova, o app baixa embeddings da SF (rápido, não regera).

## Instalação

```bash
pip install -r requirements.txt
```

## Configuração das credenciais Snowflake

Crie `.streamlit/secrets.toml` (já está no `.gitignore`):

```toml
[snowflake]
account = "seu_account_identifier"
user = "seu_usuario"
password = "sua_senha"
role = "BASES_SPDO"
warehouse = "..."
database = "BASES_SPDO"
schema = "DB_GESTAO_BANCO_PRECO_APP_CONSULTA"
```

## Execução

### Bootstrap (rodar UMA vez por ambiente novo)

Se você já tem `data/staging/embeddings_bp.parquet` localmente (embeddings já
gerados em outra etapa), sobe para a SF para evitar regerar:

```bash
python scripts/seed_snowflake.py
```

O script é idempotente — pode rodar quantas vezes quiser, só insere o que
falta. Se o parquet local não existir, pule essa etapa e use o app direto.

### Frontend Streamlit

```bash
streamlit run app.py
```

Fluxo no app:

1. **Primeiro acesso a uma máquina** — clique em **Sincronizar base** na
   sidebar. O app:
   - Lê `TBL_INSUMOS` na SF.
   - Baixa embeddings já presentes na SF para o cache local (rápido).
   - Gera embeddings só para `CD_INSUMO` que não estão em `TBL_INSUMOS_EMBEDDINGS`.
   - Insere os novos nas tabelas SF (`TBL_INSUMOS_PADRONIZADOS`,
     `TBL_INSUMOS_PREPROCESSADOS`, `TBL_INSUMOS_EMBEDDINGS`).
2. **Acessos seguintes** — apenas itens novos em `TBL_INSUMOS` são processados.
3. **Consultar** — preencha descrição (obrigatório), marca e medida (opcionais),
   ajuste pesos e threshold na sidebar.
4. **Validar/Reprovar matches** — botões por linha geram registros em
   `data/training/feedback.jsonl` para fine-tuning futuro.

### Pipeline em notebooks (referência)

Os notebooks reproduzem o pipeline original em três etapas:

1. `notebooks/0.padronizar_dados.ipynb` — padroniza a base bruta.
2. `notebooks/1.embeddings.ipynb` — gera os embeddings SBERT.
3. `notebooks/2.similarity_search.ipynb` — treina KNN e consulta.

Os notebooks fazem `chdir` automático para a raiz, então podem ser abertos
diretamente de `notebooks/` sem ajuste manual de path.

## Como funciona a sincronização incremental

`src/knowledge_base.sincronizar_base()`:

1. Lê `data/input/consulta_bp.csv` e aplica a padronização do notebook 0
   (`fillna`, criação de `INSUMO_DESCRICAO`, `MEDIDA_PAD`, `MEDIDA_ABV`,
   `padronizar_medida`, `preprocess_text`, `remove_stopwords`).
2. Regrava `df_pad.csv` e `medida_correlacao.csv` (são baratos).
3. Se `embeddings_bp.parquet` existir, lê os `CD_INSUMO` já presentes;
   senão, considera primeira carga.
4. Codifica com SBERT apenas os registros novos.
5. Concatena e grava o parquet atualizado.

Resultado: na primeira execução, processa tudo. Depois, só itens novos.

## Configuração

`src/config.py` centraliza todos os caminhos. Mude lá se precisar mover
diretórios ou apontar para outra base.

## Dados de entrada

`data/input/consulta_bp.csv` é esperado com as colunas:

```
GRP_INSUMO, CD_INSUMO, INSUMO, DESCRICAO, MARCA, CD_MEDIDA, MEDIDA,
QTD_MEDIDA, EMBALAGEM, STATUS
```

`CD_INSUMO` é a chave usada para detectar registros novos na sincronização
incremental.
