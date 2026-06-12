# Notebooks

Estes notebooks são a **referência histórica** do pipeline original do projeto,
antes da migração para o Snowflake. Reproduzem o fluxo end-to-end em três
etapas usando arquivos CSV/Parquet locais.

## Status atual

A app em produção (`app.py`) **não depende mais destes notebooks**. A fonte de
verdade dos dados é o Snowflake (`TBL_INSUMOS`, `TBL_INSUMOS_EMBEDDINGS`, etc.)
e a sincronização incremental fica em `src/knowledge_base.py`.

Eles foram mantidos no repositório como:

1. **Documentação executável** do pipeline original — útil para entender o que
   o `_padronizar_brutos` e o `gerar_embeddings` fazem internamente.
2. **Sandbox de experimentação** — se precisar testar uma mudança no
   pré-processamento, é mais rápido iterar no notebook que rodar a app inteira.
3. **Disaster recovery** — se a SF ficar indisponível e for preciso rebuildar
   a base de embeddings do zero a partir do CSV original.

## Como rodar

Os notebooks fazem `chdir` automático para a raiz do projeto e usam imports do
pacote `src.*`. Requisitos:

- Ter `data/input/consulta_bp.csv` localmente (a base bruta da FGV)
- Dependências do `requirements.txt` instaladas

Ordem:

1. `0.padronizar_dados.ipynb` — lê o CSV, aplica limpeza/normalização, gera
   bases intermediárias em `data/staging/`
2. `1.embeddings.ipynb` — gera embeddings SBERT (paraphrase-multilingual-
   MiniLM-L12-v2) → `data/staging/embeddings_bp.parquet`
3. `2.similarity_search.ipynb` — treina KNN, faz uma consulta exemplo

Após rodar os três, o estado local fica equivalente ao que a sincronização da
app produziria a partir do Snowflake.
