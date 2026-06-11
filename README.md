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
├── src/                            # Código de domínio (pacote Python)
│   ├── __init__.py
│   ├── config.py                   # Paths e constantes
│   ├── data_process.py             # Padronização de texto/medidas, stopwords
│   ├── penalty.py                  # Penalização Levenshtein
│   ├── similarity.py               # SBERT + KNN + consulta
│   └── knowledge_base.py           # Sincronização incremental de embeddings
│
├── streamlit_app/                  # Camada de apresentação
│   ├── __init__.py
│   └── services.py                 # Wrappers cacheados (modelo, KNN, embeddings)
│
├── notebooks/                      # Pipeline original em notebooks
│   ├── 0.padronizar_dados.ipynb
│   ├── 1.embeddings.ipynb
│   └── 2.similarity_search.ipynb
│
└── data/
    ├── input/
    │   └── consulta_bp.csv         # Base bruta (entrada)
    ├── staging/                    # Artefatos intermediários (gerados)
    │   ├── medida_correlacao.csv
    │   ├── df_pad.csv
    │   ├── df_embeddings.csv
    │   └── embeddings_bp.parquet   # Base de conhecimento de embeddings
    └── output/                     # Resultados de consulta exportados
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

### Frontend Streamlit (recomendado)

A partir da raiz do projeto:

```bash
streamlit run app.py
```

Fluxo no app:

1. **Primeira execução** — clique em **Sincronizar base** na barra lateral.
   Toda a base `data/input/consulta_bp.csv` será padronizada e codificada com
   SBERT. Isso pode levar **vários minutos** (pode passar de uma hora dependendo
   do hardware).
2. **Execuções seguintes** — clique em **Sincronizar base** novamente.
   Apenas os `CD_INSUMO` ausentes do parquet de embeddings serão processados;
   o restante é reaproveitado.
3. **Consultar** — preencha descrição (obrigatório), marca e medida (opcionais)
   e ajuste o `k` (vizinhos) na sidebar. Resultados são filtrados por score > 50.

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
