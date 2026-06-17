-- =====================================================================
-- Estrutura Athena (Apache Iceberg) — migração das tabelas do Snowflake
-- Projeto: Consulta por Similaridade FGV (v2)
--
-- AMBIENTE:
--   Database (Glue) : db_spdo_apps
--   Dados (S3)      : s3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/<tabela>/
--   Vetores         : S3 Vectors (bucket 'spdo-embeddings') -> ver s3vectors_setup.md
--
-- Por que Iceberg: o app faz INSERT incremental, MERGE/upsert e TRUNCATE.
-- Tabela Hive "external" não suporta esses writes; Iceberg suporta
-- INSERT/MERGE/UPDATE/DELETE (ACID) no Athena.
--
-- IMPORTANTE — os EMBEDDINGS não ficam mais no Athena:
--   Os vetores vão para o S3 Vectors (busca k-NN gerenciada). A tabela
--   tbl_insumos_embeddings vira um REGISTRO/manifesto leve (sem array<float>),
--   usado só para o diff incremental barato (LEFT JOIN). cd_insumo é a chave
--   de junção entre Athena e o índice do S3 Vectors.
--
-- Antes de rodar: configure no workgroup do Athena o "Query result location".
--
-- Notas (diferenças vs Snowflake, tratadas na fase de código):
--   * Sem DEFAULT CURRENT_TIMESTAMP -> preencher *_AT no INSERT/MERGE.
--   * Sem PRIMARY KEY enforçada    -> unicidade via MERGE INTO (upsert).
--   * VARIANT -> string (JSON texto), lido com json_extract/json_parse.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 0. Database (schema) no Glue Data Catalog — JÁ CRIADO: db_spdo_apps
--    (mantido aqui só como referência; IF NOT EXISTS é no-op se já existe)
-- ---------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS db_spdo_apps;


-- ---------------------------------------------------------------------
-- 1. TBL_INSUMOS — fonte bruta (origem da padronização).
--    Ajuste os tipos para espelhar EXATAMENTE a tabela atual no Snowflake.
--    Colunas inferidas pelo uso em knowledge_base._padronizar_brutos /
--    snowflake_io.ler_medidas_distintas.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS db_spdo_apps.tbl_insumos (
    grp_insumo   string,
    cd_insumo    bigint,
    insumo       string,
    descricao    string,
    marca        string,
    embalagem    string,
    qtd_medida   double,
    medida       string,
    cd_medida    string,
    status       string
)
LOCATION 's3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/tbl_insumos/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format'     = 'parquet'
);


-- ---------------------------------------------------------------------
-- 2. TBL_INSUMOS_PADRONIZADOS — equivalente a df_pad.csv
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS db_spdo_apps.tbl_insumos_padronizados (
    grp_insumo        string,
    cd_insumo         bigint,
    insumo_descricao  string,
    marca             string,
    medida            string,
    status            string,
    updated_at        timestamp
)
LOCATION 's3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/tbl_insumos_padronizados/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format'     = 'parquet'
);


-- ---------------------------------------------------------------------
-- 3. TBL_INSUMOS_PREPROCESSADOS — equivalente a df_embeddings.csv (sem vetor)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS db_spdo_apps.tbl_insumos_preprocessados (
    grp_insumo        string,
    cd_insumo         bigint,
    insumo_descricao  string,
    marca             string,
    medida            string,
    updated_at        timestamp
)
LOCATION 's3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/tbl_insumos_preprocessados/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format'     = 'parquet'
);


-- ---------------------------------------------------------------------
-- 4. TBL_INSUMOS_EMBEDDINGS — REGISTRO/manifesto dos vetores (NÃO guarda o vetor!)
--    Os vetores ficam no S3 Vectors (índice em 'spdo-embeddings'); aqui só
--    rastreamos quais cd_insumo já foram codificados, para o diff incremental
--    barato (LEFT JOIN). cd_insumo é a chave de junção com o índice vetorial.
--    Dimensão dos vetores: 384 (paraphrase-multilingual-MiniLM-L12-v2).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS db_spdo_apps.tbl_insumos_embeddings (
    cd_insumo      bigint,
    model_name     string,
    embedding_dim  int,
    vector_key     string,    -- chave do vetor no S3 Vectors (ex.: o próprio cd_insumo)
    created_at     timestamp
)
LOCATION 's3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/tbl_insumos_embeddings/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format'     = 'parquet'
);


-- ---------------------------------------------------------------------
-- 5. TBL_MEDIDAS_CORRELACAO — tabela pequena (TRUNCATE+INSERT no Snowflake;
--    no Iceberg vira DELETE-all + INSERT, ou CREATE OR REPLACE TABLE AS).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS db_spdo_apps.tbl_medidas_correlacao (
    cd_medida  string,
    medida     string
)
LOCATION 's3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/tbl_medidas_correlacao/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format'     = 'parquet'
);


-- ---------------------------------------------------------------------
-- 6. TBL_FEEDBACK_VALIDACOES — feedback humano para fine-tuning.
--    WEIGHTS_SNAPSHOT: VARIANT no Snowflake -> string (JSON texto) no Iceberg.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS db_spdo_apps.tbl_feedback_validacoes (
    feedback_id         string,
    timestamp_utc       timestamp,
    session_id          string,
    user_descricao      string,
    user_marca          string,
    user_medida         string,
    match_cd_insumo     bigint,
    match_grp_insumo    string,
    match_descricao     string,
    match_marca         string,
    match_medida        string,
    match_status        string,
    score_sbert         double,
    score_desc_tokens   double,
    score_marca_tokens  double,
    score_medida_num    double,
    score_final         double,
    rank_posicao        int,
    label               int,
    app_version         string,
    knn_k               int,
    weights_snapshot    string
)
LOCATION 's3://ibre-spdo-coleta-bronze/apps/consulta_similaridade/tbl_feedback_validacoes/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format'     = 'parquet'
);
