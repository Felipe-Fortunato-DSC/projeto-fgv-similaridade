-- =====================================================================
-- Padrões de query no Athena/Iceberg — mapeando snowflake_io.py
-- Referência para a fase de adaptação do código. Database: db_spdo_apps.
-- =====================================================================


-- ---------------------------------------------------------------------
-- contar_totais()  ->  totais em uma ida
-- ---------------------------------------------------------------------
SELECT
    (SELECT count(*) FROM db_spdo_apps.tbl_insumos)            AS n_brutos,
    (SELECT count(*) FROM db_spdo_apps.tbl_insumos_embeddings) AS n_codificados;


-- ---------------------------------------------------------------------
-- ler_insumos_brutos_novos()  ->  server-side diff (apenas CDs não codificados)
-- ---------------------------------------------------------------------
SELECT i.*
FROM db_spdo_apps.tbl_insumos i
LEFT JOIN db_spdo_apps.tbl_insumos_embeddings e
       ON i.cd_insumo = e.cd_insumo
WHERE e.cd_insumo IS NULL;


-- ---------------------------------------------------------------------
-- ler_medidas_distintas()
-- ---------------------------------------------------------------------
SELECT DISTINCT cd_medida, medida FROM db_spdo_apps.tbl_insumos;


-- ---------------------------------------------------------------------
-- ler_cd_insumos_codificados()
-- ---------------------------------------------------------------------
SELECT cd_insumo FROM db_spdo_apps.tbl_insumos_embeddings;


-- ---------------------------------------------------------------------
-- BUSCA DE SIMILARIDADE  ->  agora é S3 Vectors (query_vectors), NÃO Athena.
-- Ver deploy/athena/s3vectors_setup.md. O Athena entra só para buscar os
-- CAMPOS dos candidatos retornados pelo S3 Vectors (top-K cd_insumo) e
-- aplicar a penalização. Os vetores em si não vivem mais aqui.
-- ---------------------------------------------------------------------
-- Hidratar os candidatos do top-K (cd_insumo vindos do query_vectors):
SELECT
    p.grp_insumo, p.cd_insumo, p.insumo_descricao, p.marca, p.medida
FROM db_spdo_apps.tbl_insumos_preprocessados p
WHERE p.cd_insumo IN ( /* cd_insumo do top-K do S3 Vectors */ );


-- ---------------------------------------------------------------------
-- insert_padronizados_novos()  ->  INSERT (timestamp setado aqui, sem DEFAULT)
-- Para garantir idempotência, prefira MERGE (upsert) por cd_insumo.
-- ---------------------------------------------------------------------
MERGE INTO db_spdo_apps.tbl_insumos_padronizados t
USING (
    -- staging: leia de uma tabela temporária Iceberg ou de um SELECT VALUES
    SELECT * FROM db_spdo_apps.stg_padronizados
) s
ON t.cd_insumo = s.cd_insumo
WHEN NOT MATCHED THEN INSERT (
    grp_insumo, cd_insumo, insumo_descricao, marca, medida, status, updated_at
) VALUES (
    s.grp_insumo, s.cd_insumo, s.insumo_descricao, s.marca, s.medida, s.status,
    current_timestamp
);


-- ---------------------------------------------------------------------
-- insert_embeddings_novos()  ->  DOIS passos:
--   (a) put_vectors no S3 Vectors (ver s3vectors_setup.md) — grava o vetor;
--   (b) registrar no manifesto abaixo (Athena) para o diff incremental.
-- ---------------------------------------------------------------------
INSERT INTO db_spdo_apps.tbl_insumos_embeddings
    (cd_insumo, model_name, embedding_dim, vector_key, created_at)
SELECT cd_insumo, model_name, embedding_dim, vector_key, current_timestamp
FROM db_spdo_apps.stg_embeddings;


-- ---------------------------------------------------------------------
-- regravar_medida_correlacao()  ->  substituição total
-- Iceberg não tem TRUNCATE; use DELETE-all + INSERT, ou recrie a tabela.
-- ---------------------------------------------------------------------
DELETE FROM db_spdo_apps.tbl_medidas_correlacao;
INSERT INTO db_spdo_apps.tbl_medidas_correlacao (cd_medida, medida)
SELECT DISTINCT cd_medida, medida FROM db_spdo_apps.tbl_insumos;


-- ---------------------------------------------------------------------
-- insert_feedback()  ->  WEIGHTS_SNAPSHOT como JSON string
-- No código: json.dumps(weights_snapshot) antes de inserir.
-- Leitura depois: json_extract(weights_snapshot, '$.sbert') etc.
-- ---------------------------------------------------------------------
INSERT INTO db_spdo_apps.tbl_feedback_validacoes
    (feedback_id, timestamp_utc, session_id, user_descricao, user_marca,
     user_medida, match_cd_insumo, match_grp_insumo, match_descricao,
     match_marca, match_medida, match_status, score_sbert, score_desc_tokens,
     score_marca_tokens, score_medida_num, score_final, rank_posicao, label,
     app_version, knn_k, weights_snapshot)
SELECT
    feedback_id, timestamp_utc, session_id, user_descricao, user_marca,
    user_medida, match_cd_insumo, match_grp_insumo, match_descricao,
    match_marca, match_medida, match_status, score_sbert, score_desc_tokens,
    score_marca_tokens, score_medida_num, score_final, rank_posicao, label,
    app_version, knn_k, weights_snapshot
FROM db_spdo_apps.stg_feedback;
