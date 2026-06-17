# S3 Vectors — índice de embeddings

Os vetores SBERT deixam de ser carregados em memória (sklearn KNN) e passam a
viver no **S3 Vectors**, que faz a busca k-NN de forma gerenciada.

## Modelo

| Item | Valor |
|---|---|
| Vector bucket | `spdo-embeddings` (já criado) |
| Index name | `insumos-sbert-384` (sugestão) |
| Dimension | `384` (paraphrase-multilingual-MiniLM-L12-v2) |
| Data type | `float32` |
| Distance metric | `cosine` (mesma métrica do KNN atual) |
| Chave do vetor | `cd_insumo` (string) — junção com `db_spdo_apps.tbl_insumos_*` |

> A dimensão e a métrica são **imutáveis** após criar o índice. Se um dia trocar
> o modelo SBERT, crie um novo índice (ex.: `insumos-<modelo>-<dim>`).

---

## 1. Criar o índice

### Via AWS CLI
```bash
aws s3vectors create-index \
  --vector-bucket-name spdo-embeddings \
  --index-name insumos-sbert-384 \
  --data-type float32 \
  --dimension 384 \
  --distance-metric cosine \
  --region us-east-1
```

### Via boto3 (equivalente)
```python
import boto3
s3v = boto3.client("s3vectors", region_name="us-east-1")
s3v.create_index(
    vectorBucketName="spdo-embeddings",
    indexName="insumos-sbert-384",
    dataType="float32",
    dimension=384,
    distanceMetric="cosine",
)
```

> ⚠️ S3 Vectors é serviço novo; confirme os nomes exatos dos parâmetros na sua
> versão do `awscli`/`boto3` (`aws s3vectors create-index help`). A semântica
> (bucket, index, dimension, float32, cosine) é estável.

---

## 2. Verificar
```bash
aws s3vectors list-indexes --vector-bucket-name spdo-embeddings --region us-east-1
aws s3vectors get-index --vector-bucket-name spdo-embeddings --index-name insumos-sbert-384 --region us-east-1
```

---

## 3. Operações usadas pelo app (fase de código)

Substituem `snowflake_io.insert_embeddings_novos` / `ler_*_embeddings` /
`similarity.treinar_knn` + consulta.

### Gravar vetores novos (`put_vectors`) — em lotes
```python
s3v.put_vectors(
    vectorBucketName="spdo-embeddings",
    indexName="insumos-sbert-384",
    vectors=[
        {
            "key": str(cd_insumo),
            "data": {"float32": vetor.tolist()},   # 384 floats
            # metadata opcional p/ filtros futuros (marca/medida/grp):
            # "metadata": {"grp_insumo": grp, "marca": marca},
        }
        for cd_insumo, vetor in lote
    ],
)
```
Depois do put, registre no Athena (manifesto) para o diff incremental:
`INSERT INTO db_spdo_apps.tbl_insumos_embeddings (cd_insumo, model_name, embedding_dim, vector_key, created_at) ...`

### Buscar por similaridade (`query_vectors`) — substitui o KNN
```python
resp = s3v.query_vectors(
    vectorBucketName="spdo-embeddings",
    indexName="insumos-sbert-384",
    queryVector={"float32": emb_query.tolist()},
    topK=30,                 # ver limite abaixo
    returnDistance=True,
    returnMetadata=False,
)
# resp["vectors"] -> [{"key": "<cd_insumo>", "distance": <cosine_distance>}, ...]
```
Fluxo da consulta passa a ser:
1. encode da query com SBERT (no container, modelo já embutido na imagem);
2. `query_vectors` → top-K `cd_insumo` + distância cosseno;
3. buscar os campos desses `cd_insumo` em `tbl_insumos_preprocessados`/`padronizados`
   (Athena ou cache) e aplicar a **penalização v2** sobre os candidatos.

> ⚠️ **Limite de topK**: o S3 Vectors limita o `topK` por consulta (verifique o
> teto atual; historicamente 30). O slider "Vizinhos (k)" do app vai de 5 a 50 —
> ajuste o máximo para o teto do serviço, ou faça oversampling até o limite e
> reordene após a penalização.

### Outras
- `s3v.get_vectors(...)` — recuperar vetores por chave (raro; útil p/ debug).
- `s3v.delete_vectors(...)` — remover (se um insumo sair da base).
- `s3v.list_vectors(...)` — listar chaves (paginado; prefira o manifesto no
  Athena para o diff, é mais barato que varrer o índice).
