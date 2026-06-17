# Deploy — ECR + ECS

Guia para publicar a imagem no **Amazon ECR** e executar no **Amazon ECS (Fargate)**.

## Arquitetura no container

- A imagem **já contém** o modelo SBERT e os corpora NLTK (pré-baixados no build) → startup rápido, sem rede externa para o modelo.
- **Sem dados na imagem**: os dados tabulares vêm do **Athena** (`db_spdo_apps`) e os vetores do **S3 Vectors** (`spdo-embeddings`). A busca k-NN roda no S3 Vectors (não em memória), então a task precisa de pouca RAM (~2 GB).
- **Sem secrets**: o acesso à AWS é via **task role** (IAM). Nenhuma credencial vai na imagem ou na task definition.

---

## 1. Pré-requisitos (na sua máquina)

1. **Docker Desktop rodando** (`docker info` responde).
2. **AWS CLI configurada** (`aws configure`, region `us-east-1`); `aws sts get-caller-identity` deve retornar sua conta.
3. **Estrutura criada na AWS** (fase anterior): tabelas Iceberg (`deploy/athena/schema.sql`) e índice S3 Vectors (`deploy/athena/s3vectors_setup.md`), com dados semeados (`python scripts/seed_aws.py`).

---

## 2. Build & push para o ECR

```powershell
.\deploy\build_and_push.ps1 -RepoName <NOME_DO_SEU_REPO_ECR>
```
Resultado: `‹account›.dkr.ecr.us-east-1.amazonaws.com/<REPO_NAME>:latest`.

> Teste local apontando para a AWS real:
> ```powershell
> docker build -t consulta-similaridade:local .
> docker run --rm -p 8501:8501 --env-file deploy\.env `
>   -v $HOME\.aws:/root/.aws:ro consulta-similaridade:local
> # http://localhost:8501
> ```

---

## 3. Executar no ECS (Fargate)

### 3.1. IAM — duas roles

- **executionRoleArn** = `ecsTaskExecutionRole` (puxar imagem do ECR + logs CloudWatch).
- **taskRoleArn** = role da aplicação, com permissão para:
  - **Athena**: `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetQueryResults`, `athena:StopQueryExecution`
  - **Glue**: `glue:GetDatabase`, `glue:GetTable`, `glue:GetTables`, `glue:GetPartitions`, `glue:CreateTable`, `glue:UpdateTable`, `glue:BatchCreatePartition`
  - **S3** (dados + resultados Athena): `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` no bucket `ibre-spdo-coleta-bronze` (prefixo `apps/consulta_similaridade/*`)
  - **S3 Vectors**: `s3vectors:QueryVectors`, `s3vectors:PutVectors`, `s3vectors:GetIndex` (e `s3vectors:CreateIndex` se for criar pela app/seed)

### 3.2. Task definition

Use `deploy/ecs-task-definition.json`, substitua `<ACCOUNT_ID>` e `<REPO_NAME>`, e registre:
```powershell
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json --region us-east-1
```

### 3.3. Service + rede

- Service Fargate em subnets **com saída para internet** (NAT ou subnet pública com IP público) — necessário para Athena/S3/S3 Vectors e (no 1º start) cache do modelo já embutido.
- **Security group**: porta **8501** (de um ALB ou do seu IP).
- Recomendado: **ALB** com target group na porta `8501`, health check path `/_stcore/health`.

---

## 4. Configuração (variáveis de ambiente)

Definidas na task definition (e como default no Dockerfile):

| Variável | Valor |
|---|---|
| `AWS_REGION` | `us-east-1` |
| `ATHENA_DATABASE` | `db_spdo_apps` |
| `S3_DATA_BASE` | `s3://ibre-spdo-coleta-bronze/apps/consulta_similaridade` |
| `S3_VECTORS_BUCKET` | `spdo-embeddings` |
| `S3_VECTORS_INDEX` | `insumos-sbert-384` |

---

## 5. Troubleshooting

| Sintoma | Causa provável | Ação |
|---|---|---|
| Container reinicia em loop | Healthcheck antes do app subir | Aumente `startPeriod` (modelo carrega em ~10–30s). |
| `AccessDenied` em Athena/Glue/S3/S3 Vectors | Permissão faltando na task role | Revise o IAM da `taskRoleArn` (seção 3.1). |
| Consulta retorna vazio | Índice S3 Vectors vazio ou DB sem dados | Rode `scripts/seed_aws.py`; confira `tbl_*` no Athena. |
| Erro de `topK` no S3 Vectors | `top_k` acima do teto do serviço | Reduza o slider "Vizinhos (k)" / `DEFAULT_TOP_K`. |
| Lento ao abrir | Carga inicial dos metadados do Athena | Esperado uma vez por task (cacheado depois). |
