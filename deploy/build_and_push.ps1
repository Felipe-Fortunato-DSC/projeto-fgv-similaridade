<#
.SYNOPSIS
    Builda a imagem Docker e faz push para o Amazon ECR.

.DESCRIPTION
    Faz login no ECR, builda a imagem a partir do Dockerfile na raiz do
    projeto, aplica as tags e envia para o repositório informado.

.PARAMETER RepoName
    Nome do repositório no ECR (ex.: "fgv-similaridade"). Obrigatório.

.PARAMETER Region
    Região AWS. Default: us-east-1.

.PARAMETER Tag
    Tag da imagem. Default: latest. (Também aplica uma tag com o short SHA do git.)

.PARAMETER AccountId
    AWS Account ID. Se omitido, é descoberto via `aws sts get-caller-identity`.

.EXAMPLE
    .\deploy\build_and_push.ps1 -RepoName fgv-similaridade

.EXAMPLE
    .\deploy\build_and_push.ps1 -RepoName fgv-similaridade -Tag v2 -Region us-east-1
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoName,

    [string]$Region = "us-east-1",

    [string]$Tag = "latest",

    [string]$AccountId
)

$ErrorActionPreference = "Stop"

# Raiz do projeto = pasta-pai deste script.
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==> Verificando pré-requisitos..." -ForegroundColor Cyan
docker info *> $null
if (-not $?) { throw "Docker não está rodando. Inicie o Docker Desktop e tente de novo." }

if (-not $AccountId) {
    Write-Host "==> Descobrindo AWS Account ID..." -ForegroundColor Cyan
    $AccountId = (aws sts get-caller-identity --query Account --output text)
    if (-not $?) { throw "Falha ao obter Account ID. Rode 'aws configure' antes." }
}

$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageBase = "$Registry/$RepoName"

# Short SHA do git (se disponível) como tag adicional de rastreabilidade.
$GitSha = (git -C $ProjectRoot rev-parse --short HEAD 2>$null)

Write-Host ""
Write-Host "  Account : $AccountId"
Write-Host "  Region  : $Region"
Write-Host "  Repo    : $RepoName"
Write-Host "  Image   : ${ImageBase}:$Tag"
if ($GitSha) { Write-Host "  GitTag  : ${ImageBase}:$GitSha" }
Write-Host ""

Write-Host "==> Login no ECR..." -ForegroundColor Cyan
$pw = (aws ecr get-login-password --region $Region)
if (-not $?) { throw "Falha no 'aws ecr get-login-password'." }
$pw | docker login --username AWS --password-stdin $Registry
if (-not $?) { throw "Falha no 'docker login' no ECR." }

Write-Host "==> Build da imagem (pode levar alguns minutos no 1º build)..." -ForegroundColor Cyan
docker build -t "${ImageBase}:$Tag" $ProjectRoot
if (-not $?) { throw "Falha no docker build." }

if ($GitSha) {
    docker tag "${ImageBase}:$Tag" "${ImageBase}:$GitSha"
}

Write-Host "==> Push para o ECR..." -ForegroundColor Cyan
docker push "${ImageBase}:$Tag"
if (-not $?) { throw "Falha no docker push." }
if ($GitSha) {
    docker push "${ImageBase}:$GitSha"
}

Write-Host ""
Write-Host "==> Concluído!" -ForegroundColor Green
Write-Host "Imagem disponível em: ${ImageBase}:$Tag"
