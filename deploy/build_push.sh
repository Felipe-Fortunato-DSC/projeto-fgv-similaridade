#!/usr/bin/env bash
# Build + push da imagem para o ECR spdo-apps.
# Rodar DENTRO do WSL, a partir da raiz do projeto:
#   bash deploy/build_push.sh
#
# Usa o aws.exe do Windows (SSO profile fgv-dev) via interop — não precisa
# configurar credenciais AWS dentro do Ubuntu.
set -euo pipefail

ACCOUNT_ID="753771550345"
REGION="us-east-1"
REPO="spdo-apps"
TAG="app-consulta-similaridade-v0.0.1"
AWS_PROFILE="fgv-dev"

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

# aws.exe (Windows) se disponível via interop; senão aws nativo do WSL.
if command -v aws.exe >/dev/null 2>&1; then
  AWS=aws.exe
else
  AWS=aws
fi

cd "$(dirname "$0")/.."

echo ">> Login no ECR (${REGISTRY}) via ${AWS} [profile ${AWS_PROFILE}]"
"$AWS" ecr get-login-password --region "$REGION" --profile "$AWS_PROFILE" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo ">> Build da imagem ${IMAGE}"
docker build -t "$IMAGE" .

echo ">> Push ${IMAGE}"
docker push "$IMAGE"

echo ">> OK. Imagem publicada:"
echo "   $IMAGE"
