#!/usr/bin/env bash
# Run on the VPS over SSH (after cd to admix repo or set ADMIX_DIR).
#   bash scripts/diagnose-vps-admixture.sh
# Redact secrets before sharing output. Read-only; does not restart containers.

set +e

echo "=== host / cwd ==="
hostname
pwd

echo ""
echo "=== docker ==="
docker ps -a
docker compose version 2>/dev/null || true

REPO="${ADMIX_DIR:-$HOME/admix}"
if [[ -d "$REPO" ]]; then
  echo ""
  echo "=== compose (repo: $REPO) ==="
  (cd "$REPO" && docker compose ps 2>/dev/null)
  echo ""
  (cd "$REPO" && docker compose logs --tail=80 api 2>/dev/null)
else
  echo ""
  echo "=== compose: directory not found: $REPO (set ADMIX_DIR if elsewhere) ==="
fi

API_CTN=""
if [[ -d "$REPO" && -f "$REPO/docker-compose.yml" ]]; then
  API_CTN="$(cd "$REPO" && docker compose ps -q api 2>/dev/null | head -n1)"
fi
if [[ -z "$API_CTN" ]]; then
  API_CTN="$(docker ps -q -f name=admix-api 2>/dev/null | head -n1)"
fi

echo ""
echo "=== GET /docs (expect 200) ==="
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs 2>/dev/null || echo "curl failed"

echo ""
echo "=== admixture inside API container ==="
if [[ -n "$API_CTN" ]]; then
  docker exec "$API_CTN" sh -c 'echo ADMIXTURE_EXTRA_PATH=$ADMIXTURE_EXTRA_PATH; echo ADMIXTURE_HOST_PLINK_ROOT=$ADMIXTURE_HOST_PLINK_ROOT; PATH="${ADMIXTURE_EXTRA_PATH:-/host/admixture}:$PATH" which admixture 2>/dev/null; ls -la /host/admixture 2>/dev/null; ls -la /var/admixture/plink 2>/dev/null | head' 2>/dev/null || echo "docker exec failed"
else
  echo "No running API container found (compose service 'api' or name admix-api)."
fi

echo ""
echo "=== done ==="
