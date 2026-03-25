#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE_PATH="${REPO_ROOT}/server/deploy/nginx-fluxchi-https.conf"
DEST_PATH="/www/server/panel/vhost/nginx/fluxchi_https.conf"

usage() {
  cat <<'EOF'
Usage:
  FLUX_SERVER_NAME=api.example.com \
  FLUX_CERT_PATH=/etc/fluxchi/ssl/api.example.com.pem \
  FLUX_KEY_PATH=/etc/fluxchi/ssl/api.example.com.key \
  bash server/scripts/install_nginx_https.sh

Required env vars:
  FLUX_SERVER_NAME  public domain name, e.g. api.focux.me
  FLUX_CERT_PATH    absolute path to certificate chain (.pem)
  FLUX_KEY_PATH     absolute path to private key (.key)

Optional env vars:
  FLUX_PORT         upstream app port, default 8000
EOF
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "missing required env var: ${name}" >&2
    usage >&2
    exit 1
  fi
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

require_env "FLUX_SERVER_NAME"
require_env "FLUX_CERT_PATH"
require_env "FLUX_KEY_PATH"

FLUX_PORT="${FLUX_PORT:-8000}"

if [ ! -f "${TEMPLATE_PATH}" ]; then
  echo "template not found: ${TEMPLATE_PATH}" >&2
  exit 1
fi

if ! sudo test -f "${FLUX_CERT_PATH}"; then
  echo "certificate file not found: ${FLUX_CERT_PATH}" >&2
  exit 1
fi

if ! sudo test -f "${FLUX_KEY_PATH}"; then
  echo "private key file not found: ${FLUX_KEY_PATH}" >&2
  exit 1
fi

tmp_conf="$(mktemp)"
trap 'rm -f "${tmp_conf}"' EXIT

sed \
  -e "s/__SERVER_NAME__/$(escape_sed "${FLUX_SERVER_NAME}")/g" \
  -e "s/__CERT_PATH__/$(escape_sed "${FLUX_CERT_PATH}")/g" \
  -e "s/__KEY_PATH__/$(escape_sed "${FLUX_KEY_PATH}")/g" \
  -e "s/__UPSTREAM_PORT__/$(escape_sed "${FLUX_PORT}")/g" \
  "${TEMPLATE_PATH}" > "${tmp_conf}"

sudo cp "${tmp_conf}" "${DEST_PATH}"
sudo chmod 644 "${DEST_PATH}"

sudo nginx -t
sudo nginx -s reload

curl -sk --resolve "${FLUX_SERVER_NAME}:443:127.0.0.1" "https://${FLUX_SERVER_NAME}/v1/health"
