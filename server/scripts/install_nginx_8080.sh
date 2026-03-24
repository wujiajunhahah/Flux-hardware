#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${REPO_ROOT}/server/deploy/nginx-fluxchi-proxy.conf"
DST="/www/server/panel/vhost/nginx/fluxchi_proxy.conf"

# clean up any broken previous attempts
sudo rm -f "${DST}"

# copy the static config from the repo
sudo cp "${SRC}" "${DST}"
sudo chmod 644 "${DST}"

# validate and reload
sudo nginx -t && sudo nginx -s reload

# verify
curl -s http://127.0.0.1:8080/v1/health
