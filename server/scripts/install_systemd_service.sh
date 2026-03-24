#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_NAME="${FLUX_SERVICE_NAME:-fluxchi-api}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_USER="${FLUX_RUN_USER:-$USER}"

sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=FluxChi Platform API
After=network.target postgresql.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
ExecStart=/usr/bin/env bash ${REPO_ROOT}/server/scripts/run_api.sh
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
EOF

pkill -f "uvicorn server.app.main:app" || true
pkill -f "${REPO_ROOT}/server/scripts/run_api.sh" || true

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager -l
