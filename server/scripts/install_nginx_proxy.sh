#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_load_env.sh
. "${SCRIPT_DIR}/_load_env.sh"

PORT="${FLUX_PORT:-8000}"
CONF_PATH="/etc/nginx/conf.d/fluxchi-api.conf"

sudo dnf install -y nginx

sudo tee "${CONF_PATH}" >/dev/null <<EOF
server {
    listen 80;
    server_name _;
    client_max_body_size 10m;

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce)" = "Enforcing" ]; then
    sudo setsebool -P httpd_can_network_connect 1
fi

sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl restart nginx
sudo systemctl status nginx --no-pager -l
