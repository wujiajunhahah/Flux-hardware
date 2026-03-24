#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_load_env.sh
. "${SCRIPT_DIR}/_load_env.sh"

PORT="${FLUX_PORT:-8000}"

enable_proxy_selinux() {
  if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce)" = "Enforcing" ]; then
    sudo setsebool -P httpd_can_network_connect 1
  fi
}

resolve_nginx_conf_path() {
  local conf_path
  local prefix
  local main_conf

  if nginx -V 2>&1 | grep -q -- '--conf-path='; then
    main_conf="$(nginx -V 2>&1 | sed -n "s/.*--conf-path=\([^ ]*\).*/\1/p" | head -n1)"
  else
    prefix="$(nginx -V 2>&1 | sed -n "s/.*--prefix=\([^ ]*\).*/\1/p" | head -n1)"
    if [ -n "${prefix}" ]; then
      main_conf="${prefix}/conf/nginx.conf"
    fi
  fi

  if [ -n "${main_conf:-}" ] && sudo test -f "${main_conf}"; then
    conf_path="$(sudo awk '
      $1 == "include" && $2 ~ /\.conf;$/ && $2 !~ /\/tcp\// {
        gsub(/;/, "", $2)
        print $2
        exit
      }
    ' "${main_conf}")"
    if [ -n "${conf_path}" ]; then
      conf_path="${conf_path/\*/fluxchi-api}"
      printf '%s\n' "${conf_path}"
      return 0
    fi
  fi

  printf '/etc/nginx/conf.d/fluxchi-api.conf\n'
}

install_nginx_proxy() {
  local conf_path
  conf_path="$(resolve_nginx_conf_path)"

  if ! command -v nginx >/dev/null 2>&1; then
    sudo dnf install -y nginx
  fi

  sudo mkdir -p "$(dirname "${conf_path}")"
  sudo tee "${conf_path}" >/dev/null <<EOF
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

  enable_proxy_selinux
  sudo nginx -t
  sudo systemctl enable --now nginx
  sudo systemctl restart nginx
  sudo systemctl status nginx --no-pager -l
}

install_httpd_proxy() {
  local conf_path="/etc/httpd/conf.d/fluxchi-api.conf"

  sudo dnf install -y httpd

  sudo tee "${conf_path}" >/dev/null <<EOF
ProxyPreserveHost On

<VirtualHost *:80>
    ServerName _

    ProxyPass / http://127.0.0.1:${PORT}/
    ProxyPassReverse / http://127.0.0.1:${PORT}/

    RequestHeader set X-Forwarded-Proto "http"
    RequestHeader set X-Forwarded-Port "80"
</VirtualHost>
EOF

  enable_proxy_selinux
  sudo apachectl -t
  sudo systemctl enable --now httpd
  sudo systemctl restart httpd
  sudo systemctl status httpd --no-pager -l
}

if command -v nginx >/dev/null 2>&1; then
  install_nginx_proxy
elif sudo dnf list available nginx >/dev/null 2>&1; then
  install_nginx_proxy
else
  install_httpd_proxy
fi
