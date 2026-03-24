#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_load_env.sh
. "${SCRIPT_DIR}/_load_env.sh"

require_env FLUX_DATABASE_URL
require_env FLUX_SECRET_KEY

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m uvicorn server.app.main:app \
  --host "${FLUX_BIND_HOST:-127.0.0.1}" \
  --port "${FLUX_PORT:-8000}" \
  --proxy-headers
