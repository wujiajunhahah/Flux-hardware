#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_load_env.sh
. "${SCRIPT_DIR}/_load_env.sh"

require_env FLUX_DATABASE_URL

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m server.scripts.migrate "$@"
