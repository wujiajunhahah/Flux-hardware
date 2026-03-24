#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

load_optional_env_file() {
  local env_file="$1"
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
  fi
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    printf 'missing required env: %s\n' "$name" >&2
    exit 1
  fi
}

resolve_python_bin() {
  local candidate
  for candidate in \
    "${FLUX_PYTHON_BIN:-}" \
    "${REPO_ROOT}/.venv/bin/python" \
    "${HOME}/.venvs/fluxchi/bin/python" \
    "$(command -v python3 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  printf 'python interpreter not found; set FLUX_PYTHON_BIN or create %s/.venv\n' "$REPO_ROOT" >&2
  exit 1
}

load_optional_env_file "${FLUX_DB_ENV_FILE:-${HOME}/fluxchi-db.env}"
load_optional_env_file "${FLUX_APP_ENV_FILE:-${HOME}/fluxchi-app.env}"

PYTHON_BIN="$(resolve_python_bin)"
export REPO_ROOT
export PYTHON_BIN
