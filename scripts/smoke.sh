#!/usr/bin/env bash
# Smoke test for the Co-Pilot docker-compose overlay (docker-compose.copilot.yml).
#
# Verifies, via `docker compose config` (no containers started), that:
#   - the base compose file + copilot overlay merge cleanly
#   - the `agent` and `ollama` services are declared
#   - the copilot network is `internal: true` (the no-egress guarantee)
#   - a named volume for persisting ollama models is declared
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_dir="${script_dir}/../docker/development-easy"

cd "${compose_dir}"

if ! rendered="$(docker compose -f docker-compose.yml -f docker-compose.copilot.yml config)"; then
    echo "FAIL: docker compose config failed to render base + copilot overlay" >&2
    exit 1
fi

services="$(docker compose -f docker-compose.yml -f docker-compose.copilot.yml config --services)"

if ! grep -qx "agent" <<< "${services}"; then
    echo "FAIL: service 'agent' not present in rendered config" >&2
    exit 1
fi

if ! grep -qx "ollama" <<< "${services}"; then
    echo "FAIL: service 'ollama' not present in rendered config" >&2
    exit 1
fi

if ! grep -q "internal: true" <<< "${rendered}"; then
    echo "FAIL: no internal (no-egress) network declared in rendered config" >&2
    exit 1
fi

if ! grep -q "ollamamodels" <<< "${rendered}"; then
    echo "FAIL: named volume for ollama models not found in rendered config" >&2
    exit 1
fi

echo "SMOKE OK"
