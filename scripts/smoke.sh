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

# Issue #180: the `agent` service's TRACE_ARGS_HASH_SECRET is a required
# (`${VAR:?...}`) compose variable -- deliberately no default, since a
# committed literal in this public repo would be a published secret (see
# docker-compose.copilot.yml's comment on it). This smoke test only renders
# config and starts no containers, so it never touches real trace data --
# any throwaway value satisfies the required-variable check without
# needing a real secret. A caller that already exported a real one (e.g.
# from .env) keeps it; this only fills the gap when nothing is set.
export TRACE_ARGS_HASH_SECRET="${TRACE_ARGS_HASH_SECRET:-smoke-test-throwaway-value}"

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
