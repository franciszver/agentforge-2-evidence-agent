#!/usr/bin/env bash
# Container-boot smoke test for the copilot-agent service (board issue #93).
#
# Builds the production image (the same `pip install .` path the Dockerfile
# uses, i.e. no dev extras) and verifies the container actually boots and
# serves /health, rather than crashing on a missing runtime dependency.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
agent_dir="${script_dir}/../services/copilot-agent"
image="copilot-agent:smoke"
container="copilot-agent-smoke"
port=8098

cleanup() {
    docker stop "${container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "${image}" "${agent_dir}"

docker run -d --rm --name "${container}" -p "${port}:8000" "${image}" >/dev/null

echo "Waiting for /health to return 200..."
for _ in $(seq 1 20); do
    if status="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${port}/health" 2>/dev/null)" && [[ "${status}" == "200" ]]; then
        echo "AGENT BOOT SMOKE OK"
        exit 0
    fi
    sleep 1
done

echo "FAIL: /health never returned 200 within 20s" >&2
echo "Container logs:" >&2
docker logs "${container}" >&2 || true
exit 1
