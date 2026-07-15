#!/usr/bin/env bash
# One-time model provisioning for the Ollama runtime service (P0.4).
#
# The compose `ollama` service runs on the internal, no-egress
# `copilot_internal` network (see docker-compose.copilot.yml) and therefore
# CANNOT reach the model registry at runtime. This script provisions the
# model the production-correct way: it runs a standalone ollama container on
# the default (egress-capable) bridge network, bind-mounting the SAME named
# volume the compose service uses (`development-easy_ollamamodels`), pulls
# the model into that volume, then removes the standalone container. The
# runtime service never needs egress.
#
# Usage: bash scripts/pull-model.sh
set -euo pipefail

IMAGE="ollama/ollama:0.12.6@sha256:352e045b937ac29d3d9550c22fb85525f60a89e064df34c26579bee5a93b3a16"
MODEL="qwen3:4b"
VOLUME="development-easy_ollamamodels"
CONTAINER="ollama-pull"

echo "Provisioning model '${MODEL}' into volume '${VOLUME}'..."

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "Starting temporary ollama container (${CONTAINER}) with egress..."
docker run --rm --gpus all -d \
    -v "${VOLUME}:/root/.ollama" \
    --name "${CONTAINER}" \
    "${IMAGE}" serve >/dev/null

cleanup() {
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for ollama server to be ready..."
for _ in $(seq 1 30); do
    if docker exec "${CONTAINER}" ollama list >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! docker exec "${CONTAINER}" ollama list >/dev/null 2>&1; then
    echo "FAIL: ollama server did not become ready in time" >&2
    exit 1
fi

if docker exec "${CONTAINER}" ollama list | grep -qF "${MODEL}"; then
    echo "Model '${MODEL}' already present in volume; re-pulling is a cheap no-op."
fi

echo "Pulling model '${MODEL}' (this may take several minutes)..."
docker exec "${CONTAINER}" ollama pull "${MODEL}"

echo "Model list after pull:"
docker exec "${CONTAINER}" ollama list

echo "Done. Model '${MODEL}' is provisioned in volume '${VOLUME}'."
