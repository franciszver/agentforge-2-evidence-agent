#!/usr/bin/env bash
# One-time model provisioning for the Ollama runtime service (P0.4; extended
# for the Phase 2 vision-language + embedding models, #10).
#
# The compose `ollama` service runs on the internal, no-egress
# `copilot_internal` network (see docker-compose.copilot.yml) and therefore
# CANNOT reach the model registry at runtime. This script provisions the
# models the production-correct way: it runs a standalone ollama container on
# the default (egress-capable) bridge network, bind-mounting the SAME named
# volume the compose service uses (`development-easy_ollamamodels`), pulls
# each model into that volume, then removes the standalone container. The
# runtime service never needs egress.
#
# Digest pinning: the ollama registry's CLI does not support fetch-by-digest
# for its own model library (`ollama pull <model>@sha256:<digest>` returns
# "invalid model name" as of ollama 0.12.6/0.32.0) the way the `IMAGE`
# reference above is pinned. Instead, each Phase 2 model below is pinned by
# recording the expected manifest digest (`sha256sum` of
# `/root/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>`,
# same value `ollama list`'s ID column is derived from) and verifying it
# after pull, so a silent registry-side content change fails loudly instead
# of provisioning a different model than was reviewed/tested.
#
# Usage: bash scripts/pull-model.sh
set -euo pipefail

IMAGE="ollama/ollama:0.12.6@sha256:352e045b937ac29d3d9550c22fb85525f60a89e064df34c26579bee5a93b3a16"
VOLUME="development-easy_ollamamodels"
CONTAINER="ollama-pull"

# name|tag|manifest_path_suffix|expected_digest ("" = no pin enforced, e.g.
# the pre-Phase-2 qwen3:4b).
MODELS=(
    "qwen3:4b|qwen3/4b|"
    "qwen2.5vl:7b|qwen2.5vl/7b|5ced39dfa4bac325dc183dd1e4febaa1c46b3ea28bce48896c8e69c1e79611cc"
    "nomic-embed-text|nomic-embed-text/latest|0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
)

echo "Provisioning models into volume '${VOLUME}'..."

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

for spec in "${MODELS[@]}"; do
    IFS='|' read -r model manifest_suffix expected_digest <<< "${spec}"

    if docker exec "${CONTAINER}" ollama list | grep -qF "${model}"; then
        echo "Model '${model}' already present in volume; re-pulling is a cheap no-op."
    fi

    echo "Pulling model '${model}' (this may take several minutes)..."
    docker exec "${CONTAINER}" ollama pull "${model}"

    if [[ -n "${expected_digest}" ]]; then
        manifest_path="/root/.ollama/models/manifests/registry.ollama.ai/library/${manifest_suffix}"
        actual_digest="$(docker exec "${CONTAINER}" sh -c "sha256sum '${manifest_path}'" | awk '{print $1}')"
        if [[ "${actual_digest}" != "${expected_digest}" ]]; then
            echo "FAIL: model '${model}' manifest digest mismatch" >&2
            echo "  expected: ${expected_digest}" >&2
            echo "  actual:   ${actual_digest}" >&2
            exit 1
        fi
        echo "Digest verified for '${model}': ${actual_digest}"
    fi
done

echo "Model list after pull:"
docker exec "${CONTAINER}" ollama list

echo "Done. Models are provisioned in volume '${VOLUME}'."
