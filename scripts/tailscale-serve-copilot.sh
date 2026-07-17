#!/usr/bin/env bash
# Expose the OpenEMR / Clinical Co-Pilot dev stack to the owner's tailnet
# (issue #16), so it's reachable from their phone without opening any port
# to the public internet.
#
# What this exposes: OpenEMR's HTTP listener (host port 8300 -> container
# :80). Tailscale terminates tailnet HTTPS in front of it and serves it at
# https://<this-machine>.<tailnet-name>.ts.net/. The Co-Pilot panel is an
# OpenEMR module served WITHIN OpenEMR, so exposing OpenEMR exposes the
# panel too -- no separate mapping needed.
#
# What this does NOT expose: the copilot-agent FastAPI service. It stays
# internal-only on the docker network; nothing here punches a hole to it,
# and none of the `tailscale serve` commands below reference its port.
#
# Prerequisites: `tailscale` installed, logged into the tailnet, and the
# tailnet's MagicDNS + HTTPS certs enabled (Tailscale admin console ->
# DNS). If HTTPS certs aren't enabled, `tailscale serve` will fail with an
# error naming that setting -- turn it on there (this script can't do it
# for you) and re-run `up`.
#
# Verified (2026-07-16, dev stack): OpenEMR's `site_addr_oath` global was
# already unset in this dev DB, so OpenEMR builds absolute URLs from the
# incoming request host dynamically rather than a baked-in address -- no
# OpenEMR config change was needed for the tailnet hostname to work, and
# localhost:8300/9300 access keeps working unchanged. The one HARDCODED
# absolute URL in the Co-Pilot module is
# `OAuthConsentConfig::CANONICAL_REDIRECT_URI` (interface/modules/
# custom_modules/oe-module-clinical-copilot/src/Auth/OAuthConsentConfig.php),
# fixed to `https://localhost:9300/...` -- but that OAuth consent redirect
# flow is gated off by default (`clinical_copilot_oauth_consent_enabled`
# unset in this dev DB; the DevAgentToken bridge is the live dev auth
# path), so it does not block phone access. If that flow is ever enabled
# for tailnet use, CANONICAL_REDIRECT_URI needs a tailnet-aware value.
#
# Usage:
#   scripts/tailscale-serve-copilot.sh up      # start/refresh the mapping
#   scripts/tailscale-serve-copilot.sh status  # show current mapping
#   scripts/tailscale-serve-copilot.sh down    # tear down (localhost still works)
set -euo pipefail

openemr_port="${OPENEMR_HOST_PORT:-8300}"

tailscale_bin="tailscale"
if ! command -v tailscale >/dev/null 2>&1; then
    # Default Windows install location; not on PATH in every shell.
    win_default="/c/Program Files/Tailscale/tailscale"
    if [[ -x "${win_default}" ]]; then
        tailscale_bin="${win_default}"
    fi
fi

cmd="${1:-}"
case "${cmd}" in
    up)
        echo "== Mapping tailnet HTTPS -> http://127.0.0.1:${openemr_port} (OpenEMR) =="
        "${tailscale_bin}" serve --bg "${openemr_port}"
        echo
        "${tailscale_bin}" serve status
        ;;
    status)
        "${tailscale_bin}" serve status
        ;;
    down)
        echo "== Removing tailnet HTTPS mapping (localhost:${openemr_port} keeps working) =="
        "${tailscale_bin}" serve --https=443 off
        ;;
    *)
        echo "Usage: $0 {up|status|down}" >&2
        exit 1
        ;;
esac
