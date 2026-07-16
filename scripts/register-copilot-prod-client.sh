#!/usr/bin/env bash
# Production OAuth2 authorization_code client registration for the Clinical
# Co-Pilot (#124 Phase 1).
#
# Registers the confidential client the browser-driven authorization_code flow
# (Phase 2) will use. Unlike scripts/bootstrap-copilot-dev-client.sh, this
# script does NOT enable the client via a SQL shortcut -- OpenEMR registers new
# clients DISABLED, and in production an admin must approve/enable them through
# the admin UI (see the manual step printed at the end and the README).
#
# What it does:
#   1. Registers the confidential client INSIDE the agent container (only the
#      agent can reach OpenEMR on the internal `copilot_internal` network), with
#      the canonical browser-facing module OAuth callback as the redirect_uri
#      and the reconciled SMART scope set. The client_secret is written solely
#      to the container-local creds file (Settings.copilot_prod_client_creds_path,
#      default /data/openemr-prod-client.json) and is never printed.
#
# Prerequisite: the Co-Pilot stack is up (agent + openemr + mysql).
set -euo pipefail

agent_container="${AGENT_CONTAINER:-development-easy-agent-1}"

echo "== Register production confidential client (inside agent, writes creds file) =="
# The register CLI prints only "CLIENT_ID=<id>" (not a secret) plus the manual
# admin-approval reminder; the secret is written solely to the creds file.
docker exec -i "${agent_container}" python -m app.prod_client_registration

echo
echo "NEXT (manual, admin): enable the client in OpenEMR before the"
echo "authorization_code flow will issue tokens --"
echo "  Administration -> Config -> Connectors -> OAuth2 Clients -> enable the"
echo "  'copilot-agent-prod' client."
echo "(The dev SQL 'UPDATE oauth_clients SET is_enabled=1' shortcut used by"
echo " bootstrap-copilot-dev-client.sh is DEV-ONLY -- do not use it in prod.)"
