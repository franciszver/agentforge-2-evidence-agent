#!/usr/bin/env bash
# DEV-ONLY: bind the Phase 1 prod authorization_code client into the OpenEMR
# module globals the consent flow (#124 Phase 2b/6) reads.
#
# scripts/register-copilot-prod-client.sh registers the confidential client and
# writes its creds to the agent-container-local file
# (Settings.copilot_prod_client_creds_path, default /data/openemr-prod-client.json),
# but nothing yet copies those creds into the OpenEMR-side module globals that
# OAuthConsentConfig::fromEnvironment() reads. This script closes that gap for the
# dev loop so the browser consent flow is reproducible:
#
#   clinical_copilot_prod_client_id      <- creds file client_id
#   clinical_copilot_prod_client_secret  <- creds file client_secret
#   clinical_copilot_oauth_consent_enabled = 1   (turns the consent flow ON)
#   clinical_copilot_oauth_verify_ssl      = 0   (dev self-signed cert)
#
# The server-side token exchange targets the container-internal endpoint by
# default (OAuthConsentConfig::DEFAULT_INTERNAL_TOKEN_URL =
# https://openemr/oauth2/default/token); override with the
# clinical_copilot_oauth_internal_token_url global only if your internal alias
# differs.
#
# DEV-ONLY, do NOT ship: writing globals via direct SQL and disabling TLS verify
# are dev-loop shortcuts. In production an OpenEMR admin provisions the prod
# client secret out-of-band (never from a repo-checked-out creds file), enables
# the consent flag deliberately, and leaves TLS verification ON.
#
# Prerequisites:
#   1. The Co-Pilot dev stack is up (agent + openemr + mysql).
#   2. scripts/register-copilot-prod-client.sh has been run (creds file exists).
#   3. The prod client is enabled (admin UI, or the dev SQL shortcut below).
set -euo pipefail

agent_container="${AGENT_CONTAINER:-development-easy-agent-1}"
mysql_container="${MYSQL_CONTAINER:-development-easy-mysql-1}"
mysql_db="${MYSQL_DB:-openemr}"
mysql_user="${MYSQL_USER:-openemr}"
mysql_pass="${MYSQL_PASS:-openemr}"
creds_path="${PROD_CREDS_PATH:-/data/openemr-prod-client.json}"

echo "== 1. Read prod client creds from the agent container =="
creds="$(docker exec -i "${agent_container}" python -c \
    "import json; d=json.load(open('${creds_path}')); print(d['client_id']); print(d['client_secret'])")"
client_id="$(printf '%s\n' "${creds}" | sed -n '1p')"
client_secret="$(printf '%s\n' "${creds}" | sed -n '2p')"
if [[ -z "${client_id}" || -z "${client_secret}" ]]; then
    echo "ERROR: could not read client_id/client_secret from ${creds_path}" >&2
    echo "Run scripts/register-copilot-prod-client.sh first." >&2
    exit 1
fi
echo "client_id: ${client_id}"

echo
echo "== 2. Upsert module globals (creds + enable consent + dev TLS opt-out) =="
MSYS_NO_PATHCONV=1 docker exec -i "${mysql_container}" sh -c \
    "\$(command -v mariadb || command -v mysql) -u${mysql_user} -p${mysql_pass} ${mysql_db}" <<SQL 2>/dev/null
INSERT INTO globals (gl_name, gl_index, gl_value) VALUES
  ('clinical_copilot_prod_client_id', 0, '${client_id}'),
  ('clinical_copilot_prod_client_secret', 0, '${client_secret}'),
  ('clinical_copilot_oauth_consent_enabled', 0, '1'),
  ('clinical_copilot_oauth_verify_ssl', 0, '0')
ON DUPLICATE KEY UPDATE gl_value = VALUES(gl_value);
SQL
echo "globals wired."

echo
echo "COPILOT PROD-CLIENT GLOBALS WIRED OK"
echo "The consent flow is now ON. To turn it back OFF (revert to the dev-bridge"
echo "path), delete these globals or set clinical_copilot_oauth_consent_enabled=0."
