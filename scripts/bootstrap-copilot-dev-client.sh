#!/usr/bin/env bash
# DEV-ONLY: one-time bootstrap of the Clinical Co-Pilot dev-token bridge
# (issue #126, finding F4).
#
# The runtime browser path cannot fetch clinical data because the browser's
# DevAgentToken is an HMAC identity assertion, not a real OpenEMR token. This
# bootstrap provisions the confidential OAuth2 client the AGENT uses to obtain
# a REAL OpenEMR token server-side (via the dev password grant) for its tool
# calls. The real token never reaches the browser.
#
# What it does (mirrors scripts/verify-oauth-dev.sh's register+enable dance):
#   1. Registers a confidential client scoped for the resource reads, run
#      INSIDE the agent container -- only the agent can reach OpenEMR on the
#      internal `copilot_internal` network. The creds are written to the
#      container-local file the RUNNING agent reads
#      (Settings.copilot_dev_client_creds_path, default /data/openemr-dev-client.json);
#      the client_secret is never printed.
#   2. Enables the client via the dev SQL shortcut (OpenEMR registers new
#      clients disabled), the same admin shortcut verify-oauth-dev.sh uses.
#
# No container recreation: the creds land in the running agent's own
# filesystem, and the bridge reads them lazily on the next token fetch.
#
# DEV-ONLY, do NOT ship: the password grant, the demo clinician credential,
# and enabling the client via direct SQL are all dev-loop shortcuts.
# Production uses the OAuth2 authorization_code grant (#124).
#
# Prerequisite: the Co-Pilot dev stack is up (agent + openemr + mysql).
set -euo pipefail

agent_container="${AGENT_CONTAINER:-development-easy-agent-1}"
mysql_container="${MYSQL_CONTAINER:-development-easy-mysql-1}"
mysql_db="${MYSQL_DB:-openemr}"
mysql_user="${MYSQL_USER:-openemr}"
mysql_pass="${MYSQL_PASS:-openemr}"

echo "== 1. Register confidential client (inside agent, writes creds file) =="
# The register CLI prints only "CLIENT_ID=<id>" (not a secret); the secret is
# written solely to the container-local creds file.
register_output="$(docker exec -i "${agent_container}" python -m app.dev_token_bridge register)"
client_id="$(printf '%s\n' "${register_output}" | sed -n 's/^CLIENT_ID=//p')"
if [[ -z "${client_id}" ]]; then
    echo "ERROR: client registration did not return a CLIENT_ID" >&2
    echo "${register_output}" >&2
    exit 1
fi
echo "registered client: ${client_id}"

echo
echo "== 2. Enable client (OpenEMR registers new clients disabled) =="
# Resolve the MariaDB client in-container; MSYS_NO_PATHCONV stops Git Bash from
# mangling the SQL on the way in (same guard as verify-oauth-dev.sh).
MSYS_NO_PATHCONV=1 docker exec -i "${mysql_container}" sh -c \
    "\$(command -v mariadb || command -v mysql) -u${mysql_user} -p${mysql_pass} ${mysql_db} -e \"UPDATE oauth_clients SET is_enabled=1 WHERE client_id='${client_id}';\"" 2>/dev/null
echo "enabled client: ${client_id}"

echo
echo "COPILOT DEV-TOKEN BRIDGE BOOTSTRAP OK"
