#!/usr/bin/env bash
# DEV-ONLY: live end-to-end verification of the OpenEMR OAuth dev token flow (P0.5).
#
# Proves, against a RUNNING dev stack, that the agent can:
#   1. register a confidential OAuth2 client (dynamic registration),
#   2. have that client enabled (OpenEMR registers clients DISABLED),
#   3. obtain a USER bearer token via the password grant (dev shortcut), and
#   4. make an authenticated OpenEMR API call (GET fhir/Patient -> 200).
# It also proves the bad-credential path fails cleanly with no secret leak.
#
# DEV-ONLY, do NOT use in production:
#   * The password grant is a dev-loop shortcut. Production uses the
#     authorization_code grant against an admin-enabled client (plan §4.2).
#   * Enabling the client via direct SQL is an admin shortcut for the dev DB.
#   * TLS verification is off because the dev stack uses a self-signed cert.
#
# Secrets: the client secret + tokens are written ONLY to the gitignored file
# services/copilot-agent/.openemr-dev-client.json and are NEVER printed.
#
# Prerequisites: the dev stack is up (openemr + mysql), and the copilot-agent
# venv exists with the package installed (see services/copilot-agent/README.md).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
py="${repo_root}/services/copilot-agent/.venv/Scripts/python.exe"
[[ -x "${py}" ]] || py="${repo_root}/services/copilot-agent/.venv/bin/python"

mysql_container="${MYSQL_CONTAINER:-development-easy-mysql-1}"
mysql_db="${MYSQL_DB:-openemr}"
mysql_user="${MYSQL_USER:-openemr}"
mysql_pass="${MYSQL_PASS:-openemr}"

run_py() { PYTHONPATH="${repo_root}/services/copilot-agent" "${py}" "${script_dir}/verify_oauth_dev.py" "$@"; }

echo "== 1. Register confidential client =="
run_py register

client_id="$(run_py client-id)"
echo
echo "== 2. Enable client (OpenEMR registers new clients disabled) =="
# Run the enable entirely inside the container via `sh -c`: the DB container
# ships the MariaDB client (`mariadb`; older images use `mysql`), and resolving
# it in-container avoids host-shell (Git Bash) absolute-path translation.
# MSYS_NO_PATHCONV keeps Git Bash from mangling the SQL/args on the way in.
MSYS_NO_PATHCONV=1 docker exec -i "${mysql_container}" sh -c \
    "\$(command -v mariadb || command -v mysql) -u${mysql_user} -p${mysql_pass} ${mysql_db} -e \"UPDATE oauth_clients SET is_enabled=1 WHERE client_id='${client_id}';\"" 2>/dev/null
echo "client ${client_id} enabled"

echo
echo "== 3-4. Fetch user token (password grant) + authenticated API call =="
run_py token-and-call

echo
echo "== 5. Bad-credential path (must fail cleanly, no secret leak) =="
run_py bad-password

echo
echo "OAUTH DEV VERIFY OK"
