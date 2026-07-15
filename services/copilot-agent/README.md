# copilot-agent

FastAPI service for the Clinical Co-Pilot agent.

## Tests

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

## OpenEMR OAuth (dev token flow)

`app/openemr_auth.py` registers a confidential OAuth2 client and obtains a
user bearer token so the agent can call OpenEMR APIs. To verify end-to-end
against a running dev stack:

```bash
bash scripts/verify-oauth-dev.sh
```

This registers a confidential client, enables it, fetches a token via the
password grant, and calls `GET /apis/default/fhir/Patient` (expects HTTP 200;
an empty Bundle is fine — demo data is seeded in P2.0). It also proves the
bad-credential path fails cleanly.

**DEV-ONLY.** Two shortcuts here are for the local dev loop only and must not
ship to production:

- **Password grant** (username/password → token). Production uses the OAuth2
  `authorization_code` grant (per plan §4.2).
- **Enabling the client via direct SQL.** OpenEMR registers new clients
  *disabled*; production enables them through admin UI/approval.

TLS verification is off because the dev stack uses a self-signed certificate
(`openemr_verify_ssl` defaults to `False`; set it `True` where a real cert is
enforced). The dev client secret and tokens are written only to the gitignored
`services/copilot-agent/.openemr-dev-client.json` and are never printed or
committed.

## Container

```bash
docker build -t copilot-agent:dev .
docker run -d --rm -p 8099:8000 --name copilot-agent-test copilot-agent:dev
curl http://localhost:8099/health
docker stop copilot-agent-test
```
