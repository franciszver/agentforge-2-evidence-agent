# copilot-agent

FastAPI service for the Clinical Co-Pilot agent.

## Tests

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

## Container

```bash
docker build -t copilot-agent:dev .
docker run -d --rm -p 8099:8000 --name copilot-agent-test copilot-agent:dev
curl http://localhost:8099/health
docker stop copilot-agent-test
```
