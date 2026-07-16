"""FastAPI application entry point for the Clinical Co-Pilot agent service."""

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from app.chat import chat_endpoint
from app.correlation import CorrelationIdMiddleware, configure_logging
from app.feedback import feedback_endpoint
from app.readiness import ReadinessReport, compute_readiness

configure_logging()

CHAT_SHELL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Co-Pilot</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    font-family: system-ui, sans-serif;
    background: #f5f5f5;
    color: #1a1a1a;
  }
  header {
    padding: 0.75rem 1rem;
    background: #0b5a8a;
    color: #fff;
  }
  header h1 {
    margin: 0;
    font-size: 1.1rem;
  }
  #chat-stream {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
  }
  #chat-form {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem;
    border-top: 1px solid #ddd;
    background: #fff;
  }
  #chat-input {
    flex: 1;
    min-width: 0;
    padding: 0.6rem;
    font-size: 1rem;
    resize: none;
  }
  #chat-send {
    padding: 0.6rem 1rem;
    font-size: 1rem;
    background: #0b5a8a;
    color: #fff;
    border: none;
  }

  @media (min-width: 768px) {
    body {
      align-items: center;
    }
    header, #chat-stream, #chat-form {
      width: 100%;
      max-width: 720px;
    }
  }
</style>
</head>
<body>
<header>
<h1>Clinical Co-Pilot</h1>
</header>
<main id="chat-stream" data-testid="chat-stream"></main>
<form id="chat-form">
<textarea id="chat-input" data-testid="chat-input" rows="1"></textarea>
<button id="chat-send" data-testid="chat-send" type="submit">Send</button>
</form>
</body>
</html>
"""


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(title="copilot-agent")
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(report: ReadinessReport = Depends(compute_readiness)) -> JSONResponse:
        body = {
            "status": "ready" if report.ready else "not_ready",
            "checks": {
                name: {"ok": result.ok, "detail": result.detail}
                for name, result in report.checks.items()
            },
        }
        return JSONResponse(status_code=200 if report.ready else 503, content=body)

    @app.get("/chat")
    def chat_shell() -> HTMLResponse:
        return HTMLResponse(content=CHAT_SHELL_HTML)

    app.add_api_route("/chat", chat_endpoint, methods=["POST"])
    app.add_api_route("/feedback", feedback_endpoint, methods=["POST"], status_code=201)

    return app


app = create_app()
