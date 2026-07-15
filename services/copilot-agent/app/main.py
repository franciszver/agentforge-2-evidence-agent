"""FastAPI application entry point for the Clinical Co-Pilot agent service."""

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from app.readiness import ReadinessReport, compute_readiness


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(title="copilot-agent")

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

    return app


app = create_app()
