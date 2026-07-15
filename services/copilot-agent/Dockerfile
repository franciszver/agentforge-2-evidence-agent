FROM python:3.11-slim@sha256:baf89808ec37adeaab83cec287adb4a2afa4a11c1d51e961c7ec737877e61af6

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin appuser

# Trace store lives at /data (trace_db_path in app/config.py); provision it
# owned by appuser before dropping root. Mounting a volume over /data shadows
# this ownership and would reintroduce the PermissionError from #111 — re-chown
# after mount if persistence is ever added.
RUN mkdir /data && chown appuser:appuser /data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
