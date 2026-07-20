FROM python:3.11-slim@sha256:baf89808ec37adeaab83cec287adb4a2afa4a11c1d51e961c7ec737877e61af6

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
# app/retrieval.py's CORPUS_DIR (P3.9 guideline-corpus evidence retrieval)
# resolves to ./corpus at runtime -- without this, build_retriever_from_corpus
# parses a directory that doesn't exist in the image, silently returning zero
# chunks for every query (parse_corpus on a missing dir, no error raised).
# Found live during the P5.1 demo dry run (issue #27): with
# COPILOT_EVIDENCE_RETRIEVAL_ENABLED=true the guideline-citation path still
# never produced a citation until this line was added -- the feature was
# unreachable in every built image, not just flag-gated off.
COPY corpus ./corpus

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
