FROM python:3.11-slim@sha256:baf89808ec37adeaab83cec287adb4a2afa4a11c1d51e961c7ec737877e61af6

WORKDIR /app

COPY pyproject.toml requirements.txt ./
# requirements.txt (scripts/lock-deps.sh, issue 213) hash-pins the entire
# production runtime closure -- transitives included -- so every build from
# this commit resolves byte-identical dependencies, not just something
# inside pyproject.toml's floor/ceiling ranges.
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
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

# --no-deps: the runtime closure is already installed and hash-verified
# above; reinstalling deps here would resolve against pyproject's ranges
# again, bypassing the lock. --no-build-isolation: skips fetching a fresh,
# unpinned setuptools/wheel from PyPI for the build backend -- it reuses
# the base image's own pinned setuptools/pip (fixed by this Dockerfile's
# digest-pinned FROM line), keeping the build-backend closure locked too.
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

RUN useradd --create-home --shell /usr/sbin/nologin appuser

# /data holds client OAuth creds (copilot_dev/prod_client_creds_path) and
# the raw document-ingestion store (copilot_ingestion_base_dir); the trace
# store (trace_db_path) lives at /data/traces, a subdirectory, since #180
# gave it its own named volume in docker-compose.copilot.yml (scoped so
# the OAuth creds and ingested patient documents keep dying with the
# container -- see that compose file's comment on the `agent` service).
# Both provisioned owned by appuser before dropping root: #180's review
# probed mounting a FRESH named volume over /data/traces and confirmed
# Docker's copy-up behavior copies this directory's ownership into the
# volume on first mount, so appuser keeps write access without a
# re-chown step -- this does NOT reintroduce #111's PermissionError (a
# prior version of this comment warned that mounting anything over /data
# would; that was true for a MOUNT WITHOUT copy-up support, e.g. a bind
# mount over a nonexistent path, not for a named volume over a path the
# image already owns correctly).
RUN mkdir -p /data/traces && chown -R appuser:appuser /data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
