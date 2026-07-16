"""Capacity load-test harness for the SSE ``POST /chat`` endpoint (P5.1/#60).

Sends concurrent chat requests, fully consumes each SSE stream to the
``done`` frame, and reports latency/throughput/verdict statistics. Does NOT
run against a live agent by itself when imported -- the summary math lives
in the pure, unit-tested ``summarize()`` function; only ``main()`` touches
the network.

Run inside the agent container (stdlib + ``httpx``, no other deps):

    docker cp services/copilot-agent/scripts/capacity_run.py development-easy-agent-1:/tmp/ \\
        && docker exec development-easy-agent-1 python /tmp/capacity_run.py --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass

import httpx

_DEFAULT_URL = "http://localhost:8000/chat"
_DEFAULT_BEARER = "capacitytoken"  # arbitrary placeholder; dev stub validator accepts any non-empty bearer
_DEFAULT_MESSAGE = "What medications is this patient taking?"
_DONE_EVENT = "done"
_VERIFICATION_EVENT = "verification"


@dataclass(frozen=True)
class RequestResult:
    """Outcome of one ``/chat`` SSE call.

    ``start_ts``/``end_ts`` are ``time.monotonic()`` readings spanning the
    request send to its terminal event (a ``done`` frame, an HTTP error, an
    exception, or a timeout) -- always recorded, even on failure, so slow
    failures show up in the latency distribution rather than being dropped.
    """

    start_ts: float
    end_ts: float
    status_code: int | None
    done_received: bool
    verdict: str | None
    error: str | None

    @property
    def latency_s(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def ok(self) -> bool:
        return self.done_received and self.error is None


@dataclass(frozen=True)
class RunSummary:
    """Aggregated statistics over a batch of ``RequestResult``s."""

    count: int
    successes: int
    failures: int
    error_breakdown: dict[str, int]
    p50_latency_s: float
    p95_latency_s: float
    max_latency_s: float
    throughput_rps: float
    verdict_histogram: dict[str, int]


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile (matches numpy's default method).

    ``sorted_values`` must already be sorted ascending. Returns 0.0 for an
    empty input.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


_STABLE_ERROR_LABELS = ("timeout", "http_error", "unexpected_error")


def _error_category(result: RequestResult) -> str:
    """Bucket a failed ``RequestResult`` into a short, stable category label.

    Exception-derived errors (see ``_send_one``) carry a stable label
    followed by request-varying exception text, e.g. ``"timeout: <exc>"``.
    Only the label is used as the aggregation key so failures of the same
    kind land in one bucket; the full text stays on ``RequestResult.error``
    for per-request detail.
    """
    if result.error:
        label = result.error.split(":", 1)[0]
        if label in _STABLE_ERROR_LABELS:
            return label
        return result.error
    if result.status_code is not None and result.status_code != 200:
        return f"http_{result.status_code}"
    if not result.done_received:
        return "no_done_frame"
    return "unknown"


def summarize(results: list[RequestResult]) -> RunSummary:
    """Pure aggregation over a batch of request results. No I/O.

    Throughput is successes divided by the whole run's wall-clock span
    (earliest ``start_ts`` to latest ``end_ts`` across all results), so it
    reflects real concurrent throughput rather than summed per-request time.
    """
    count = len(results)
    if count == 0:
        return RunSummary(
            count=0,
            successes=0,
            failures=0,
            error_breakdown={},
            p50_latency_s=0.0,
            p95_latency_s=0.0,
            max_latency_s=0.0,
            throughput_rps=0.0,
            verdict_histogram={},
        )

    successes = sum(1 for r in results if r.ok)
    failures = count - successes

    error_breakdown: dict[str, int] = {}
    for r in results:
        if not r.ok:
            category = _error_category(r)
            error_breakdown[category] = error_breakdown.get(category, 0) + 1

    verdict_histogram: dict[str, int] = {}
    for r in results:
        if r.verdict is not None:
            verdict_histogram[r.verdict] = verdict_histogram.get(r.verdict, 0) + 1

    latencies = sorted(r.latency_s for r in results)
    total_wall_time_s = max(r.end_ts for r in results) - min(r.start_ts for r in results)
    throughput_rps = successes / total_wall_time_s if total_wall_time_s > 0 else 0.0

    return RunSummary(
        count=count,
        successes=successes,
        failures=failures,
        error_breakdown=error_breakdown,
        p50_latency_s=_percentile(latencies, 0.50),
        p95_latency_s=_percentile(latencies, 0.95),
        max_latency_s=latencies[-1],
        throughput_rps=throughput_rps,
        verdict_histogram=verdict_histogram,
    )


def _parse_sse_stream(lines: list[str]) -> tuple[bool, str | None]:
    """Parse raw SSE lines into (``done`` seen, ``verification`` verdict)."""
    done_received = False
    verdict: str | None = None
    event_name = ""
    data_lines: list[str] = []

    def _flush() -> None:
        nonlocal verdict, done_received
        if event_name == _DONE_EVENT:
            done_received = True
        elif event_name == _VERIFICATION_EVENT and data_lines:
            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                return
            if isinstance(payload, dict):
                seen_verdict = payload.get("verdict")
                if isinstance(seen_verdict, str):
                    verdict = seen_verdict

    for line in lines:
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "":
            _flush()
            event_name = ""
            data_lines = []
    _flush()  # trailing block with no final blank line

    return done_received, verdict


async def _send_one(
    client: httpx.AsyncClient,
    *,
    url: str,
    token: str,
    patient_id: int,
    message: str,
    timeout: float,
) -> RequestResult:
    start_ts = time.monotonic()
    status_code: int | None = None
    done_received = False
    verdict: str | None = None
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            url,
            json={"message": message, "patient_id": patient_id},
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            lines = [line async for line in response.aiter_lines()]
            done_received, verdict = _parse_sse_stream(lines)
            if status_code >= 400:
                error = f"http_{status_code}"
    except httpx.TimeoutException as exc:
        error = f"timeout: {exc}"
    except httpx.HTTPError as exc:
        error = f"http_error: {exc}"
    except Exception as exc:  # noqa: BLE001 - one bad request must not abort the run
        error = f"unexpected_error: {exc}"

    end_ts = time.monotonic()
    return RequestResult(
        start_ts=start_ts,
        end_ts=end_ts,
        status_code=status_code,
        done_received=done_received,
        verdict=verdict,
        error=error,
    )


async def _run_load(args: argparse.Namespace) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:

        async def _bounded() -> RequestResult:
            async with semaphore:
                return await _send_one(
                    client,
                    url=args.url,
                    token=args.token,
                    patient_id=args.patient_id,
                    message=args.message,
                    timeout=args.timeout,
                )

        tasks = [asyncio.create_task(_bounded()) for _ in range(args.requests)]
        return await asyncio.gather(*tasks)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity load test for POST /chat (SSE).")
    parser.add_argument(
        "--concurrency", type=int, required=True, help="number of simultaneous in-flight chat requests"
    )
    parser.add_argument(
        "--requests", type=int, default=None, help="total requests to send (default: --concurrency, one wave)"
    )
    parser.add_argument("--url", type=str, default=_DEFAULT_URL)
    parser.add_argument("--token", type=str, default=_DEFAULT_BEARER)
    parser.add_argument("--patient-id", type=int, default=1)
    parser.add_argument("--message", type=str, default=_DEFAULT_MESSAGE)
    parser.add_argument("--timeout", type=float, default=180.0)

    args = parser.parse_args(argv)
    if args.requests is None:
        args.requests = args.concurrency
    return args


def _print_summary(summary: RunSummary) -> None:
    print("Capacity run summary")
    print(f"  requests:    {summary.count}")
    print(f"  successes:   {summary.successes}")
    print(f"  failures:    {summary.failures}")
    print(f"  errors:      {summary.error_breakdown}")
    print(f"  p50 latency: {summary.p50_latency_s:.3f}s")
    print(f"  p95 latency: {summary.p95_latency_s:.3f}s")
    print(f"  max latency: {summary.max_latency_s:.3f}s")
    print(f"  throughput:  {summary.throughput_rps:.3f} req/s")
    print(f"  verdicts:    {summary.verdict_histogram}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = asyncio.run(_run_load(args))
    summary = summarize(results)
    _print_summary(summary)
    print("SUMMARY_JSON: " + json.dumps(asdict(summary)))


if __name__ == "__main__":
    main()
