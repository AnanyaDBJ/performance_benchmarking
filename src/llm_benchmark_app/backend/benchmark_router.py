"""
API routes for LLM benchmark operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated

from fastapi import Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .benchmark.engine import (
    BenchmarkConfig,
    RunStatus,
    cancel_run,
    create_run,
    get_run,
    list_runs,
    start_run,
)
from .benchmark.report import generate_report_bytes
from .benchmark.stats import generate_csv_bytes, generate_json_bytes, generate_summary
from .core import Dependency, create_router
from .models import (
    BenchmarkConfigIn,
    BenchmarkResultItem,
    BenchmarkResultsOut,
    BenchmarkRunOut,
    EndpointOut,
)

logger = logging.getLogger(__name__)

benchmark_router = create_router()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_obo_token(
    token: Annotated[str | None, Header(alias="X-Forwarded-Access-Token")] = None,
) -> str:
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Forwarded-Access-Token header for user authentication",
        )
    return token


def _run_to_out(run) -> BenchmarkRunOut:
    return BenchmarkRunOut(
        id=run.id,
        status=run.status.value,
        progress=run.progress,
        progress_message=run.progress_message,
        created_at=run.created_at,
        completed_at=run.completed_at,
        error=run.error,
        endpoint_names=run.config.endpoint_names if run.config else [],
        result_count=len(run.results),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/endpoints",
    response_model=list[EndpointOut],
    operation_id="listEndpoints",
)
def list_endpoints(user_ws: Dependency.UserClient):
    """List available LLM serving endpoints in the workspace."""
    try:
        endpoints = []
        for ep in user_ws.serving_endpoints.list():
            model_name = None
            model_type = None
            task = None

            if ep.config and ep.config.served_entities:
                entity = ep.config.served_entities[0]
                model_name = getattr(entity, "name", None) or getattr(
                    entity, "entity_name", None
                )
                model_type = getattr(entity, "entity_version", None)

            if ep.task:
                task = str(ep.task)

            state_str = ""
            if ep.state:
                state_str = str(ep.state.ready) if ep.state.ready else str(ep.state.config_update) if ep.state.config_update else "UNKNOWN"

            endpoints.append(
                EndpointOut(
                    name=ep.name or "",
                    state=state_str,
                    model_name=model_name,
                    model_type=model_type,
                    creator=getattr(ep, "creator", None),
                    task=task,
                )
            )
        return endpoints
    except Exception as exc:
        logger.exception("Failed to list serving endpoints")
        raise HTTPException(status_code=502, detail=f"Failed to list endpoints: {exc}")


# ---------------------------------------------------------------------------
# Benchmark CRUD
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/benchmarks",
    response_model=list[BenchmarkRunOut],
    operation_id="listBenchmarks",
)
def list_benchmark_runs():
    """List recent benchmark runs."""
    return [_run_to_out(r) for r in list_runs()]


@benchmark_router.post(
    "/benchmarks",
    response_model=BenchmarkRunOut,
    operation_id="startBenchmark",
)
async def start_benchmark(
    body: BenchmarkConfigIn,
    ws: Dependency.Client,
    token: Annotated[str | None, Header(alias="X-Forwarded-Access-Token")] = None,
):
    """Start a new benchmark run."""
    obo_token = _get_obo_token(token)

    workspace_host = str(ws.config.host or "")
    if not workspace_host:
        raise HTTPException(status_code=500, detail="Cannot determine workspace host")

    config = BenchmarkConfig(
        endpoint_names=body.endpoint_names,
        workspace_host=workspace_host,
        api_token=obo_token,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        qps_list=body.qps_list,
        parallel_workers=body.parallel_workers,
        requests_per_worker=body.requests_per_worker,
        timeout=body.timeout,
        max_retries=body.max_retries,
    )

    run = create_run(config)
    start_run(run)
    return _run_to_out(run)


@benchmark_router.get(
    "/benchmarks/{run_id}",
    response_model=BenchmarkResultsOut,
    operation_id="getBenchmark",
)
def get_benchmark(run_id: str):
    """Get benchmark run status and results."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    summary = None
    if run.results:
        data = {**run.results_meta, "results": run.results}
        summary = generate_summary(data)

    return BenchmarkResultsOut(
        id=run.id,
        status=run.status.value,
        results=[BenchmarkResultItem(**r) for r in run.results],
        summary=summary,
        meta=run.results_meta if run.results_meta else None,
    )


@benchmark_router.post(
    "/benchmarks/{run_id}/cancel",
    response_model=BenchmarkRunOut,
    operation_id="cancelBenchmark",
)
def cancel_benchmark(run_id: str):
    """Cancel a running benchmark."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
        raise HTTPException(status_code=400, detail="Benchmark is not running")
    cancel_run(run)
    return _run_to_out(run)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/benchmarks/{run_id}/stream",
    operation_id="streamBenchmark",
)
async def stream_benchmark(run_id: str):
    """SSE stream of benchmark progress."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(run.progress_queue.get(), timeout=30)
                yield {"event": "progress", "data": json.dumps(event)}
                if event.get("type") == "done":
                    return
            except asyncio.TimeoutError:
                # Send keepalive
                yield {
                    "event": "ping",
                    "data": json.dumps(
                        {"type": "ping", "status": run.status.value, "progress": run.progress}
                    ),
                }
                if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                    return

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/benchmarks/{run_id}/pdf",
    operation_id="downloadPdf",
)
def download_pdf(run_id: str):
    """Download PDF report for a completed benchmark."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    if not run.results:
        raise HTTPException(status_code=400, detail="No results available")

    pdf_bytes = generate_report_bytes(run.results, run.results_meta)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="benchmark_report_{run_id[:8]}.pdf"'},
    )


@benchmark_router.get(
    "/benchmarks/{run_id}/csv",
    operation_id="downloadCsv",
)
def download_csv(run_id: str):
    """Download CSV summary for a completed benchmark."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    if not run.results:
        raise HTTPException(status_code=400, detail="No results available")

    data = {**run.results_meta, "results": run.results}
    csv_bytes = generate_csv_bytes(data)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="benchmark_summary_{run_id[:8]}.csv"'},
    )


@benchmark_router.get(
    "/benchmarks/{run_id}/json",
    operation_id="downloadJson",
)
def download_json(run_id: str):
    """Download raw JSON results for a completed benchmark."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    if not run.results:
        raise HTTPException(status_code=400, detail="No results available")

    data = {**run.results_meta, "results": run.results}
    json_bytes = generate_json_bytes(data)
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="benchmark_results_{run_id[:8]}.json"'},
    )
