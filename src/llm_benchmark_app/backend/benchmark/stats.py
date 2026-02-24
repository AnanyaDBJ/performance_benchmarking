"""
Summary statistics generation from benchmark results.

Adapted from generate_summary_stats.py to work with in-memory data and
produce CSV/JSON bytes for download.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections import defaultdict
from typing import Any


def generate_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Generate comprehensive summary statistics from benchmark data."""
    results = data.get("results", [])

    return {
        "timestamp": data.get("timestamp", "N/A"),
        "endpoints_tested": data.get("endpoints", []),
        "total_configurations": len(results),
        "by_endpoint": _by_endpoint(results),
        "by_worker_count": _by_workers(results),
        "overall_metrics": _overall(results),
        "performance_rankings": _rankings(results),
    }


def generate_csv_bytes(data: dict[str, Any]) -> bytes:
    """Generate CSV summary as bytes."""
    summary = generate_summary(data)
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow([
        "Endpoint",
        "Configurations",
        "Success Rate (%)",
        "Avg Median Latency (s)",
        "Min Median Latency (s)",
        "Max Median Latency (s)",
        "Avg P95 Latency (s)",
        "Avg Throughput (tok/s)",
        "Min Throughput (tok/s)",
        "Max Throughput (tok/s)",
        "Avg Input Tokens",
        "Avg Output Tokens",
        "Total Requests",
        "Successful Requests",
        "Failed Requests",
    ])

    def _fmt(val: float | None, fmt: str) -> str:
        return format(val, fmt) if val is not None else "N/A"

    for endpoint, m in summary["by_endpoint"].items():
        writer.writerow([
            endpoint,
            m["num_configurations"],
            f"{m['success_rate']:.2f}",
            _fmt(m["avg_median_latency"], ".4f"),
            _fmt(m["min_median_latency"], ".4f"),
            _fmt(m["max_median_latency"], ".4f"),
            _fmt(m["avg_p95_latency"], ".4f"),
            _fmt(m["avg_throughput"], ".2f"),
            _fmt(m["min_throughput"], ".2f"),
            _fmt(m["max_throughput"], ".2f"),
            _fmt(m["avg_input_tokens"], ".1f"),
            _fmt(m["avg_output_tokens"], ".1f"),
            m["total_successful_requests"] + m["total_failed_requests"],
            m["total_successful_requests"],
            m["total_failed_requests"],
        ])

    return buf.getvalue().encode("utf-8")


def generate_json_bytes(data: dict[str, Any]) -> bytes:
    """Generate summary JSON as bytes."""
    summary = generate_summary(data)
    return json.dumps(summary, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _success_rate(results: list[dict]) -> float:
    total = sum(r["total_requests"] for r in results)
    ok = sum(r["successful_requests"] for r in results)
    return (ok / total * 100) if total > 0 else 0.0


def _has_metrics(r: dict) -> bool:
    return r.get("median_latency") is not None


def _by_endpoint(results: list[dict]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["endpoint_name"]].append(r)

    out = {}
    for ep, rs in groups.items():
        ok = [r for r in rs if _has_metrics(r)]
        if ok:
            out[ep] = {
                "num_configurations": len(rs),
                "avg_median_latency": statistics.mean([r["median_latency"] for r in ok]),
                "min_median_latency": min(r["median_latency"] for r in ok),
                "max_median_latency": max(r["median_latency"] for r in ok),
                "avg_p95_latency": statistics.mean([r["p95_latency"] for r in ok]),
                "min_p95_latency": min(r["p95_latency"] for r in ok),
                "max_p95_latency": max(r["p95_latency"] for r in ok),
                "avg_throughput": statistics.mean([r["throughput"] for r in ok]),
                "min_throughput": min(r["throughput"] for r in ok),
                "max_throughput": max(r["throughput"] for r in ok),
                "total_successful_requests": sum(r["successful_requests"] for r in rs),
                "total_failed_requests": sum(r["failed_requests"] for r in rs),
                "success_rate": _success_rate(rs),
                "avg_input_tokens": statistics.mean([r["avg_input_tokens"] for r in ok]),
                "avg_output_tokens": statistics.mean([r["avg_output_tokens"] for r in ok]),
            }
        else:
            out[ep] = {
                "num_configurations": len(rs),
                "avg_median_latency": None,
                "min_median_latency": None,
                "max_median_latency": None,
                "avg_p95_latency": None,
                "min_p95_latency": None,
                "max_p95_latency": None,
                "avg_throughput": None,
                "min_throughput": None,
                "max_throughput": None,
                "total_successful_requests": 0,
                "total_failed_requests": sum(r["failed_requests"] for r in rs),
                "success_rate": 0.0,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
            }
    return out


def _by_workers(results: list[dict]) -> dict[str, Any]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["num_workers"]].append(r)

    out = {}
    for w, rs in groups.items():
        ok = [r for r in rs if _has_metrics(r)]
        entry: dict[str, Any] = {
            "num_endpoints": len(rs),
            "total_successful_requests": sum(r["successful_requests"] for r in rs),
            "total_failed_requests": sum(r["failed_requests"] for r in rs),
            "success_rate": _success_rate(rs),
        }
        if ok:
            entry["avg_median_latency"] = statistics.mean([r["median_latency"] for r in ok])
            entry["avg_p95_latency"] = statistics.mean([r["p95_latency"] for r in ok])
            entry["avg_throughput"] = statistics.mean([r["throughput"] for r in ok])
        else:
            entry["avg_median_latency"] = None
            entry["avg_p95_latency"] = None
            entry["avg_throughput"] = None
        out[str(w)] = entry
    return out


def _overall(results: list[dict]) -> dict[str, Any]:
    if not results:
        return {}
    ok = [r for r in results if _has_metrics(r)]
    out: dict[str, Any] = {
        "total_requests": sum(r["total_requests"] for r in results),
        "total_successful": sum(r["successful_requests"] for r in results),
        "total_failed": sum(r["failed_requests"] for r in results),
        "overall_success_rate": _success_rate(results),
    }
    if ok:
        out.update({
            "avg_median_latency": statistics.mean([r["median_latency"] for r in ok]),
            "avg_p95_latency": statistics.mean([r["p95_latency"] for r in ok]),
            "avg_throughput": statistics.mean([r["throughput"] for r in ok]),
            "median_latency_range": {
                "min": min(r["median_latency"] for r in ok),
                "max": max(r["median_latency"] for r in ok),
            },
            "throughput_range": {
                "min": min(r["throughput"] for r in ok),
                "max": max(r["throughput"] for r in ok),
            },
        })
    else:
        out.update({
            "avg_median_latency": None,
            "avg_p95_latency": None,
            "avg_throughput": None,
            "median_latency_range": None,
            "throughput_range": None,
        })
    return out


def _rankings(results: list[dict]) -> dict[str, Any]:
    ep_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"lat": [], "thr": [], "p95": []})
    for r in results:
        ep = r["endpoint_name"]
        if _has_metrics(r):
            ep_data[ep]["lat"].append(r["median_latency"])
            ep_data[ep]["thr"].append(r["throughput"])
            ep_data[ep]["p95"].append(r["p95_latency"])
        else:
            ep_data.setdefault(ep, {"lat": [], "thr": [], "p95": []})

    avg: dict[str, dict[str, float | None]] = {}
    for ep, v in ep_data.items():
        if v["lat"]:
            avg[ep] = {
                "avg_median_latency": statistics.mean(v["lat"]),
                "avg_throughput": statistics.mean(v["thr"]),
                "avg_p95_latency": statistics.mean(v["p95"]),
            }
        else:
            avg[ep] = {
                "avg_median_latency": None,
                "avg_throughput": None,
                "avg_p95_latency": None,
            }

    _INF = float("inf")

    return {
        "by_lowest_latency": [
            {"endpoint": ep, "avg_median_latency": m["avg_median_latency"]}
            for ep, m in sorted(
                avg.items(),
                key=lambda x: x[1]["avg_median_latency"] if x[1]["avg_median_latency"] is not None else _INF,
            )
        ],
        "by_highest_throughput": [
            {"endpoint": ep, "avg_throughput": m["avg_throughput"]}
            for ep, m in sorted(
                avg.items(),
                key=lambda x: x[1]["avg_throughput"] if x[1]["avg_throughput"] is not None else -1,
                reverse=True,
            )
        ],
        "by_lowest_p95_latency": [
            {"endpoint": ep, "avg_p95_latency": m["avg_p95_latency"]}
            for ep, m in sorted(
                avg.items(),
                key=lambda x: x[1]["avg_p95_latency"] if x[1]["avg_p95_latency"] is not None else _INF,
            )
        ],
    }
