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

    for endpoint, m in summary["by_endpoint"].items():
        writer.writerow([
            endpoint,
            m["num_configurations"],
            f"{m['success_rate']:.2f}",
            f"{m['avg_median_latency']:.4f}",
            f"{m['min_median_latency']:.4f}",
            f"{m['max_median_latency']:.4f}",
            f"{m['avg_p95_latency']:.4f}",
            f"{m['avg_throughput']:.2f}",
            f"{m['min_throughput']:.2f}",
            f"{m['max_throughput']:.2f}",
            f"{m['avg_input_tokens']:.1f}",
            f"{m['avg_output_tokens']:.1f}",
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


def _by_endpoint(results: list[dict]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["endpoint_name"]].append(r)

    out = {}
    for ep, rs in groups.items():
        out[ep] = {
            "num_configurations": len(rs),
            "avg_median_latency": statistics.mean([r["median_latency"] for r in rs]),
            "min_median_latency": min(r["median_latency"] for r in rs),
            "max_median_latency": max(r["median_latency"] for r in rs),
            "avg_p95_latency": statistics.mean([r["p95_latency"] for r in rs]),
            "min_p95_latency": min(r["p95_latency"] for r in rs),
            "max_p95_latency": max(r["p95_latency"] for r in rs),
            "avg_throughput": statistics.mean([r["throughput"] for r in rs]),
            "min_throughput": min(r["throughput"] for r in rs),
            "max_throughput": max(r["throughput"] for r in rs),
            "total_successful_requests": sum(r["successful_requests"] for r in rs),
            "total_failed_requests": sum(r["failed_requests"] for r in rs),
            "success_rate": _success_rate(rs),
            "avg_input_tokens": statistics.mean([r["avg_input_tokens"] for r in rs]),
            "avg_output_tokens": statistics.mean([r["avg_output_tokens"] for r in rs]),
        }
    return out


def _by_workers(results: list[dict]) -> dict[str, Any]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["num_workers"]].append(r)

    out = {}
    for w, rs in groups.items():
        out[str(w)] = {
            "num_endpoints": len(rs),
            "avg_median_latency": statistics.mean([r["median_latency"] for r in rs]),
            "avg_p95_latency": statistics.mean([r["p95_latency"] for r in rs]),
            "avg_throughput": statistics.mean([r["throughput"] for r in rs]),
            "total_successful_requests": sum(r["successful_requests"] for r in rs),
            "total_failed_requests": sum(r["failed_requests"] for r in rs),
            "success_rate": _success_rate(rs),
        }
    return out


def _overall(results: list[dict]) -> dict[str, Any]:
    if not results:
        return {}
    return {
        "total_requests": sum(r["total_requests"] for r in results),
        "total_successful": sum(r["successful_requests"] for r in results),
        "total_failed": sum(r["failed_requests"] for r in results),
        "overall_success_rate": _success_rate(results),
        "avg_median_latency": statistics.mean([r["median_latency"] for r in results]),
        "avg_p95_latency": statistics.mean([r["p95_latency"] for r in results]),
        "avg_throughput": statistics.mean([r["throughput"] for r in results]),
        "median_latency_range": {
            "min": min(r["median_latency"] for r in results),
            "max": max(r["median_latency"] for r in results),
        },
        "throughput_range": {
            "min": min(r["throughput"] for r in results),
            "max": max(r["throughput"] for r in results),
        },
    }


def _rankings(results: list[dict]) -> dict[str, Any]:
    ep_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"lat": [], "thr": [], "p95": []})
    for r in results:
        ep = r["endpoint_name"]
        ep_data[ep]["lat"].append(r["median_latency"])
        ep_data[ep]["thr"].append(r["throughput"])
        ep_data[ep]["p95"].append(r["p95_latency"])

    avg = {
        ep: {
            "avg_median_latency": statistics.mean(v["lat"]),
            "avg_throughput": statistics.mean(v["thr"]),
            "avg_p95_latency": statistics.mean(v["p95"]),
        }
        for ep, v in ep_data.items()
    }

    return {
        "by_lowest_latency": [
            {"endpoint": ep, "avg_median_latency": m["avg_median_latency"]}
            for ep, m in sorted(avg.items(), key=lambda x: x[1]["avg_median_latency"])
        ],
        "by_highest_throughput": [
            {"endpoint": ep, "avg_throughput": m["avg_throughput"]}
            for ep, m in sorted(avg.items(), key=lambda x: x[1]["avg_throughput"], reverse=True)
        ],
        "by_lowest_p95_latency": [
            {"endpoint": ep, "avg_p95_latency": m["avg_p95_latency"]}
            for ep, m in sorted(avg.items(), key=lambda x: x[1]["avg_p95_latency"])
        ],
    }
