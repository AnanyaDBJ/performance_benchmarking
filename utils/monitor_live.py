#!/usr/bin/env python3
"""Live monitor for up to 4 Databricks serving endpoints."""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List

import requests


def get_endpoint_status(api_root: str, api_token: str, endpoint_name: str) -> Dict:
    """Get current endpoint status."""
    endpoint_url = f"{api_root}/api/2.0/serving-endpoints/{endpoint_name}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    response = requests.get(endpoint_url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


def monitor(api_root: str, api_token: str, endpoints: List[str], interval: int) -> None:
    """Monitor one or more endpoints in real time."""
    print(f"\n{'=' * 110}")
    print("LIVE ENDPOINT MONITOR")
    print(f"{'=' * 110}")
    print(f"API Root: {api_root}")
    print(f"Endpoints: {', '.join(endpoints)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 110}\n")
    print("Press Ctrl+C to stop\n")
    print(f"{'Time':<10} {'Endpoint':<30} {'Connection':<14} {'Ready':<18} {'Config':<18}")
    print(f"{'-' * 110}")

    last_states = {}
    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            for endpoint in endpoints:
                try:
                    status = get_endpoint_status(api_root, api_token, endpoint)
                    state = status.get("state", {}).get("ready", "UNKNOWN")
                    config_state = status.get("state", {}).get("config_update", "N/A")
                    state_key = f"{state}:{config_state}"

                    if last_states.get(endpoint) != state_key or iteration % 6 == 0:
                        print(
                            f"{timestamp:<10} {endpoint:<30} {'Connected':<14} "
                            f"{state:<18} {config_state:<18}"
                        )
                        last_states[endpoint] = state_key
                except Exception as err:
                    print(
                        f"{timestamp:<10} {endpoint:<30} {'Error':<14} "
                        f"{'N/A':<18} {type(err).__name__:<18}"
                    )

            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n\n{'=' * 110}")
        print("Monitor stopped")
        print(f"{'=' * 110}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor 1-4 Databricks serving endpoints in real time."
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        required=True,
        help="Endpoint names to monitor (space-separated).",
    )
    parser.add_argument(
        "--api-root",
        default=os.getenv("API_ROOT", "https://your-workspace.cloud.databricks.com"),
        help="Databricks API root URL. Defaults to API_ROOT env var.",
    )
    parser.add_argument(
        "--api-token",
        default=os.getenv("DATABRICKS_API_TOKEN"),
        help="Databricks API token. Defaults to DATABRICKS_API_TOKEN env var.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Polling interval in seconds (default: 5).",
    )
    args = parser.parse_args()

    if len(args.endpoints) < 1 or len(args.endpoints) > 4:
        parser.error("--endpoints must contain between 1 and 4 endpoint names.")
    if not args.api_token:
        parser.error("--api-token is required (or set DATABRICKS_API_TOKEN).")
    if args.interval < 1:
        parser.error("--interval must be >= 1.")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    monitor(
        api_root=cli_args.api_root,
        api_token=cli_args.api_token,
        endpoints=cli_args.endpoints,
        interval=cli_args.interval,
    )
