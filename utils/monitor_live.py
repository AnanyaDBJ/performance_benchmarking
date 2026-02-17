#!/usr/bin/env python3
"""Live monitor for up to 4 Databricks serving endpoints."""

import argparse
import os
import time
import sys
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse

import requests


def get_endpoint_status(api_root: str, api_token: str, endpoint_name: str) -> Dict:
    """Get current endpoint status."""
    endpoint_url = f"{api_root}/api/2.0/serving-endpoints/{endpoint_name}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    response = requests.get(endpoint_url, headers=headers, timeout=10, allow_redirects=False)
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


def validate_api_root(parser: argparse.ArgumentParser, api_root: str) -> str:
    normalized = api_root.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        parser.error("--api-root must be a valid https URL (example: https://your-workspace.cloud.databricks.com)")
    return normalized


def resolve_api_token(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    sources = int(bool(args.api_token)) + int(bool(args.api_token_env)) + int(args.api_token_stdin)
    if sources > 1:
        parser.error("Use only one token source: --api-token OR --api-token-env OR --api-token-stdin")

    if args.api_token:
        print(
            "Warning: passing tokens via --api-token can expose secrets in shell history and process lists. "
            "Prefer --api-token-env or --api-token-stdin.",
            file=sys.stderr,
        )
        token = args.api_token.strip()
    elif args.api_token_env:
        token = os.getenv(args.api_token_env, "").strip()
        if not token:
            parser.error(f"Environment variable {args.api_token_env} is empty or not set.")
    elif args.api_token_stdin:
        token = sys.stdin.readline().strip()
        if not token:
            parser.error("--api-token-stdin was set but no token was provided on stdin.")
    else:
        token = os.getenv("DATABRICKS_API_TOKEN", "").strip()
        if not token:
            token = os.getenv("API_TOKEN", "").strip()
        if not token:
            parser.error(
                "API token is required. Provide --api-token, --api-token-env <ENV_NAME>, "
                "--api-token-stdin, or set DATABRICKS_API_TOKEN."
            )
    return token


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
        help="Databricks API token (less secure; prefer --api-token-env or --api-token-stdin).",
    )
    parser.add_argument(
        "--api-token-env",
        help="Read API token from the named environment variable.",
    )
    parser.add_argument(
        "--api-token-stdin",
        action="store_true",
        help="Read API token from stdin (first line).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Polling interval in seconds (default: 5).",
    )
    args = parser.parse_args()
    args.api_root = validate_api_root(parser, args.api_root)
    args.api_token = resolve_api_token(parser, args)

    if len(args.endpoints) < 1 or len(args.endpoints) > 4:
        parser.error("--endpoints must contain between 1 and 4 endpoint names.")
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
