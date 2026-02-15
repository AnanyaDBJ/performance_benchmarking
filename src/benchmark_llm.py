#!/usr/bin/env python3
"""
LLM Endpoint Benchmarking Tool
Run load tests against Databricks LLM serving endpoints
"""

import asyncio
import time
import aiohttp
import json
import statistics
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Optional


class LLMBenchmark:
    def __init__(self, endpoint_name: str, api_root: str, api_token: str,
                 request_timeout: int = 300, max_retries: int = 3):
        self.endpoint_name = endpoint_name
        self.api_root = api_root
        self.api_token = api_token
        self.endpoint_url = f'{api_root}/serving-endpoints/{endpoint_name}/invocations'
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.latencies = []
        self.failed_requests = 0

    def get_request(self, in_tokens: int, out_tokens: int) -> dict:
        """Generate request payload with approximate token count"""
        overhead_tokens = 50
        user_content_tokens = max(1, in_tokens - overhead_tokens)
        repeat_count = max(1, user_content_tokens // 2)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You must generate exactly {out_tokens} tokens in your response. Write a detailed explanation that uses precisely this number of tokens."
                },
                {
                    "role": "user",
                    "content": "Hello! " * repeat_count
                }
            ],
            "max_tokens": out_tokens,
            "temperature": 0.0
        }

    async def worker(self, index: int, num_requests: int, in_tokens: int,
                    out_tokens: int, qps: float):
        """Single worker that processes requests with error handling"""
        input_data = self.get_request(in_tokens, out_tokens)

        # Offset threads slightly to avoid thundering herd
        await asyncio.sleep(0.1 * index)

        for i in range(num_requests):
            # Rate limiting: wait between requests based on QPS
            if i > 0:
                await asyncio.sleep(1.0 / qps)

            request_start_time = time.time()
            retry_count = 0
            success = False

            while not success and retry_count < self.max_retries:
                try:
                    timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(self.endpoint_url, headers=self.headers,
                                               json=input_data) as response:
                            if response.ok:
                                chunks = []
                                async for chunk, _ in response.content.iter_chunks():
                                    chunks.append(chunk)

                                latency = time.time() - request_start_time
                                result = json.loads(b''.join(chunks))
                                self.latencies.append((
                                    result['usage']['prompt_tokens'],
                                    result['usage']['completion_tokens'],
                                    latency
                                ))
                                success = True
                                print(f"  ✓ Worker {index}: Request {i+1}/{num_requests} completed in {latency:.2f}s")
                            else:
                                retry_count += 1
                                print(f"  ✗ Worker {index}: Request {i+1} failed (HTTP {response.status}), retry {retry_count}/{self.max_retries}")
                                if retry_count < self.max_retries:
                                    await asyncio.sleep(1 * retry_count)

                except asyncio.TimeoutError:
                    retry_count += 1
                    print(f"  ⏱ Worker {index}: Request {i+1} timed out, retry {retry_count}/{self.max_retries}")
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)
                except Exception as e:
                    retry_count += 1
                    print(f"  ⚠ Worker {index}: Request {i+1} error: {e}, retry {retry_count}/{self.max_retries}")
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)

            if not success:
                self.failed_requests += 1
                print(f"  ✗ Worker {index}: Request {i+1} failed after {self.max_retries} retries")

    async def run_benchmark(self, num_workers: int, num_requests_per_worker: int,
                           in_tokens: int, out_tokens: int, qps: float) -> Optional[Dict]:
        """Run benchmark with specified parameters"""
        print(f"\n{'='*70}")
        print(f"Benchmark: {num_workers} workers | {num_requests_per_worker} req/worker | "
              f"{out_tokens} out tokens | QPS={qps}")
        print(f"{'='*70}")

        self.latencies.clear()
        self.failed_requests = 0

        start_time = time.time()

        # Create and run workers
        tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(
                self.worker(i, num_requests_per_worker, in_tokens, out_tokens, qps)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        if not self.latencies:
            print(f"\n⚠ No successful requests!")
            return None

        # Compute statistics
        avg_input_tokens = statistics.mean([inp for inp, _, _ in self.latencies])
        avg_output_tokens = statistics.mean([outp for _, outp, _ in self.latencies])
        median_latency = statistics.median([latency for _, _, latency in self.latencies])
        tokens_per_sec = (avg_input_tokens + avg_output_tokens) * num_workers / median_latency

        print(f"\n{'─'*70}")
        print(f"Results:")
        print(f"  Input tokens:    {avg_input_tokens:.0f}")
        print(f"  Output tokens:   {avg_output_tokens:.0f}")
        print(f"  Median latency:  {median_latency:.2f}s")
        print(f"  Throughput:      {tokens_per_sec:.1f} tokens/sec")
        print(f"  Failed requests: {self.failed_requests}/{num_workers * num_requests_per_worker}")
        print(f"  Total time:      {elapsed_time:.2f}s")
        print(f"{'─'*70}")

        return {
            'num_workers': num_workers,
            'median_latency': median_latency,
            'throughput': tokens_per_sec,
            'avg_input_tokens': avg_input_tokens,
            'avg_output_tokens': avg_output_tokens,
            'failed_requests': self.failed_requests,
            'total_requests': num_workers * num_requests_per_worker,
            'elapsed_time': elapsed_time
        }


async def run_test_suite(benchmark: LLMBenchmark, args):
    """Run comprehensive test suite"""
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# LLM Endpoint Benchmarking Suite")
    print(f"# Endpoint: {benchmark.endpoint_name}")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    for qps in args.qps_list:
        for out_tokens in args.output_tokens:
            for num_workers in args.parallel_workers:
                result = await benchmark.run_benchmark(
                    num_workers=num_workers,
                    num_requests_per_worker=args.requests_per_worker,
                    in_tokens=args.input_tokens,
                    out_tokens=out_tokens,
                    qps=qps
                )

                if result:
                    result['qps'] = qps
                    result['output_tokens'] = out_tokens
                    all_results.append(result)

                # Brief pause between tests
                await asyncio.sleep(2)

    print(f"\n{'#'*70}")
    print(f"# Test Suite Complete!")
    print(f"# End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # Print summary table
    print_summary_table(all_results)

    # Save results to file if requested
    if args.output_file:
        save_results(all_results, args.output_file)


def print_summary_table(results: List[Dict]):
    """Print formatted summary table"""
    if not results:
        return

    print(f"\n{'='*100}")
    print(f"SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'QPS':<8} {'Out Tokens':<12} {'Workers':<10} {'Latency (s)':<15} "
          f"{'Throughput':<20} {'Failed':<10}")
    print(f"{'-'*100}")

    for r in results:
        print(f"{r['qps']:<8} {int(r['output_tokens']):<12} {r['num_workers']:<10} "
              f"{r['median_latency']:<15.2f} {r['throughput']:<20.1f} "
              f"{r['failed_requests']}/{r['total_requests']:<10}")

    print(f"{'='*100}\n")


def save_results(results: List[Dict], filename: str):
    """Save results to JSON file"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark LLM endpoint performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  %(prog)s --endpoint gpt-oss-20b-load-testing --api-token YOUR_TOKEN

  # Custom test with specific parameters
  %(prog)s --endpoint my-llm --api-token TOKEN --qps 1 2 --output-tokens 500 1000 --workers 4 8

  # Save results to file
  %(prog)s --endpoint my-llm --api-token TOKEN --output results.json
        """
    )

    # Required arguments
    parser.add_argument('--endpoint', required=True,
                       help='Name of the Databricks serving endpoint')
    parser.add_argument('--api-token', required=True,
                       help='Databricks API token')

    # Optional arguments
    parser.add_argument('--api-root', default='https://your-workspace.cloud.databricks.com',
                       help='Databricks API root URL')
    parser.add_argument('--input-tokens', type=int, default=1000,
                       help='Number of input tokens (default: 1000)')
    parser.add_argument('--output-tokens', type=int, nargs='+', default=[200, 500, 1000],
                       help='Output token sizes to test (default: 200 500 1000)')
    parser.add_argument('--qps-list', type=float, nargs='+', default=[0.5, 1.0],
                       help='QPS rates to test (default: 0.5 1.0)')
    parser.add_argument('--parallel-workers', type=int, nargs='+', default=[4, 6],
                       help='Number of parallel workers to test (default: 4 6)')
    parser.add_argument('--requests-per-worker', type=int, default=5,
                       help='Number of requests per worker (default: 5)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts (default: 3)')
    parser.add_argument('--output-file', '-o',
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = LLMBenchmark(
        endpoint_name=args.endpoint,
        api_root=args.api_root,
        api_token=args.api_token,
        request_timeout=args.timeout,
        max_retries=args.max_retries
    )

    # Run the test suite
    try:
        asyncio.run(run_test_suite(benchmark, args))
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
