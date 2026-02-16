#!/usr/bin/env python3
"""
Dual LLM Endpoint Comparison Tool
Compare performance of two endpoints with separate analysis files per parallel query configuration
"""

import asyncio
import time
import aiohttp
import json
import statistics
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class EndpointBenchmark:
    """Handles benchmarking for a single endpoint"""

    def __init__(self, name: str, endpoint_name: str, api_root: str, api_token: str,
                 request_timeout: int = 300, max_retries: int = 3):
        self.name = name
        self.endpoint_name = endpoint_name
        self.api_root = self._normalize_api_root(api_root)
        self.api_token = api_token
        self.endpoint_url = f'{self.api_root}/serving-endpoints/{endpoint_name}/invocations'
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.latencies = []
        self.failed_requests = 0

    @staticmethod
    def _normalize_api_root(api_root: str) -> str:
        normalized = api_root.rstrip("/")
        suffix = "/api/2.0"
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
        return normalized

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
        await asyncio.sleep(0.1 * index)

        for i in range(num_requests):
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
                                print(f"  [{self.name}] Worker {index}: Request {i+1}/{num_requests} ✓ {latency:.2f}s")
                            else:
                                response_text = (await response.text()).replace("\n", " ")[:300]
                                request_id = response.headers.get("x-databricks-request-id", "n/a")
                                print(
                                    f"  [{self.name}] Worker {index}: Request {i+1}/{num_requests} "
                                    f"failed (HTTP {response.status}, req_id={request_id}) "
                                    f"body={response_text!r}, retry {retry_count+1}/{self.max_retries}"
                                )
                                retry_count += 1
                                if retry_count < self.max_retries:
                                    await asyncio.sleep(1 * retry_count)

                except asyncio.TimeoutError:
                    print(f"  [{self.name}] Worker {index}: Request {i+1}/{num_requests} TIMEOUT, retry {retry_count+1}/{self.max_retries}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)
                except Exception as e:
                    print(f"  [{self.name}] Worker {index}: Request {i+1}/{num_requests} ERROR ({type(e).__name__}: {str(e)[:120]}), retry {retry_count+1}/{self.max_retries}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)

            if not success:
                self.failed_requests += 1
                print(f"  [{self.name}] Worker {index}: Request {i+1}/{num_requests} FAILED after {self.max_retries} retries")

    async def run_benchmark(self, num_workers: int, num_requests_per_worker: int,
                           in_tokens: int, out_tokens: int, qps: float) -> Optional[Dict]:
        """Run benchmark with specified parameters"""
        self.latencies.clear()
        self.failed_requests = 0

        tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(
                self.worker(i, num_requests_per_worker, in_tokens, out_tokens, qps)
            )
            tasks.append(task)

        start_time = time.time()
        await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        if not self.latencies:
            return None

        avg_input_tokens = statistics.mean([inp for inp, _, _ in self.latencies])
        avg_output_tokens = statistics.mean([outp for _, outp, _ in self.latencies])
        median_latency = statistics.median([latency for _, _, latency in self.latencies])
        p95_latency = statistics.quantiles([latency for _, _, latency in self.latencies], n=20)[18] if len(self.latencies) > 1 else median_latency
        tokens_per_sec = (avg_input_tokens + avg_output_tokens) * num_workers / median_latency

        return {
            'endpoint_name': self.name,
            'num_workers': num_workers,
            'median_latency': median_latency,
            'p95_latency': p95_latency,
            'throughput': tokens_per_sec,
            'avg_input_tokens': avg_input_tokens,
            'avg_output_tokens': avg_output_tokens,
            'successful_requests': len(self.latencies),
            'failed_requests': self.failed_requests,
            'total_requests': num_workers * num_requests_per_worker,
            'elapsed_time': elapsed_time
        }


async def compare_endpoints(endpoint_a: EndpointBenchmark, endpoint_b: EndpointBenchmark,
                           num_workers: int, num_requests_per_worker: int,
                           in_tokens: int, out_tokens: int, qps: float) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Run both endpoints concurrently and return results"""

    print(f"\n{'='*80}")
    print(f"Running: {num_workers} workers | QPS={qps} | Output tokens={out_tokens}")
    print(f"{'='*80}")

    # Run both endpoints in parallel
    result_a, result_b = await asyncio.gather(
        endpoint_a.run_benchmark(num_workers, num_requests_per_worker, in_tokens, out_tokens, qps),
        endpoint_b.run_benchmark(num_workers, num_requests_per_worker, in_tokens, out_tokens, qps)
    )

    # Print comparison
    if result_a and result_b:
        print(f"\n{'─'*80}")
        print(f"{'Metric':<25} {endpoint_a.name:<25} {endpoint_b.name:<25}")
        print(f"{'─'*80}")
        print(f"{'Median Latency':<25} {result_a['median_latency']:.2f}s{'':<19} {result_b['median_latency']:.2f}s")
        print(f"{'P95 Latency':<25} {result_a['p95_latency']:.2f}s{'':<19} {result_b['p95_latency']:.2f}s")
        print(f"{'Throughput':<25} {result_a['throughput']:.1f} tok/s{'':<12} {result_b['throughput']:.1f} tok/s")
        print(f"{'Failed Requests':<25} {result_a['failed_requests']}/{result_a['total_requests']}{'':<19} {result_b['failed_requests']}/{result_b['total_requests']}")

        # Calculate performance difference
        latency_diff = ((result_b['median_latency'] - result_a['median_latency']) / result_a['median_latency']) * 100
        throughput_diff = ((result_a['throughput'] - result_b['throughput']) / result_b['throughput']) * 100

        print(f"{'─'*80}")
        if latency_diff < 0:
            print(f"⚡ {endpoint_a.name} is {abs(latency_diff):.1f}% FASTER (lower latency)")
        else:
            print(f"⚡ {endpoint_b.name} is {abs(latency_diff):.1f}% FASTER (lower latency)")

        if throughput_diff > 0:
            print(f"🚀 {endpoint_a.name} has {abs(throughput_diff):.1f}% HIGHER throughput")
        else:
            print(f"🚀 {endpoint_b.name} has {abs(throughput_diff):.1f}% HIGHER throughput")
        print(f"{'─'*80}")

    return result_a, result_b


def create_comparison_charts(results_by_workers: Dict[int, List[Dict]],
                            endpoint_a_name: str, endpoint_b_name: str,
                            output_dir: str):
    """Create separate comparison charts for each parallel worker configuration"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Color scheme
    colors_a = {'latency': '#e74c3c', 'throughput': '#3498db', 'p95': '#c0392b'}
    colors_b = {'latency': '#f39c12', 'throughput': '#27ae60', 'p95': '#d68910'}

    for num_workers, results in results_by_workers.items():
        # Separate results by endpoint
        results_a = [r for r in results if r['endpoint_name'] == endpoint_a_name]
        results_b = [r for r in results if r['endpoint_name'] == endpoint_b_name]

        if not results_a or not results_b:
            continue

        # Group by QPS for better visualization
        qps_groups = {}
        for r in results_a:
            qps = r.get('qps', 'unknown')
            out_tokens = r.get('output_tokens', 0)
            key = (qps, out_tokens)
            if key not in qps_groups:
                qps_groups[key] = {'a': None, 'b': None}
            qps_groups[key]['a'] = r

        for r in results_b:
            qps = r.get('qps', 'unknown')
            out_tokens = r.get('output_tokens', 0)
            key = (qps, out_tokens)
            if key not in qps_groups:
                qps_groups[key] = {'a': None, 'b': None}
            qps_groups[key]['b'] = r

        # Sort by QPS and output tokens
        sorted_groups = sorted(qps_groups.items(), key=lambda x: (x[0][0], x[0][1]))

        # Create labels for x-axis
        labels = [f"QPS={qps}\n{out_tok}tok" for (qps, out_tok), _ in sorted_groups]
        x_pos = range(len(labels))

        # Extract data
        latency_a = [group['a']['median_latency'] if group['a'] else 0 for _, group in sorted_groups]
        latency_b = [group['b']['median_latency'] if group['b'] else 0 for _, group in sorted_groups]
        p95_a = [group['a']['p95_latency'] if group['a'] else 0 for _, group in sorted_groups]
        p95_b = [group['b']['p95_latency'] if group['b'] else 0 for _, group in sorted_groups]
        throughput_a = [group['a']['throughput'] if group['a'] else 0 for _, group in sorted_groups]
        throughput_b = [group['b']['throughput'] if group['b'] else 0 for _, group in sorted_groups]
        failed_a = [group['a']['failed_requests'] if group['a'] else 0 for _, group in sorted_groups]
        failed_b = [group['b']['failed_requests'] if group['b'] else 0 for _, group in sorted_groups]

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Endpoint Comparison - {num_workers} Parallel Workers',
                    fontsize=18, fontweight='bold', y=0.995)

        # 1. Median Latency Comparison
        ax1 = plt.subplot(2, 2, 1)
        width = 0.35
        x_a = [i - width/2 for i in x_pos]
        x_b = [i + width/2 for i in x_pos]

        bars_a = ax1.bar(x_a, latency_a, width, label=endpoint_a_name,
                        color=colors_a['latency'], alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_b = ax1.bar(x_b, latency_b, width, label=endpoint_b_name,
                        color=colors_b['latency'], alpha=0.8, edgecolor='black', linewidth=1.5)

        ax1.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Median Latency (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Median Latency Comparison', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars_a, bars_b]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 2. P95 Latency Comparison
        ax2 = plt.subplot(2, 2, 2)
        bars_a_p95 = ax2.bar(x_a, p95_a, width, label=endpoint_a_name,
                            color=colors_a['p95'], alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_b_p95 = ax2.bar(x_b, p95_b, width, label=endpoint_b_name,
                            color=colors_b['p95'], alpha=0.8, edgecolor='black', linewidth=1.5)

        ax2.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax2.set_ylabel('P95 Latency (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('P95 Latency Comparison (95th Percentile)', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=11, loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        for bars in [bars_a_p95, bars_b_p95]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 3. Throughput Comparison
        ax3 = plt.subplot(2, 2, 3)
        bars_a_thr = ax3.bar(x_a, throughput_a, width, label=endpoint_a_name,
                            color=colors_a['throughput'], alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_b_thr = ax3.bar(x_b, throughput_b, width, label=endpoint_b_name,
                            color=colors_b['throughput'], alpha=0.8, edgecolor='black', linewidth=1.5)

        ax3.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Throughput (tokens/second)', fontsize=12, fontweight='bold')
        ax3.set_title('Throughput Comparison', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax3.legend(fontsize=11, loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')

        for bars in [bars_a_thr, bars_b_thr]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 4. Failure Rate Comparison
        ax4 = plt.subplot(2, 2, 4)
        bars_a_fail = ax4.bar(x_a, failed_a, width, label=endpoint_a_name,
                             color='#e84118', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_b_fail = ax4.bar(x_b, failed_b, width, label=endpoint_b_name,
                             color='#fbc531', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax4.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Failed Requests', fontsize=12, fontweight='bold')
        ax4.set_title('Failure Rate Comparison', fontsize=14, fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=11, loc='upper left')
        ax4.grid(True, alpha=0.3, axis='y')

        for bars in [bars_a_fail, bars_b_fail]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()

        # Save figure
        filename = f"{output_dir}/comparison_{num_workers}workers.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
        plt.close()


def create_summary_chart(all_results: List[Dict], endpoint_a_name: str,
                        endpoint_b_name: str, output_dir: str):
    """Create overall summary chart showing trends across all worker configurations"""

    # Group by endpoint and workers
    data_by_endpoint = {endpoint_a_name: {}, endpoint_b_name: {}}

    for result in all_results:
        ep_name = result['endpoint_name']
        workers = result['num_workers']

        if workers not in data_by_endpoint[ep_name]:
            data_by_endpoint[ep_name][workers] = {
                'latencies': [],
                'throughputs': [],
                'p95_latencies': []
            }

        data_by_endpoint[ep_name][workers]['latencies'].append(result['median_latency'])
        data_by_endpoint[ep_name][workers]['throughputs'].append(result['throughput'])
        data_by_endpoint[ep_name][workers]['p95_latencies'].append(result['p95_latency'])

    # Calculate averages
    workers_list = sorted(set(r['num_workers'] for r in all_results))

    avg_latency_a = [statistics.mean(data_by_endpoint[endpoint_a_name][w]['latencies'])
                     if w in data_by_endpoint[endpoint_a_name] else 0 for w in workers_list]
    avg_latency_b = [statistics.mean(data_by_endpoint[endpoint_b_name][w]['latencies'])
                     if w in data_by_endpoint[endpoint_b_name] else 0 for w in workers_list]

    avg_throughput_a = [statistics.mean(data_by_endpoint[endpoint_a_name][w]['throughputs'])
                        if w in data_by_endpoint[endpoint_a_name] else 0 for w in workers_list]
    avg_throughput_b = [statistics.mean(data_by_endpoint[endpoint_b_name][w]['throughputs'])
                        if w in data_by_endpoint[endpoint_b_name] else 0 for w in workers_list]

    # Create summary figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Overall Performance Summary - Average Across All Traffic Conditions',
                fontsize=16, fontweight='bold')

    # Latency trend
    ax1.plot(workers_list, avg_latency_a, marker='o', linewidth=3, markersize=10,
            label=endpoint_a_name, color='#e74c3c')
    ax1.plot(workers_list, avg_latency_b, marker='s', linewidth=3, markersize=10,
            label=endpoint_b_name, color='#f39c12')
    ax1.set_xlabel('Parallel Workers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Median Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Scaling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    for i, (w, la, lb) in enumerate(zip(workers_list, avg_latency_a, avg_latency_b)):
        ax1.annotate(f'{la:.2f}s', (w, la), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
        ax1.annotate(f'{lb:.2f}s', (w, lb), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=9, color='#f39c12', fontweight='bold')

    # Throughput trend
    ax2.plot(workers_list, avg_throughput_a, marker='o', linewidth=3, markersize=10,
            label=endpoint_a_name, color='#3498db')
    ax2.plot(workers_list, avg_throughput_b, marker='s', linewidth=3, markersize=10,
            label=endpoint_b_name, color='#27ae60')
    ax2.set_xlabel('Parallel Workers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Throughput (tokens/second)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    for i, (w, ta, tb) in enumerate(zip(workers_list, avg_throughput_a, avg_throughput_b)):
        ax2.annotate(f'{int(ta)}', (w, ta), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, color='#3498db', fontweight='bold')
        ax2.annotate(f'{int(tb)}', (w, tb), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=9, color='#27ae60', fontweight='bold')

    plt.tight_layout()

    filename = f"{output_dir}/summary_overall.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


async def run_comparison_suite(args):
    """Run comprehensive comparison test suite"""

    # Create endpoint instances
    endpoint_a = EndpointBenchmark(
        name=args.endpoint_a_name,
        endpoint_name=args.endpoint_a,
        api_root=args.api_root_a,
        api_token=args.api_token_a,
        request_timeout=args.timeout,
        max_retries=args.max_retries
    )

    endpoint_b = EndpointBenchmark(
        name=args.endpoint_b_name,
        endpoint_name=args.endpoint_b,
        api_root=args.api_root_b,
        api_token=args.api_token_b,
        request_timeout=args.timeout,
        max_retries=args.max_retries
    )

    print(f"\n{'#'*80}")
    print(f"# Dual Endpoint Comparison Benchmarking Suite")
    print(f"# Endpoint A: {args.endpoint_a_name} ({args.endpoint_a})")
    print(f"# Endpoint B: {args.endpoint_b_name} ({args.endpoint_b})")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    all_results = []
    results_by_workers = {}

    # Run tests for each configuration
    for num_workers in args.parallel_workers:
        results_by_workers[num_workers] = []

        print(f"\n{'*'*80}")
        print(f"* Testing with {num_workers} Parallel Workers")
        print(f"{'*'*80}")

        for qps in args.qps_list:
            for out_tokens in args.output_tokens:
                result_a, result_b = await compare_endpoints(
                    endpoint_a, endpoint_b,
                    num_workers=num_workers,
                    num_requests_per_worker=args.requests_per_worker,
                    in_tokens=args.input_tokens,
                    out_tokens=out_tokens,
                    qps=qps
                )

                if result_a:
                    result_a['qps'] = qps
                    result_a['output_tokens'] = out_tokens
                    all_results.append(result_a)
                    results_by_workers[num_workers].append(result_a)

                if result_b:
                    result_b['qps'] = qps
                    result_b['output_tokens'] = out_tokens
                    all_results.append(result_b)
                    results_by_workers[num_workers].append(result_b)

                # Brief pause between tests
                await asyncio.sleep(2)

    print(f"\n{'#'*80}")
    print(f"# Benchmark Complete!")
    print(f"# End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    # Generate comparison charts
    print(f"\n{'='*80}")
    print("Generating comparison analysis files...")
    print(f"{'='*80}\n")

    create_comparison_charts(results_by_workers, args.endpoint_a_name,
                           args.endpoint_b_name, args.output_dir)
    create_summary_chart(all_results, args.endpoint_a_name,
                        args.endpoint_b_name, args.output_dir)

    # Save detailed results to JSON
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'endpoint_a': {
            'name': args.endpoint_a_name,
            'endpoint': args.endpoint_a
        },
        'endpoint_b': {
            'name': args.endpoint_b_name,
            'endpoint': args.endpoint_b
        },
        'results': all_results
    }

    json_file = f"{args.output_dir}/comparison_results.json"
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved: {json_file}")

    print(f"\n{'='*80}")
    print(f"All analysis files saved to: {args.output_dir}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare two LLM endpoints with detailed analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  %(prog)s \\
    --endpoint-a gpt-model-a --endpoint-a-name "GPT-A" --api-token-a TOKEN_A \\
    --endpoint-b gpt-model-b --endpoint-b-name "GPT-B" --api-token-b TOKEN_B

  # Full comparison with custom traffic patterns
  %(prog)s \\
    --endpoint-a gpt-20b --endpoint-a-name "GPT-20B" --api-token-a TOKEN_A \\
    --endpoint-b llama-70b --endpoint-b-name "Llama-70B" --api-token-b TOKEN_B \\
    --qps-list 0.5 1 2 4 \\
    --output-tokens 200 500 1000 \\
    --parallel-workers 4 8 16 \\
    --output-dir comparison_results
        """
    )

    # Endpoint A
    parser.add_argument('--endpoint-a', required=True, help='First endpoint name')
    parser.add_argument('--endpoint-a-name', required=True, help='Display name for endpoint A')
    parser.add_argument('--api-token-a', required=True, help='API token for endpoint A')
    parser.add_argument('--api-root-a', default=os.getenv('API_ROOT', 'https://your-workspace.cloud.databricks.com'),
                       help='API root for endpoint A')

    # Endpoint B
    parser.add_argument('--endpoint-b', required=True, help='Second endpoint name')
    parser.add_argument('--endpoint-b-name', required=True, help='Display name for endpoint B')
    parser.add_argument('--api-token-b', required=True, help='API token for endpoint B')
    parser.add_argument('--api-root-b', default=os.getenv('API_ROOT', 'https://your-workspace.cloud.databricks.com'),
                       help='API root for endpoint B')

    # Test parameters
    parser.add_argument('--input-tokens', type=int, default=1000,
                       help='Number of input tokens (default: 1000)')
    parser.add_argument('--output-tokens', type=int, nargs='+', default=[200, 500, 1000],
                       help='Output token sizes to test (default: 200 500 1000)')
    parser.add_argument('--qps-list', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                       help='QPS rates to test (default: 0.5 1.0 2.0)')
    parser.add_argument('--parallel-workers', type=int, nargs='+', default=[4, 6, 8],
                       help='Number of parallel workers to test (default: 4 6 8)')
    parser.add_argument('--requests-per-worker', type=int, default=5,
                       help='Number of requests per worker (default: 5)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts (default: 3)')
    parser.add_argument('--output-dir', default='comparison_analysis',
                       help='Output directory for analysis files (default: comparison_analysis)')

    args = parser.parse_args()

    try:
        asyncio.run(run_comparison_suite(args))
    except KeyboardInterrupt:
        print("\n\n⚠ Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
