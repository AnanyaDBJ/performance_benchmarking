#!/usr/bin/env python3
"""
Generate summary statistics from LLM endpoint comparison results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics


class SummaryStatsGenerator:
    """Generate comprehensive summary statistics from benchmark results."""

    def __init__(self, results_path: str):
        """Initialize with path to results.json file."""
        self.results_path = results_path
        self.data = self._load_results()

    def _load_results(self) -> Dict:
        """Load results from JSON file."""
        with open(self.results_path, 'r') as f:
            return json.load(f)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            'timestamp': self.data.get('timestamp', 'N/A'),
            'endpoints_tested': self.data.get('endpoints', []),
            'total_configurations': len(self.data.get('results', [])),
            'by_endpoint': self._summarize_by_endpoint(),
            'by_worker_count': self._summarize_by_workers(),
            'overall_metrics': self._calculate_overall_metrics(),
            'performance_rankings': self._rank_performance(),
        }
        return summary

    def _summarize_by_endpoint(self) -> Dict[str, Dict]:
        """Summarize metrics grouped by endpoint."""
        endpoint_data = defaultdict(list)

        for result in self.data.get('results', []):
            endpoint_name = result['endpoint_name']
            endpoint_data[endpoint_name].append(result)

        summary = {}
        for endpoint, results in endpoint_data.items():
            summary[endpoint] = {
                'num_configurations': len(results),
                'avg_median_latency': statistics.mean([r['median_latency'] for r in results]),
                'min_median_latency': min([r['median_latency'] for r in results]),
                'max_median_latency': max([r['median_latency'] for r in results]),
                'avg_p95_latency': statistics.mean([r['p95_latency'] for r in results]),
                'min_p95_latency': min([r['p95_latency'] for r in results]),
                'max_p95_latency': max([r['p95_latency'] for r in results]),
                'avg_throughput': statistics.mean([r['throughput'] for r in results]),
                'min_throughput': min([r['throughput'] for r in results]),
                'max_throughput': max([r['throughput'] for r in results]),
                'total_successful_requests': sum([r['successful_requests'] for r in results]),
                'total_failed_requests': sum([r['failed_requests'] for r in results]),
                'success_rate': self._calculate_success_rate(results),
                'avg_input_tokens': statistics.mean([r['avg_input_tokens'] for r in results]),
                'avg_output_tokens': statistics.mean([r['avg_output_tokens'] for r in results]),
            }

        return summary

    def _summarize_by_workers(self) -> Dict[int, Dict]:
        """Summarize metrics grouped by worker count."""
        worker_data = defaultdict(list)

        for result in self.data.get('results', []):
            num_workers = result['num_workers']
            worker_data[num_workers].append(result)

        summary = {}
        for workers, results in worker_data.items():
            summary[workers] = {
                'num_endpoints': len(results),
                'avg_median_latency': statistics.mean([r['median_latency'] for r in results]),
                'avg_p95_latency': statistics.mean([r['p95_latency'] for r in results]),
                'avg_throughput': statistics.mean([r['throughput'] for r in results]),
                'total_successful_requests': sum([r['successful_requests'] for r in results]),
                'total_failed_requests': sum([r['failed_requests'] for r in results]),
                'success_rate': self._calculate_success_rate(results),
            }

        return summary

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics across all configurations."""
        results = self.data.get('results', [])

        if not results:
            return {}

        return {
            'total_requests': sum([r['total_requests'] for r in results]),
            'total_successful': sum([r['successful_requests'] for r in results]),
            'total_failed': sum([r['failed_requests'] for r in results]),
            'overall_success_rate': self._calculate_success_rate(results),
            'avg_median_latency': statistics.mean([r['median_latency'] for r in results]),
            'avg_p95_latency': statistics.mean([r['p95_latency'] for r in results]),
            'avg_throughput': statistics.mean([r['throughput'] for r in results]),
            'median_latency_range': {
                'min': min([r['median_latency'] for r in results]),
                'max': max([r['median_latency'] for r in results]),
            },
            'throughput_range': {
                'min': min([r['throughput'] for r in results]),
                'max': max([r['throughput'] for r in results]),
            }
        }

    def _rank_performance(self) -> Dict[str, List]:
        """Rank endpoints by various performance metrics."""
        results = self.data.get('results', [])

        # Group by endpoint and calculate average metrics
        endpoint_metrics = defaultdict(lambda: {
            'median_latency': [],
            'throughput': [],
            'p95_latency': []
        })

        for result in results:
            endpoint = result['endpoint_name']
            endpoint_metrics[endpoint]['median_latency'].append(result['median_latency'])
            endpoint_metrics[endpoint]['throughput'].append(result['throughput'])
            endpoint_metrics[endpoint]['p95_latency'].append(result['p95_latency'])

        # Calculate averages
        avg_metrics = {}
        for endpoint, metrics in endpoint_metrics.items():
            avg_metrics[endpoint] = {
                'avg_median_latency': statistics.mean(metrics['median_latency']),
                'avg_throughput': statistics.mean(metrics['throughput']),
                'avg_p95_latency': statistics.mean(metrics['p95_latency']),
            }

        # Rank by different metrics
        rankings = {
            'lowest_latency': sorted(
                avg_metrics.items(),
                key=lambda x: x[1]['avg_median_latency']
            ),
            'highest_throughput': sorted(
                avg_metrics.items(),
                key=lambda x: x[1]['avg_throughput'],
                reverse=True
            ),
            'lowest_p95_latency': sorted(
                avg_metrics.items(),
                key=lambda x: x[1]['avg_p95_latency']
            ),
        }

        return {
            'by_lowest_latency': [
                {'endpoint': ep, 'avg_median_latency': metrics['avg_median_latency']}
                for ep, metrics in rankings['lowest_latency']
            ],
            'by_highest_throughput': [
                {'endpoint': ep, 'avg_throughput': metrics['avg_throughput']}
                for ep, metrics in rankings['highest_throughput']
            ],
            'by_lowest_p95_latency': [
                {'endpoint': ep, 'avg_p95_latency': metrics['avg_p95_latency']}
                for ep, metrics in rankings['lowest_p95_latency']
            ],
        }

    def _calculate_success_rate(self, results: List[Dict]) -> float:
        """Calculate success rate as percentage."""
        total = sum([r['total_requests'] for r in results])
        successful = sum([r['successful_requests'] for r in results])
        return (successful / total * 100) if total > 0 else 0.0

    def print_summary(self, summary: Dict):
        """Print formatted summary to console."""
        print("=" * 80)
        print("LLM ENDPOINT BENCHMARK SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Endpoints Tested: {', '.join(summary['endpoints_tested'])}")
        print(f"Total Configurations: {summary['total_configurations']}")

        # Overall Metrics
        print("\n" + "=" * 80)
        print("OVERALL METRICS")
        print("=" * 80)
        overall = summary['overall_metrics']
        print(f"Total Requests: {overall['total_requests']}")
        print(f"Successful: {overall['total_successful']}")
        print(f"Failed: {overall['total_failed']}")
        print(f"Success Rate: {overall['overall_success_rate']:.2f}%")
        print(f"Average Median Latency: {overall['avg_median_latency']:.4f}s")
        print(f"Average P95 Latency: {overall['avg_p95_latency']:.4f}s")
        print(f"Average Throughput: {overall['avg_throughput']:.2f} tokens/s")
        print(f"Latency Range: {overall['median_latency_range']['min']:.4f}s - {overall['median_latency_range']['max']:.4f}s")
        print(f"Throughput Range: {overall['throughput_range']['min']:.2f} - {overall['throughput_range']['max']:.2f} tokens/s")

        # Performance Rankings
        print("\n" + "=" * 80)
        print("PERFORMANCE RANKINGS")
        print("=" * 80)

        print("\nLowest Latency (Best to Worst):")
        for i, item in enumerate(summary['performance_rankings']['by_lowest_latency'], 1):
            print(f"  {i}. {item['endpoint']}: {item['avg_median_latency']:.4f}s")

        print("\nHighest Throughput (Best to Worst):")
        for i, item in enumerate(summary['performance_rankings']['by_highest_throughput'], 1):
            print(f"  {i}. {item['endpoint']}: {item['avg_throughput']:.2f} tokens/s")

        print("\nLowest P95 Latency (Best to Worst):")
        for i, item in enumerate(summary['performance_rankings']['by_lowest_p95_latency'], 1):
            print(f"  {i}. {item['endpoint']}: {item['avg_p95_latency']:.4f}s")

        # By Endpoint
        print("\n" + "=" * 80)
        print("SUMMARY BY ENDPOINT")
        print("=" * 80)
        for endpoint, metrics in summary['by_endpoint'].items():
            print(f"\n{endpoint}:")
            print(f"  Configurations Tested: {metrics['num_configurations']}")
            print(f"  Success Rate: {metrics['success_rate']:.2f}%")
            print(f"  Median Latency: avg={metrics['avg_median_latency']:.4f}s, min={metrics['min_median_latency']:.4f}s, max={metrics['max_median_latency']:.4f}s")
            print(f"  P95 Latency: avg={metrics['avg_p95_latency']:.4f}s, min={metrics['min_p95_latency']:.4f}s, max={metrics['max_p95_latency']:.4f}s")
            print(f"  Throughput: avg={metrics['avg_throughput']:.2f}, min={metrics['min_throughput']:.2f}, max={metrics['max_throughput']:.2f} tokens/s")
            print(f"  Avg Input Tokens: {metrics['avg_input_tokens']:.1f}")
            print(f"  Avg Output Tokens: {metrics['avg_output_tokens']:.1f}")
            print(f"  Total Requests: {metrics['total_successful_requests'] + metrics['total_failed_requests']} (Success: {metrics['total_successful_requests']}, Failed: {metrics['total_failed_requests']})")

        # By Worker Count
        print("\n" + "=" * 80)
        print("SUMMARY BY WORKER COUNT")
        print("=" * 80)
        for workers, metrics in sorted(summary['by_worker_count'].items()):
            print(f"\n{workers} Workers:")
            print(f"  Endpoints Tested: {metrics['num_endpoints']}")
            print(f"  Success Rate: {metrics['success_rate']:.2f}%")
            print(f"  Avg Median Latency: {metrics['avg_median_latency']:.4f}s")
            print(f"  Avg P95 Latency: {metrics['avg_p95_latency']:.4f}s")
            print(f"  Avg Throughput: {metrics['avg_throughput']:.2f} tokens/s")
            print(f"  Total Requests: {metrics['total_successful_requests'] + metrics['total_failed_requests']}")

        print("\n" + "=" * 80)

    def save_summary_json(self, summary: Dict, output_path: str):
        """Save summary to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {output_path}")

    def save_summary_csv(self, summary: Dict, output_path: str):
        """Save endpoint summary to CSV file."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Endpoint',
                'Configurations',
                'Success Rate (%)',
                'Avg Median Latency (s)',
                'Min Median Latency (s)',
                'Max Median Latency (s)',
                'Avg P95 Latency (s)',
                'Avg Throughput (tokens/s)',
                'Min Throughput (tokens/s)',
                'Max Throughput (tokens/s)',
                'Avg Input Tokens',
                'Avg Output Tokens',
                'Total Requests',
                'Successful Requests',
                'Failed Requests'
            ])

            # Data rows
            for endpoint, metrics in summary['by_endpoint'].items():
                writer.writerow([
                    endpoint,
                    metrics['num_configurations'],
                    f"{metrics['success_rate']:.2f}",
                    f"{metrics['avg_median_latency']:.4f}",
                    f"{metrics['min_median_latency']:.4f}",
                    f"{metrics['max_median_latency']:.4f}",
                    f"{metrics['avg_p95_latency']:.4f}",
                    f"{metrics['avg_throughput']:.2f}",
                    f"{metrics['min_throughput']:.2f}",
                    f"{metrics['max_throughput']:.2f}",
                    f"{metrics['avg_input_tokens']:.1f}",
                    f"{metrics['avg_output_tokens']:.1f}",
                    metrics['total_successful_requests'] + metrics['total_failed_requests'],
                    metrics['total_successful_requests'],
                    metrics['total_failed_requests']
                ])

        print(f"CSV summary saved to: {output_path}")


def main():
    """Main function to generate summary statistics."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate summary statistics from LLM endpoint benchmark results'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default='comparison_results/comparison_results/results.json',
        help='Path to results.json file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='comparison_results',
        help='Directory to save summary outputs'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Save summary as JSON file'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Save summary as CSV file'
    )
    parser.add_argument(
        '--no-print',
        action='store_true',
        help='Skip printing summary to console'
    )

    args = parser.parse_args()

    # Check if results file exists
    if not os.path.exists(args.results_path):
        print(f"Error: Results file not found at {args.results_path}")
        return 1

    # Generate summary
    print(f"Loading results from: {args.results_path}")
    generator = SummaryStatsGenerator(args.results_path)
    summary = generator.generate_summary()

    # Print to console
    if not args.no_print:
        generator.print_summary(summary)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    if args.json:
        json_path = os.path.join(args.output_dir, 'summary_statistics.json')
        generator.save_summary_json(summary, json_path)

    if args.csv:
        csv_path = os.path.join(args.output_dir, 'summary_statistics.csv')
        generator.save_summary_csv(summary, csv_path)

    return 0


if __name__ == '__main__':
    exit(main())
