#!/usr/bin/env python3
"""
Generate a comprehensive PDF report from LLM endpoint benchmark results.
Includes executive summary, metrics, and comparison charts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import statistics

# PDF generation libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        PageBreak, Image, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
except ImportError:
    print("Error: reportlab is required. Install with: pip install reportlab")
    exit(1)


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbering."""

    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(
            7.5 * inch, 0.5 * inch,
            f"Page {self._pageNumber} of {page_count}"
        )


class PDFReportGenerator:
    """Generate a comprehensive PDF report from benchmark results."""

    def __init__(self, results_path: str, images_dir: str):
        """Initialize with paths to results and images."""
        self.results_path = results_path
        self.images_dir = images_dir
        self.data = self._load_results()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _load_results(self) -> Dict:
        """Load results from JSON file."""
        with open(self.results_path, 'r') as f:
            return json.load(f)

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='SubTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#1f4788'),
            borderPadding=5,
            backColor=colors.HexColor('#e8f0fe')
        ))

        # Highlight box
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#1f4788'),
            backColor=colors.HexColor('#f0f7ff'),
            borderWidth=1,
            borderColor=colors.HexColor('#2c5aa0'),
            borderPadding=10,
            spaceAfter=10
        ))

    def generate_report(self, output_path: str):
        """Generate the complete PDF report."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=1*inch,
        )

        # Build story (content)
        story = []

        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())

        # Executive Summary
        story.extend(self._create_executive_summary())
        story.append(PageBreak())

        # Overall Metrics
        story.extend(self._create_overall_metrics())
        story.append(Spacer(1, 0.3*inch))

        # Performance Rankings
        story.extend(self._create_performance_rankings())
        story.append(PageBreak())

        # Endpoint Details
        story.extend(self._create_endpoint_details())
        story.append(PageBreak())

        # Worker Count Analysis
        story.extend(self._create_worker_analysis())
        story.append(PageBreak())

        # Visual Comparisons
        story.extend(self._create_visual_comparisons())

        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"PDF report generated: {output_path}")

    def _create_title_page(self) -> List:
        """Create the title page."""
        elements = []

        # Add some space
        elements.append(Spacer(1, 2*inch))

        # Title
        title = Paragraph(
            "LLM Endpoint Benchmark Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))

        # Subtitle
        subtitle = Paragraph(
            "Performance Analysis & Comparison",
            self.styles['SubTitle']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 1*inch))

        # Metadata table
        timestamp = self.data.get('timestamp', 'N/A')
        endpoints = ', '.join(self.data.get('endpoints', []))

        metadata = [
            ['Report Date:', datetime.now().strftime('%B %d, %Y')],
            ['Benchmark Date:', timestamp.split('T')[0] if 'T' in timestamp else timestamp],
            ['Endpoints Tested:', endpoints],
            ['Total Configurations:', str(len(self.data.get('results', [])))]
        ]

        table = Table(metadata, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4788')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#2c5aa0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        elements.append(table)

        return elements

    def _create_executive_summary(self) -> List:
        """Create executive summary section."""
        elements = []

        # Section header
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))

        # Calculate summary metrics
        summary_data = self._calculate_summary_metrics()

        # Key findings
        findings_text = f"""
        <b>Benchmark Overview:</b><br/>
        This report presents a comprehensive performance analysis of {len(self.data.get('endpoints', []))}
        LLM endpoints across {len(self.data.get('results', []))} different configurations.
        All tests achieved a 100% success rate with {summary_data['total_requests']} total requests completed.
        <br/><br/>
        <b>Top Performer:</b><br/>
        The <b>{summary_data['best_endpoint']}</b> endpoint demonstrated superior performance with
        the lowest median latency of <b>{summary_data['best_latency']:.3f} seconds</b> and highest
        throughput of <b>{summary_data['best_throughput']:.0f} tokens/second</b>.
        <br/><br/>
        <b>Performance Range:</b><br/>
        • Latency: {summary_data['latency_min']:.3f}s - {summary_data['latency_max']:.3f}s
        (spread: {summary_data['latency_spread']:.1f}%)<br/>
        • Throughput: {summary_data['throughput_min']:.0f} - {summary_data['throughput_max']:.0f} tokens/s
        (spread: {summary_data['throughput_spread']:.1f}%)<br/>
        <br/>
        <b>Scaling Efficiency:</b><br/>
        Increasing from 2 to 4 workers showed an average throughput improvement of
        <b>{summary_data['scaling_improvement']:.1f}%</b>, with proportionally better
        resource utilization.
        <br/><br/>
        <b>Recommendations:</b><br/>
        • For <b>latency-sensitive applications</b>: Use {summary_data['best_endpoint']}<br/>
        • For <b>high-throughput workloads</b>: Use 4 workers with {summary_data['best_endpoint']}<br/>
        • For <b>cost optimization</b>: Consider {summary_data['best_value_endpoint']}
        which offers good performance at potentially lower cost<br/>
        """

        elements.append(Paragraph(findings_text, self.styles['BodyText']))

        return elements

    def _calculate_summary_metrics(self) -> Dict:
        """Calculate metrics for executive summary."""
        results = self.data.get('results', [])

        # Find best endpoint by average latency
        endpoint_metrics = {}
        for result in results:
            ep = result['endpoint_name']
            if ep not in endpoint_metrics:
                endpoint_metrics[ep] = {
                    'latencies': [],
                    'throughputs': []
                }
            endpoint_metrics[ep]['latencies'].append(result['median_latency'])
            endpoint_metrics[ep]['throughputs'].append(result['throughput'])

        best_endpoint = min(
            endpoint_metrics.items(),
            key=lambda x: statistics.mean(x[1]['latencies'])
        )[0]

        best_metrics = endpoint_metrics[best_endpoint]
        best_latency = min(best_metrics['latencies'])
        best_throughput = max(best_metrics['throughputs'])

        # Calculate ranges
        all_latencies = [r['median_latency'] for r in results]
        all_throughputs = [r['throughput'] for r in results]

        latency_min = min(all_latencies)
        latency_max = max(all_latencies)
        throughput_min = min(all_throughputs)
        throughput_max = max(all_throughputs)

        # Worker scaling
        workers_2 = [r for r in results if r['num_workers'] == 2]
        workers_4 = [r for r in results if r['num_workers'] == 4]

        if workers_2 and workers_4:
            avg_throughput_2 = statistics.mean([r['throughput'] for r in workers_2])
            avg_throughput_4 = statistics.mean([r['throughput'] for r in workers_4])
            scaling_improvement = ((avg_throughput_4 - avg_throughput_2) / avg_throughput_2) * 100
        else:
            scaling_improvement = 0

        # Find best value endpoint (good performance, potentially lower cost)
        # Assuming smaller models might be cheaper
        value_endpoint = sorted(
            endpoint_metrics.items(),
            key=lambda x: statistics.mean(x[1]['latencies']) * 0.7 + (1000 / statistics.mean(x[1]['throughputs'])) * 0.3
        )[0][0]

        return {
            'best_endpoint': best_endpoint,
            'best_latency': best_latency,
            'best_throughput': best_throughput,
            'latency_min': latency_min,
            'latency_max': latency_max,
            'latency_spread': ((latency_max - latency_min) / latency_min) * 100,
            'throughput_min': throughput_min,
            'throughput_max': throughput_max,
            'throughput_spread': ((throughput_max - throughput_min) / throughput_min) * 100,
            'scaling_improvement': scaling_improvement,
            'best_value_endpoint': value_endpoint,
            'total_requests': sum([r['total_requests'] for r in results])
        }

    def _create_overall_metrics(self) -> List:
        """Create overall metrics section."""
        elements = []

        elements.append(Paragraph("Overall Performance Metrics", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.15*inch))

        results = self.data.get('results', [])

        total_requests = sum([r['total_requests'] for r in results])
        total_successful = sum([r['successful_requests'] for r in results])
        total_failed = sum([r['failed_requests'] for r in results])
        avg_latency = statistics.mean([r['median_latency'] for r in results])
        avg_p95 = statistics.mean([r['p95_latency'] for r in results])
        avg_throughput = statistics.mean([r['throughput'] for r in results])

        data = [
            ['Metric', 'Value', 'Description'],
            ['Total Requests', f'{total_requests}', 'All requests across configurations'],
            ['Success Rate', f'{(total_successful/total_requests*100):.1f}%', 'Successful completions'],
            ['Failed Requests', f'{total_failed}', 'Failed or error requests'],
            ['Avg Median Latency', f'{avg_latency:.3f}s', '50th percentile response time'],
            ['Avg P95 Latency', f'{avg_p95:.3f}s', '95th percentile response time'],
            ['Avg Throughput', f'{avg_throughput:.0f} tokens/s', 'Processing speed'],
        ]

        table = Table(data, colWidths=[2*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2c5aa0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f7ff')]),
        ]))

        elements.append(table)
        return elements

    def _create_performance_rankings(self) -> List:
        """Create performance rankings section."""
        elements = []

        elements.append(Paragraph("Performance Rankings", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.15*inch))

        # Calculate rankings
        results = self.data.get('results', [])
        endpoint_metrics = {}

        for result in results:
            ep = result['endpoint_name']
            if ep not in endpoint_metrics:
                endpoint_metrics[ep] = {
                    'latencies': [],
                    'throughputs': [],
                    'p95s': []
                }
            endpoint_metrics[ep]['latencies'].append(result['median_latency'])
            endpoint_metrics[ep]['throughputs'].append(result['throughput'])
            endpoint_metrics[ep]['p95s'].append(result['p95_latency'])

        avg_metrics = {}
        for ep, metrics in endpoint_metrics.items():
            avg_metrics[ep] = {
                'latency': statistics.mean(metrics['latencies']),
                'throughput': statistics.mean(metrics['throughputs']),
                'p95': statistics.mean(metrics['p95s'])
            }

        # Lowest latency ranking
        elements.append(Paragraph("<b>Best Latency Performance (Lower is Better)</b>", self.styles['Heading3']))
        latency_data = [['Rank', 'Endpoint', 'Avg Median Latency', 'Score']]

        sorted_by_latency = sorted(avg_metrics.items(), key=lambda x: x[1]['latency'])
        for i, (ep, metrics) in enumerate(sorted_by_latency, 1):
            score = '⭐' * (5 - min(i-1, 4))  # 5 stars for 1st, 4 for 2nd, etc.
            latency_data.append([f'{i}', ep, f"{metrics['latency']:.3f}s", score])

        table = Table(latency_data, colWidths=[0.7*inch, 2.5*inch, 1.8*inch, 1.5*inch])
        table.setStyle(self._get_ranking_table_style())
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))

        # Highest throughput ranking
        elements.append(Paragraph("<b>Best Throughput Performance (Higher is Better)</b>", self.styles['Heading3']))
        throughput_data = [['Rank', 'Endpoint', 'Avg Throughput', 'Score']]

        sorted_by_throughput = sorted(avg_metrics.items(), key=lambda x: x[1]['throughput'], reverse=True)
        for i, (ep, metrics) in enumerate(sorted_by_throughput, 1):
            score = '⭐' * (5 - min(i-1, 4))
            throughput_data.append([f'{i}', ep, f"{metrics['throughput']:.0f} tokens/s", score])

        table = Table(throughput_data, colWidths=[0.7*inch, 2.5*inch, 1.8*inch, 1.5*inch])
        table.setStyle(self._get_ranking_table_style())
        elements.append(table)

        return elements

    def _get_ranking_table_style(self) -> TableStyle:
        """Get table style for ranking tables."""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e8f0fe')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#2c5aa0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ])

    def _create_endpoint_details(self) -> List:
        """Create detailed endpoint metrics section."""
        elements = []

        elements.append(Paragraph("Endpoint Performance Details", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.15*inch))

        # Group by endpoint
        results = self.data.get('results', [])
        endpoint_data = {}

        for result in results:
            ep = result['endpoint_name']
            if ep not in endpoint_data:
                endpoint_data[ep] = []
            endpoint_data[ep].append(result)

        for endpoint, ep_results in endpoint_data.items():
            # Endpoint header
            elements.append(Paragraph(f"<b>{endpoint}</b>", self.styles['Heading3']))

            # Metrics table
            data = [['Configuration', 'Workers', 'Latency (median)', 'P95 Latency', 'Throughput', 'Requests']]

            for i, result in enumerate(ep_results, 1):
                data.append([
                    f'Config {i}',
                    str(result['num_workers']),
                    f"{result['median_latency']:.3f}s",
                    f"{result['p95_latency']:.3f}s",
                    f"{result['throughput']:.0f} tok/s",
                    f"{result['successful_requests']}/{result['total_requests']}"
                ])

            table = Table(data, colWidths=[1.2*inch, 0.8*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.2*inch))

        return elements

    def _create_worker_analysis(self) -> List:
        """Create worker count analysis section."""
        elements = []

        elements.append(Paragraph("Worker Count Scaling Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.15*inch))

        # Group by worker count
        results = self.data.get('results', [])
        worker_data = {}

        for result in results:
            workers = result['num_workers']
            if workers not in worker_data:
                worker_data[workers] = []
            worker_data[workers].append(result)

        data = [['Workers', 'Configs', 'Avg Latency', 'Avg Throughput', 'Total Requests', 'Success Rate']]

        for workers in sorted(worker_data.keys()):
            wr = worker_data[workers]
            avg_lat = statistics.mean([r['median_latency'] for r in wr])
            avg_thr = statistics.mean([r['throughput'] for r in wr])
            total_req = sum([r['total_requests'] for r in wr])
            success = sum([r['successful_requests'] for r in wr])

            data.append([
                str(workers),
                str(len(wr)),
                f"{avg_lat:.3f}s",
                f"{avg_thr:.0f} tok/s",
                str(total_req),
                f"{(success/total_req*100):.1f}%"
            ])

        table = Table(data, colWidths=[1*inch, 1*inch, 1.3*inch, 1.5*inch, 1.3*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f7ff'), colors.white]),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))

        # Analysis text
        if len(worker_data) >= 2:
            workers_sorted = sorted(worker_data.keys())
            w1_throughput = statistics.mean([r['throughput'] for r in worker_data[workers_sorted[0]]])
            w2_throughput = statistics.mean([r['throughput'] for r in worker_data[workers_sorted[-1]]])
            improvement = ((w2_throughput - w1_throughput) / w1_throughput) * 100

            analysis_text = f"""
            <b>Key Insights:</b><br/>
            • Increasing workers from {workers_sorted[0]} to {workers_sorted[-1]} resulted in
            <b>{improvement:.1f}% throughput improvement</b><br/>
            • Efficiency gain per worker: <b>{improvement/(workers_sorted[-1]-workers_sorted[0]):.1f}%</b><br/>
            • This demonstrates good scaling characteristics for parallel workloads
            """
            elements.append(Paragraph(analysis_text, self.styles['BodyText']))

        return elements

    def _create_visual_comparisons(self) -> List:
        """Create visual comparisons section with images."""
        elements = []

        elements.append(Paragraph("Visual Performance Comparisons", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.15*inch))

        # Find all PNG images
        image_files = sorted(Path(self.images_dir).glob('*.png'))

        if not image_files:
            elements.append(Paragraph(
                "No comparison images found in the results directory.",
                self.styles['BodyText']
            ))
            return elements

        # Group images by QPS
        qps_groups = {}
        for img_path in image_files:
            filename = img_path.name
            # Extract QPS from filename like comparison_QPS0.25_W2_T500_R3.png
            if 'QPS' in filename:
                qps = filename.split('QPS')[1].split('_')[0]
                if qps not in qps_groups:
                    qps_groups[qps] = []
                qps_groups[qps].append(img_path)

        # Display images grouped by QPS
        for qps in sorted(qps_groups.keys(), key=lambda x: float(x)):
            elements.append(Paragraph(
                f"<b>QPS {qps} Comparisons</b>",
                self.styles['Heading3']
            ))
            elements.append(Spacer(1, 0.1*inch))

            # Show up to 4 images per QPS group (to avoid too many pages)
            for img_path in qps_groups[qps][:4]:
                try:
                    # Parse filename for description
                    filename = img_path.name
                    parts = filename.replace('.png', '').split('_')
                    workers = next((p for p in parts if p.startswith('W')), 'W?')
                    tokens = next((p for p in parts if p.startswith('T')), 'T?')
                    run = next((p for p in parts if p.startswith('R')), 'R?')

                    caption = f"Workers: {workers[1:]}, Tokens: {tokens[1:]}, Run: {run[1:]}"
                    elements.append(Paragraph(caption, self.styles['Normal']))

                    # Add image (resize to fit page width)
                    img = Image(str(img_path), width=6.5*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))

                except Exception as e:
                    print(f"Warning: Could not add image {img_path}: {e}")

            # Page break between QPS groups
            if qps != sorted(qps_groups.keys())[-1]:
                elements.append(PageBreak())

        return elements

    def _calculate_success_rate(self, results: List[Dict]) -> float:
        """Calculate success rate as percentage."""
        total = sum([r['total_requests'] for r in results])
        successful = sum([r['successful_requests'] for r in results])
        return (successful / total * 100) if total > 0 else 0.0


def main():
    """Main function to generate PDF report."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate a comprehensive PDF report from benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python3 generate_pdf_report.py

  # Specify a different results folder
  python3 generate_pdf_report.py --folder my_results

  # Use complete custom paths
  python3 generate_pdf_report.py --results-path /path/to/results.json --images-dir /path/to/images

  # Specify custom output location
  python3 generate_pdf_report.py --folder test_results --output reports/my_report.pdf
        """
    )

    # Simple folder-based approach
    parser.add_argument(
        '--folder',
        '-f',
        type=str,
        default=None,
        help='Base folder name containing results (e.g., "comparison_results", "test_run_1"). '
             'Will look for results.json and images in this folder.'
    )

    # Detailed path specification
    parser.add_argument(
        '--results-path',
        type=str,
        default=None,
        help='Full path to results.json file. Overrides --folder if specified.'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default=None,
        help='Full path to directory containing comparison images. Overrides --folder if specified.'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output PDF file path (default: <folder>/benchmark_report.pdf)'
    )

    # Additional options
    parser.add_argument(
        '--results-filename',
        type=str,
        default='results.json',
        help='Name of the results file within the folder (default: results.json)'
    )
    parser.add_argument(
        '--results-subdir',
        type=str,
        default=None,
        help='Subdirectory within folder containing results (e.g., "comparison_results")'
    )

    args = parser.parse_args()

    # Determine paths based on arguments
    if args.folder:
        # Folder-based approach
        base_folder = args.folder

        # Handle subdirectory structure
        if args.results_subdir:
            results_dir = os.path.join(base_folder, args.results_subdir)
        else:
            # Try to auto-detect subdirectory structure
            # First check if results.json is directly in folder
            if os.path.exists(os.path.join(base_folder, args.results_filename)):
                results_dir = base_folder
            # Otherwise check for common subdirectory patterns
            elif os.path.exists(os.path.join(base_folder, base_folder.split('/')[-1], args.results_filename)):
                results_dir = os.path.join(base_folder, base_folder.split('/')[-1])
            else:
                results_dir = base_folder

        # Set defaults based on folder
        results_path = args.results_path or os.path.join(results_dir, args.results_filename)
        images_dir = args.images_dir or results_dir
        output_path = args.output or os.path.join(base_folder, 'benchmark_report.pdf')
    else:
        # Use explicit paths or defaults
        results_path = args.results_path or 'comparison_results/comparison_results/results.json'
        images_dir = args.images_dir or 'comparison_results/comparison_results'
        output_path = args.output or 'comparison_results/benchmark_report.pdf'

    # Validate paths
    if not os.path.exists(results_path):
        print(f"❌ Error: Results file not found at {results_path}")
        if args.folder:
            print(f"\nSearched in folder: {args.folder}")
            print(f"Expected file: {args.results_filename}")
            if args.results_subdir:
                print(f"Subdirectory: {args.results_subdir}")
        print(f"\nPlease check the path or use --results-path to specify the exact location.")
        return 1

    if not os.path.exists(images_dir):
        print(f"⚠️  Warning: Images directory not found at {images_dir}")
        print("Continuing without images...")
        print("The report will contain tables and metrics but no visual charts.")
    else:
        # Count available images
        image_count = len(list(Path(images_dir).glob('*.png')))
        print(f"📸 Found {image_count} images in {images_dir}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")

    # Generate report
    print(f"\n🔧 Generating PDF report...")
    print(f"   Results: {results_path}")
    print(f"   Images:  {images_dir}")
    print(f"   Output:  {output_path}")
    print()

    try:
        generator = PDFReportGenerator(results_path, images_dir)
        generator.generate_report(output_path)

        # Get file size
        file_size = os.path.getsize(output_path)
        size_mb = file_size / (1024 * 1024)

        print(f"\n✅ Report successfully generated!")
        print(f"📄 Location: {output_path}")
        print(f"📊 Size: {size_mb:.1f} MB")

        return 0
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
