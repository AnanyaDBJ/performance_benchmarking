"""
PDF report generation from benchmark results.

Adapted from generate_pdf_report.py to work with in-memory data and return
PDF bytes for streaming download.
"""

from __future__ import annotations

import io
import statistics
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Numbered canvas for page footers
# ---------------------------------------------------------------------------


class _NumberedCanvas(canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self) -> None:  # noqa: N802
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.setFont("Helvetica", 9)
            self.drawRightString(
                7.5 * inch, 0.5 * inch, f"Page {self._pageNumber} of {num_pages}"
            )
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


# ---------------------------------------------------------------------------
# Chart generation helpers (matplotlib -> bytes)
# ---------------------------------------------------------------------------


def _generate_comparison_chart_bytes(
    results: list[dict], qps: float, num_workers: int, out_tokens: int
) -> bytes | None:
    """Generate a single comparison chart as PNG bytes."""
    if not results or len(results) < 1:
        return None

    results = sorted(results, key=lambda x: x["endpoint_name"])
    endpoint_names = [r["endpoint_name"] for r in results]
    median_latencies = [r["median_latency"] for r in results]
    p95_latencies = [r["p95_latency"] for r in results]
    throughputs = [r["throughput"] for r in results]
    failures = [r["failed_requests"] for r in results]

    color_sets = {
        "latency": ["#e74c3c", "#f39c12", "#9b59b6", "#e67e22"],
        "p95": ["#c0392b", "#d68910", "#8e44ad", "#d35400"],
        "throughput": ["#3498db", "#27ae60", "#16a085", "#2980b9"],
        "failure": ["#e84118", "#fbc531", "#9c88ff", "#ff9ff3"],
    }

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Endpoint Comparison\nQPS={qps} | {num_workers} Workers | {out_tokens} Output Tokens",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    x_pos = range(len(endpoint_names))
    bar_width = 0.6

    for ax_idx, (metric, values, label, color_key, better) in enumerate(
        [
            (1, median_latencies, "Median Latency (s)", "latency", "lower"),
            (2, p95_latencies, "P95 Latency (s)", "p95", "lower"),
            (3, throughputs, "Throughput (tok/s)", "throughput", "higher"),
            (4, failures, "Failed Requests", "failure", "lower"),
        ],
        1,
    ):
        ax = plt.subplot(2, 2, ax_idx)
        bar_colors = [color_sets[color_key][i % 4] for i in range(len(endpoint_names))]
        bars = ax.bar(x_pos, values, bar_width, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Endpoint", fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        direction = "Lower" if better == "lower" else "Higher"
        ax.set_title(f"{label} ({direction} is Better)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(endpoint_names, rotation=15, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        for bar, val in zip(bars, values):
            fmt = f"{val:.2f}" if isinstance(val, float) and val < 100 else f"{int(val)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PDF Report generator
# ---------------------------------------------------------------------------


class PDFReportGenerator:
    """Generate a PDF report from in-memory benchmark results."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SubTitle",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#2c5aa0"),
                spaceAfter=12,
                spaceBefore=12,
                fontName="Helvetica-Bold",
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=10,
                spaceBefore=15,
                fontName="Helvetica-Bold",
                borderWidth=1,
                borderColor=colors.HexColor("#1f4788"),
                borderPadding=5,
                backColor=colors.HexColor("#e8f0fe"),
            )
        )

    # ----- public API -----

    def generate_pdf_bytes(self) -> bytes:
        """Return the full PDF report as bytes."""
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=1 * inch,
        )

        story: list[Any] = []
        story.extend(self._title_page())
        story.append(PageBreak())
        story.extend(self._executive_summary())
        story.append(PageBreak())
        story.extend(self._overall_metrics())
        story.append(Spacer(1, 0.3 * inch))
        story.extend(self._performance_rankings())
        story.append(PageBreak())
        story.extend(self._endpoint_details())
        story.append(PageBreak())
        story.extend(self._worker_analysis())
        story.append(PageBreak())
        story.extend(self._chart_pages())

        doc.build(story, canvasmaker=_NumberedCanvas)
        return buf.getvalue()

    # ----- private builders -----

    def _title_page(self) -> list:
        elements: list[Any] = [Spacer(1, 2 * inch)]
        elements.append(Paragraph("LLM Endpoint Benchmark Report", self.styles["CustomTitle"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Performance Analysis &amp; Comparison", self.styles["SubTitle"]))
        elements.append(Spacer(1, 1 * inch))

        timestamp = self.data.get("timestamp", "N/A")
        endpoints_raw = self.data.get("endpoints", [])
        if endpoints_raw and isinstance(endpoints_raw[0], dict):
            endpoints_str = ", ".join(ep.get("name", ep.get("endpoint", "?")) for ep in endpoints_raw)
        else:
            endpoints_str = ", ".join(endpoints_raw) if endpoints_raw else "N/A"

        meta = [
            ["Report Date:", datetime.now().strftime("%B %d, %Y")],
            ["Benchmark Date:", timestamp.split("T")[0] if "T" in str(timestamp) else str(timestamp)],
            ["Endpoints Tested:", endpoints_str],
            ["Total Configurations:", str(len(self.data.get("results", [])))],
        ]
        table = Table(meta, colWidths=[2 * inch, 4 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                    ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1f4788")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#2c5aa0")),
                ]
            )
        )
        elements.append(table)
        return elements

    def _executive_summary(self) -> list:
        elements: list[Any] = []
        elements.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.2 * inch))

        results = self.data.get("results", [])
        if not results:
            elements.append(Paragraph("No results available.", self.styles["BodyText"]))
            return elements

        ep_metrics: dict[str, dict[str, list[float]]] = {}
        for r in results:
            ep = r["endpoint_name"]
            ep_metrics.setdefault(ep, {"latencies": [], "throughputs": []})
            ep_metrics[ep]["latencies"].append(r["median_latency"])
            ep_metrics[ep]["throughputs"].append(r["throughput"])

        best_ep = min(ep_metrics, key=lambda e: statistics.mean(ep_metrics[e]["latencies"]))
        best_lat = min(ep_metrics[best_ep]["latencies"])
        best_thr = max(ep_metrics[best_ep]["throughputs"])
        total_req = sum(r["total_requests"] for r in results)

        text = (
            f"<b>Benchmark Overview:</b><br/>"
            f"This report analyses {len(ep_metrics)} endpoint(s) across "
            f"{len(results)} configurations with {total_req} total requests.<br/><br/>"
            f"<b>Top Performer:</b> <b>{best_ep}</b> with best median latency "
            f"<b>{best_lat:.3f}s</b> and peak throughput <b>{best_thr:.0f} tok/s</b>."
        )
        elements.append(Paragraph(text, self.styles["BodyText"]))
        return elements

    def _overall_metrics(self) -> list:
        elements: list[Any] = []
        elements.append(Paragraph("Overall Performance Metrics", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.15 * inch))

        results = self.data.get("results", [])
        if not results:
            return elements

        total_req = sum(r["total_requests"] for r in results)
        total_ok = sum(r["successful_requests"] for r in results)
        total_fail = sum(r["failed_requests"] for r in results)

        data = [
            ["Metric", "Value"],
            ["Total Requests", str(total_req)],
            ["Success Rate", f"{total_ok / total_req * 100:.1f}%" if total_req else "N/A"],
            ["Failed Requests", str(total_fail)],
            ["Avg Median Latency", f"{statistics.mean([r['median_latency'] for r in results]):.3f}s"],
            ["Avg P95 Latency", f"{statistics.mean([r['p95_latency'] for r in results]):.3f}s"],
            ["Avg Throughput", f"{statistics.mean([r['throughput'] for r in results]):.0f} tok/s"],
        ]
        table = Table(data, colWidths=[2.5 * inch, 2.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#2c5aa0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f7ff")]),
                ]
            )
        )
        elements.append(table)
        return elements

    def _performance_rankings(self) -> list:
        elements: list[Any] = []
        elements.append(Paragraph("Performance Rankings", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.15 * inch))

        results = self.data.get("results", [])
        ep_avg: dict[str, dict[str, float]] = {}
        for r in results:
            ep = r["endpoint_name"]
            ep_avg.setdefault(ep, {"lats": [], "thrs": []})  # type: ignore[arg-type]
            ep_avg[ep]["lats"].append(r["median_latency"])  # type: ignore[union-attr]
            ep_avg[ep]["thrs"].append(r["throughput"])  # type: ignore[union-attr]

        ranked = {
            ep: {
                "latency": statistics.mean(v["lats"]),  # type: ignore[arg-type]
                "throughput": statistics.mean(v["thrs"]),  # type: ignore[arg-type]
            }
            for ep, v in ep_avg.items()
        }

        # Latency ranking
        elements.append(Paragraph("<b>Best Latency (Lower is Better)</b>", self.styles["Heading3"]))
        lat_data = [["Rank", "Endpoint", "Avg Median Latency"]]
        for i, (ep, m) in enumerate(sorted(ranked.items(), key=lambda x: x[1]["latency"]), 1):
            lat_data.append([str(i), ep, f"{m['latency']:.3f}s"])
        table = Table(lat_data, colWidths=[0.7 * inch, 3 * inch, 2 * inch])
        table.setStyle(self._ranking_style())
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        # Throughput ranking
        elements.append(Paragraph("<b>Best Throughput (Higher is Better)</b>", self.styles["Heading3"]))
        thr_data = [["Rank", "Endpoint", "Avg Throughput"]]
        for i, (ep, m) in enumerate(sorted(ranked.items(), key=lambda x: x[1]["throughput"], reverse=True), 1):
            thr_data.append([str(i), ep, f"{m['throughput']:.0f} tok/s"])
        table = Table(thr_data, colWidths=[0.7 * inch, 3 * inch, 2 * inch])
        table.setStyle(self._ranking_style())
        elements.append(table)
        return elements

    def _ranking_style(self) -> TableStyle:
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2c5aa0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ]
        )

    def _endpoint_details(self) -> list:
        elements: list[Any] = []
        elements.append(Paragraph("Endpoint Performance Details", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.15 * inch))

        results = self.data.get("results", [])
        by_ep: dict[str, list[dict]] = {}
        for r in results:
            by_ep.setdefault(r["endpoint_name"], []).append(r)

        for ep, ep_results in by_ep.items():
            elements.append(Paragraph(f"<b>{ep}</b>", self.styles["Heading3"]))
            data = [["Workers", "QPS", "Out Tokens", "Latency", "P95", "Throughput", "Requests"]]
            for r in ep_results:
                data.append([
                    str(r["num_workers"]),
                    str(r.get("qps", "")),
                    str(r.get("output_tokens", "")),
                    f"{r['median_latency']:.3f}s",
                    f"{r['p95_latency']:.3f}s",
                    f"{r['throughput']:.0f} tok/s",
                    f"{r['successful_requests']}/{r['total_requests']}",
                ])
            table = Table(data, colWidths=[0.8 * inch] * 7)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))
        return elements

    def _worker_analysis(self) -> list:
        elements: list[Any] = []
        elements.append(Paragraph("Worker Count Scaling Analysis", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.15 * inch))

        results = self.data.get("results", [])
        by_w: dict[int, list[dict]] = {}
        for r in results:
            by_w.setdefault(r["num_workers"], []).append(r)

        data = [["Workers", "Configs", "Avg Latency", "Avg Throughput", "Success Rate"]]
        for w in sorted(by_w):
            wr = by_w[w]
            total_req = sum(r["total_requests"] for r in wr)
            total_ok = sum(r["successful_requests"] for r in wr)
            data.append([
                str(w),
                str(len(wr)),
                f"{statistics.mean([r['median_latency'] for r in wr]):.3f}s",
                f"{statistics.mean([r['throughput'] for r in wr]):.0f} tok/s",
                f"{total_ok / total_req * 100:.1f}%" if total_req else "N/A",
            ])

        table = Table(data, colWidths=[1 * inch, 1 * inch, 1.5 * inch, 1.5 * inch, 1.3 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f7ff"), colors.white]),
                ]
            )
        )
        elements.append(table)
        return elements

    def _chart_pages(self) -> list:
        """Generate comparison charts and embed them in the PDF."""
        elements: list[Any] = []
        elements.append(Paragraph("Visual Performance Comparisons", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.15 * inch))

        results = self.data.get("results", [])
        if not results:
            elements.append(Paragraph("No results to chart.", self.styles["BodyText"]))
            return elements

        # Group results by (qps, workers, out_tokens)
        groups: dict[tuple, list[dict]] = {}
        for r in results:
            key = (r.get("qps", 0), r.get("num_workers", 0), r.get("output_tokens", 0))
            groups.setdefault(key, []).append(r)

        for (qps, workers, tokens), group_results in sorted(groups.items()):
            chart_bytes = _generate_comparison_chart_bytes(group_results, qps, workers, tokens)
            if chart_bytes:
                caption = f"QPS={qps} | Workers={workers} | Output Tokens={tokens}"
                elements.append(Paragraph(f"<b>{caption}</b>", self.styles["Normal"]))
                img_buf = io.BytesIO(chart_bytes)
                img = Image(img_buf, width=6.5 * inch, height=4 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2 * inch))

        return elements


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def generate_report_bytes(results: list[dict], meta: dict[str, Any]) -> bytes:
    """Generate a PDF report from benchmark results and metadata."""
    data = {**meta, "results": results}
    gen = PDFReportGenerator(data)
    return gen.generate_pdf_bytes()
