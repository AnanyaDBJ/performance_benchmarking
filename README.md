# LLM Benchmark Reporter

Professional toolkit for benchmarking LLM endpoints and generating executive reports.

---

## 📦 Quick Install

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package (recommended)
pip install -e .
```

---

## 🎯 What This Does

1. **Benchmark** LLM endpoints (test performance under load)
2. **Compare** multiple endpoints side-by-side
3. **Generate** professional PDF reports with charts and insights
4. **Export** statistics in CSV/JSON formats

---

## 🚀 Quick Start

### 1. Compare Two Endpoints

```bash
python3 src/compare_endpoints.py \
  --endpoint1 gpt-4o \
  --endpoint2 databricks-gpt-120b \
  --qps 1.0 \
  --num_workers 4
```

### 2. Generate PDF Report

```bash
# Simple way
./utils/generate_report.sh comparison_results

# Or with Python
python3 src/generate_pdf_report.py --folder comparison_results
```

### 3. Get Statistics

```bash
python3 src/generate_summary_stats.py --csv --json
```

**Output:** PDF report + CSV/JSON data in `comparison_results/`

---

## 📁 Repository Structure

```
llm-benchmark-reporter/
│
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── setup.py                       # Package config
│
├── src/                           # BENCHMARKING & REPORTING SCRIPTS
│   ├── benchmark_llm.py           # Test single endpoint
│   ├── compare_endpoints.py       # Compare 2 endpoints
│   ├── compare_multi_endpoints.py # Compare 2+ endpoints
│   ├── generate_pdf_report.py     # Create PDF reports
│   └── generate_summary_stats.py  # Export CSV/JSON stats
│
├── utils/                         # UTILITY SCRIPTS
│   ├── monitor_live.py            # Real-time monitoring
│   ├── check_activity.sh          # Check activity
│   └── generate_report.sh         # Quick report wrapper
│
└── reports/                       # OUTPUT
    └── comparison_results/        # Results saved here
```

---

## 📖 Core Scripts Reference

### Benchmark Single Endpoint

**Script:** `src/benchmark_llm.py`
**Use when:** Testing one endpoint's performance

```bash
python3 src/benchmark_llm.py \
  --endpoint your-endpoint \
  --qps 1.0 \
  --num_workers 4 \
  --max_output_tokens 200
```

**Key parameters:**
- `--endpoint`: Endpoint name/URL
- `--qps`: Queries per second (0.25, 0.5, 1.0, 2.0, etc.)
- `--num_workers`: Parallel workers (2, 4, 8, 16)
- `--max_output_tokens`: Max response tokens

---

### Compare Two Endpoints

**Script:** `src/compare_endpoints.py`
**Use when:** Deciding between two models

```bash
python3 src/compare_endpoints.py \
  --endpoint1 model-a \
  --endpoint2 model-b \
  --qps 0.5 1.0 \
  --num_workers 2 4 \
  --max_output_tokens 200
```

---

### Compare Multiple Endpoints

**Script:** `src/compare_multi_endpoints.py`
**Use when:** Evaluating 3+ models at once

```bash
python3 src/compare_multi_endpoints.py \
  --endpoints model1 model2 model3 model4 \
  --qps 0.5 1.0 2.0 \
  --num_workers 2 4 8 \
  --max_output_tokens 200 500 \
  --runs 3
```

**Common parameters:**
- `--endpoints`: Space-separated list of endpoints
- `--qps`: List of QPS values to test
- `--num_workers`: List of worker counts
- `--max_output_tokens`: List of token limits
- `--runs`: Number of test iterations (default: 1)
- `--output_dir`: Where to save results (default: comparison_results)

---

### Generate PDF Report

**Script:** `src/generate_pdf_report.py`
**Use when:** You want a professional report for stakeholders

```bash
# Basic usage
python3 src/generate_pdf_report.py --folder comparison_results

# Custom output location
python3 src/generate_pdf_report.py -f my_results -o reports/report.pdf

# Different folder structure
python3 src/generate_pdf_report.py \
  --results-path /path/to/results.json \
  --images-dir /path/to/images
```

**Report includes:**
- Executive summary with recommendations
- Performance rankings (⭐⭐⭐⭐⭐ ratings)
- Detailed metrics by endpoint
- Scaling analysis (how workers affect performance)
- All benchmark charts

**Quick wrapper:**
```bash
./utils/generate_report.sh my_results
```

---

### Export Statistics

**Script:** `src/generate_summary_stats.py`
**Use when:** You need data for spreadsheets/analysis

```bash
# Console output only
python3 src/generate_summary_stats.py

# Export to CSV
python3 src/generate_summary_stats.py --csv

# Export to JSON
python3 src/generate_summary_stats.py --json

# Both formats
python3 src/generate_summary_stats.py --csv --json
```

**Outputs:**
- Console: Formatted tables and rankings
- CSV: `summary_statistics.csv` (spreadsheet-ready)
- JSON: `summary_statistics.json` (programmatic access)

---

### Utilities

**Monitor live performance:**
```bash
python3 utils/monitor_live.py --endpoint your-endpoint
```

**Check activity:**
```bash
./utils/check_activity.sh
```

---

## 🎓 Common Usage Patterns

### Pattern 1: Quick Performance Check
```bash
# Run benchmark
python3 src/benchmark_llm.py --endpoint my-endpoint --qps 1.0 --num_workers 4

# Get report
./utils/generate_report.sh comparison_results
```

### Pattern 2: Compare Models for Procurement
```bash
# Test 4 vendors
python3 src/compare_multi_endpoints.py \
  --endpoints vendor-a vendor-b vendor-c vendor-d \
  --qps 0.5 1.0 2.0 \
  --num_workers 2 4 \
  --runs 5

# Generate decision report
python3 src/generate_pdf_report.py --folder comparison_results
python3 src/generate_summary_stats.py --csv
```

### Pattern 3: Production Readiness Test
```bash
# Stress test at expected load
python3 src/benchmark_llm.py \
  --endpoint prod-endpoint \
  --qps 5.0 \
  --num_workers 16 \
  --max_output_tokens 500 \
  --runs 10

# Get detailed analysis
python3 src/generate_summary_stats.py --csv --json
```

### Pattern 4: A/B Test New Model
```bash
# Compare old vs new
python3 src/compare_endpoints.py \
  --endpoint1 current-prod-v1 \
  --endpoint2 candidate-v2 \
  --qps 1.0 2.0 5.0 \
  --num_workers 4 8 \
  --runs 10

# Review results
python3 src/generate_pdf_report.py --folder comparison_results
```

### Pattern 5: Daily Automated Testing
```bash
#!/bin/bash
# daily_benchmark.sh

DATE=$(date +%Y%m%d)

python3 src/compare_multi_endpoints.py \
  --endpoints ep1 ep2 \
  --qps 1.0 \
  --num_workers 4 \
  --output_dir "results_${DATE}"

python3 src/generate_pdf_report.py \
  --folder "results_${DATE}" \
  --output "reports/daily_${DATE}.pdf"
```

---

## ⚙️ Configuration

### Folder Structures Supported

**Flat structure:**
```
my_results/
├── results.json
└── *.png
```

**Nested structure (default):**
```
comparison_results/
└── comparison_results/
    ├── results.json
    └── *.png
```

**Custom structure:**
```
test/
└── data/
    ├── results.json
    └── charts/*.png
```

For custom structures, use:
```bash
python3 src/generate_pdf_report.py \
  --folder test \
  --results-subdir data \
  --images-dir test/data/charts
```

---

## 📊 Understanding Output

### Sample Console Output
```
🏆 Performance Rankings:

Lowest Latency (Best to Worst):
  1. gpt-oss-20b: 1.474s ⭐⭐⭐⭐⭐
  2. gpt-oss-120b: 1.774s ⭐⭐⭐⭐
  3. gpt-4-mini: 1.888s ⭐⭐⭐
  4. gpt-4o: 2.824s ⭐⭐

Highest Throughput:
  1. gpt-oss-20b: 2,242 tokens/s ⭐⭐⭐⭐⭐
  2. gpt-oss-120b: 1,996 tokens/s ⭐⭐⭐⭐

✅ Success Rate: 100% (24/24 requests)
⚡ Scaling: 127% throughput gain with 4 workers
```

### Key Metrics Explained

- **QPS**: Queries Per Second - request rate
- **Workers**: Number of parallel processes
- **Median Latency**: Middle response time (50th percentile)
- **P95 Latency**: 95% of requests faster than this
- **Throughput**: Tokens processed per second
- **Success Rate**: Percentage of successful requests

---

## 🔧 Troubleshooting

### "Results file not found"
```bash
# Check what's in your folder
ls -la comparison_results/

# Specify exact path
python3 src/generate_pdf_report.py --results-path /full/path/to/results.json
```

### "reportlab is required"
```bash
python3 -m pip install reportlab pillow
```

### High failure rates
- Reduce QPS or workers
- Increase timeout
- Check endpoint health

### Connection errors
- Verify endpoint URL in your command arguments
- Check API tokens are valid and not expired
- Ensure your workspace URL is correct

---

## 📦 Package Installation

Install as a reusable CLI tool:

```bash
# Install
pip install -e .

# Use from anywhere
llm-benchmark-report --folder my_results
llm-benchmark-stats --csv --json
```

---

## 🎯 Use Cases

| Scenario | Script to Use |
|----------|---------------|
| Quick health check | `src/benchmark_llm.py` |
| Compare 2 models | `src/compare_endpoints.py` |
| Evaluate 3+ models | `src/compare_multi_endpoints.py` |
| Vendor selection | `src/compare_multi_endpoints.py` + PDF report |
| Production readiness | `src/benchmark_llm.py` with high QPS/workers |
| A/B testing | `src/compare_endpoints.py` |
| Daily monitoring | Automate with cron + `utils/generate_report.sh` |
| Cost optimization | Compare different model sizes |

---

## 💡 Pro Tips

1. **Start small**: Begin with low QPS (0.5) and few workers (2)
2. **Run multiple iterations**: Use `--runs 5` for statistical confidence
3. **Save results**: Always use `--output_dir` to preserve data
4. **Archive reports**: Keep PDFs by date for trend analysis
5. **Test gradually**: Increase load step-by-step to find limits

---

## 📋 Requirements

- Python 3.8+
- reportlab >= 4.0.0
- pillow >= 9.0.0

Install: `pip install -r requirements.txt`

---

## 🔄 Typical Workflow

```
1. Run Benchmark
   └─> compare_multi_endpoints.py
       └─> Outputs: results.json + charts/*.png

2. Generate Report
   └─> generate_pdf_report.py
       └─> Outputs: benchmark_report.pdf

3. Export Stats (optional)
   └─> generate_summary_stats.py
       └─> Outputs: CSV/JSON files

4. Review & Decide
   └─> Share PDF with stakeholders
   └─> Analyze CSV in Excel/Sheets
```

---

## 🤝 Support

- **Issues**: File a GitHub issue
- **Questions**: Check this README first
- **Help**: Run any script with `--help`

---

## 📄 License

MIT License

---

**That's it!** Everything you need is in this README. Start with the Quick Start section above.

*Last Updated: February 2026*
