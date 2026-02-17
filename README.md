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

### Prerequisites

Before running benchmarks, you need:
1. **API Root URL** (hostname): Your Databricks workspace URL (e.g., `https://your-workspace.cloud.databricks.com`)
2. **API Token**: Your Databricks personal access token
3. **Endpoint Name**: The name of your LLM serving endpoint

### 1. Test a Single Endpoint

```bash
export DATABRICKS_API_TOKEN="dapi..."

python3 src/benchmark_llm.py \
  --endpoint your-endpoint-name \
  --api-token-env DATABRICKS_API_TOKEN \
  --api-root https://your-workspace.cloud.databricks.com \
  --qps-list 0.5 1.0 \
  --parallel-workers 4 6 \
  --output-dir my_benchmark_results
```

**Note:** Using `--output-dir` saves results JSON AND generates performance charts (PNG images) that can be used for PDF reports.

### 2. Compare Multiple Endpoints (2-4)

```bash
export API_ROOT="https://your-workspace.cloud.databricks.com"
export API_TOKEN="dapi..."

python3 src/compare_multi_endpoints.py \
  --endpoint-1 gpt-model-a \
  --endpoint-1-name "GPT Model A" \
  --api-token-1 "$API_TOKEN" \
  --api-root-1 "$API_ROOT" \
  --endpoint-2 gpt-model-b \
  --endpoint-2-name "GPT Model B" \
  --api-token-2 "$API_TOKEN" \
  --api-root-2 "$API_ROOT" \
  --qps-list 0.5 1.0 \
  --parallel-workers 4 8
```

### 3. Compare 3-4 Endpoints

```bash
export API_ROOT="https://your-workspace.cloud.databricks.com"
export API_TOKEN="dapi..."

python3 src/compare_multi_endpoints.py \
  --endpoint-1 model-a \
  --endpoint-1-name "Model A" \
  --api-token-1 "$API_TOKEN" \
  --api-root-1 "$API_ROOT" \
  --endpoint-2 model-b \
  --endpoint-2-name "Model B" \
  --api-token-2 "$API_TOKEN" \
  --api-root-2 "$API_ROOT" \
  --endpoint-3 model-c \
  --endpoint-3-name "Model C" \
  --api-token-3 "$API_TOKEN" \
  --api-root-3 "$API_ROOT" \
  --endpoint-4 model-d \
  --endpoint-4-name "Model D" \
  --api-token-4 "$API_TOKEN" \
  --api-root-4 "$API_ROOT" \
  --qps-list 0.5 1.0 \
  --parallel-workers 4 8 \
  --output-tokens 200 500

# Generate PDF report (note: multi-endpoint saves to all_results.json)
python3 src/generate_pdf_report.py \
  --folder multi_endpoint_comparison \
  --results-filename all_results.json
```

### 4. Generate PDF Report

```bash
# Simple way
./utils/generate_report.sh comparison_results

# Or with Python
python3 src/generate_pdf_report.py --folder comparison_results
```

### 5. Get Statistics

```bash
python3 src/generate_summary_stats.py --csv --json
```

**Output:** PDF report + CSV/JSON data in results folder

---

## ⚡ Complete Examples (Copy & Paste)

Replace these placeholder values before running:
- `YOUR_TOKEN` → Your Databricks API token (e.g., `dapi1234567890abcdef`)
- `YOUR_WORKSPACE` → Your workspace URL (e.g., `https://your-workspace.cloud.databricks.com`)
- `ENDPOINT_NAME` → Your endpoint name (e.g., `gpt-4o-mini`)

### Example 1: Test a Single Endpoint (Simplest)
```bash
export DATABRICKS_API_TOKEN="YOUR_TOKEN"

python3 src/benchmark_llm.py \
  --endpoint ENDPOINT_NAME \
  --api-token-env DATABRICKS_API_TOKEN \
  --api-root YOUR_WORKSPACE
```

### Example 2: Compare Two Models
```bash
export API_ROOT="YOUR_WORKSPACE"
export API_TOKEN="YOUR_TOKEN"

python3 src/compare_multi_endpoints.py \
  --endpoint-1 gpt-4o-mini \
  --endpoint-1-name "GPT-4o Mini" \
  --api-token-1 "$API_TOKEN" \
  --api-root-1 "$API_ROOT" \
  --endpoint-2 gpt-4o \
  --endpoint-2-name "GPT-4o" \
  --api-token-2 "$API_TOKEN" \
  --api-root-2 "$API_ROOT" \
  --qps-list 0.5 1.0 \
  --parallel-workers 4 8
```

### Example 3: Compare 4 Models with Custom Settings
```bash
export API_ROOT="YOUR_WORKSPACE"
export API_TOKEN="YOUR_TOKEN"

python3 src/compare_multi_endpoints.py \
  --endpoint-1 gpt-4o-mini \
  --endpoint-1-name "GPT-4o Mini" \
  --api-token-1 "$API_TOKEN" \
  --api-root-1 "$API_ROOT" \
  --endpoint-2 gpt-4o \
  --endpoint-2-name "GPT-4o" \
  --api-token-2 "$API_TOKEN" \
  --api-root-2 "$API_ROOT" \
  --endpoint-3 claude-3-5-sonnet \
  --endpoint-3-name "Claude 3.5 Sonnet" \
  --api-token-3 "$API_TOKEN" \
  --api-root-3 "$API_ROOT" \
  --endpoint-4 llama-3-70b \
  --endpoint-4-name "Llama 3 70B" \
  --api-token-4 "$API_TOKEN" \
  --api-root-4 "$API_ROOT" \
  --qps-list 0.5 1.0 \
  --parallel-workers 4 8 \
  --output-tokens 200 500 \
  --output-dir model_comparison_2024
```

### Example 4: Using Environment Variables (Recommended)
```bash
# First, set environment variables (add to ~/.bashrc or ~/.zshrc for persistence)
export API_ROOT="https://your-workspace.cloud.databricks.com"
export DATABRICKS_API_TOKEN="dapi1234567890abcdef"

# Now you can omit --api-root and --api-token
python3 src/benchmark_llm.py \
  --endpoint my-endpoint \
  --api-token-env DATABRICKS_API_TOKEN \
  --api-root $API_ROOT
```

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
│   ├── compare_multi_endpoints.py # Compare 2-4 endpoints
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
  --endpoint your-endpoint-name \
  --api-token-env DATABRICKS_API_TOKEN \
  --api-root https://your-workspace.cloud.databricks.com \
  --input-tokens 500 1000 1500 \
  --output-tokens 200 500 1000 \
  --qps-list 0.5 1.0 2.0 \
  --parallel-workers 4 6 8 \
  --requests-per-worker 5 \
  --timeout 300 \
  --max-retries 3 \
  --output-dir benchmark_results
```

**Required parameters:**
- `--endpoint`: Your LLM serving endpoint name (NOT the full URL)
- One of: `--api-token`, `--api-token-env`, `--api-token-stdin`, or `DATABRICKS_API_TOKEN` env var

**Optional parameters:**
- `--api-root`: Databricks workspace URL (default: `https://your-workspace.cloud.databricks.com`)
- `--input-tokens`: List of input token sizes to test (default: 15000). You can pass multiple values like `--input-tokens 500 1000 1500`
- `--api-token-env`: Read token from a named env var (recommended)
- `--api-token-stdin`: Read token from stdin first line (recommended for automation)
- `--output-tokens`: List of output token sizes to test (default: 200 500 1000)
- `--qps-list`: List of queries per second rates (default: 0.5 1.0)
- `--parallel-workers`: List of parallel worker counts (default: 4 6)
- `--requests-per-worker`: Requests per worker (default: 5)
- `--timeout`: Request timeout in seconds (default: 300)
- `--max-retries`: Maximum retry attempts (default: 3)
- `--output-dir`: Output directory for results and charts (creates JSON + PNG charts for PDF reports)
- `--output-file`: Save results to JSON file (if used without --output-dir, won't generate charts)

**Chart Generation:**
When using `--output-dir`, the script generates:
- Performance charts for each worker configuration (latency, P95, throughput, failures)
- Overall summary chart showing scaling trends
- `results.json` file with all benchmark data
- All files are saved in the specified output directory and can be used to generate PDF reports

---

### Compare Multiple Endpoints (2-4)

**Script:** `src/compare_multi_endpoints.py`
**Use when:** Evaluating 3-4 models at once

```bash
python3 src/compare_multi_endpoints.py \
  --endpoint-1 model1 \
  --endpoint-1-name "Model 1" \
  --api-token-1 TOKEN_1 \
  --api-root-1 https://workspace1.databricks.com \
  --endpoint-2 model2 \
  --endpoint-2-name "Model 2" \
  --api-token-2 TOKEN_2 \
  --api-root-2 https://workspace2.databricks.com \
  --endpoint-3 model3 \
  --endpoint-3-name "Model 3" \
  --api-token-3 TOKEN_3 \
  --api-root-3 https://workspace3.databricks.com \
  --endpoint-4 model4 \
  --endpoint-4-name "Model 4" \
  --api-token-4 TOKEN_4 \
  --api-root-4 https://workspace4.databricks.com \
  --input-tokens 1000 \
  --output-tokens 200 500 1000 \
  --qps-list 0.5 1.0 2.0 \
  --parallel-workers 4 6 8 \
  --requests-per-worker 5 \
  --timeout 300 \
  --output-dir multi_endpoint_comparison
```

**Required parameters:**
- Endpoints 1 and 2 are **required** (minimum 2 endpoints)
- Endpoints 3 and 4 are **optional**
- Each endpoint requires:
  - `--endpoint-N`: Endpoint name
  - `--endpoint-N-name`: Display name
  - `--api-token-N`: API token
  - `--api-root-N`: Workspace URL (optional, has default)

**Test parameters:**
- Same as single endpoint benchmark
- `--requests-per-worker`: Maximum 5 requests per worker
- `--output-dir`: Output directory (default: multi_endpoint_comparison)

**Output files:**
- Results saved to `all_results.json` (not `results.json`)
- Separate comparison chart for each (QPS × Workers × Tokens) combination

**Generate PDF report:**
```bash
python3 src/generate_pdf_report.py \
  --folder multi_endpoint_comparison \
  --results-filename all_results.json
```

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

## 📊 Single vs Multiple Benchmarks: When to Use What

### Decision Guide

| Scenario | Script to Use | Why |
|----------|---------------|-----|
| Test ONE endpoint performance | `benchmark_llm.py` | Simplest - just need endpoint name and token |
| Compare 2-4 endpoints | `compare_multi_endpoints.py` | Comprehensive comparison across multiple models |

### Key Differences

#### Single Endpoint (`benchmark_llm.py`)
- **Input:** 1 endpoint
- **Output:** Performance metrics + summary table + optional JSON + performance charts (when using --output-dir)
- **Charts generated (with --output-dir):**
  - Performance charts per worker configuration showing latency, P95, throughput, failures
  - Overall summary chart showing scaling trends
- **Use case:** Load testing, performance validation, capacity planning, PDF report generation
- **Example:**
  ```bash
  python3 src/benchmark_llm.py \
    --endpoint my-gpt-model \
    --api-token dapi123abc \
    --api-root https://workspace.databricks.com \
    --output-dir my_benchmark_results

  # Then generate PDF report:
  python3 src/generate_pdf_report.py --folder my_benchmark_results
  ```

#### Multiple Endpoints (`compare_multi_endpoints.py`)
- **Input:** 2-4 endpoints
- **Output:** Individual comparison charts for EVERY configuration combination + comprehensive JSON (`all_results.json`)
- **Charts generated:**
  - Separate chart for each (QPS × Workers × Output Tokens) combination
  - Example: 3 QPS values × 3 worker counts × 3 token sizes = 27 charts
- **Use case:** Vendor selection, comprehensive model evaluation
- **Example:**
  ```bash
  python3 src/compare_multi_endpoints.py \
    --endpoint-1 model1 --endpoint-1-name "Model 1" --api-token-1 TOKEN_1 \
    --endpoint-2 model2 --endpoint-2-name "Model 2" --api-token-2 TOKEN_2 \
    --endpoint-3 model3 --endpoint-3-name "Model 3" --api-token-3 TOKEN_3

  # Generate PDF report
  python3 src/generate_pdf_report.py \
    --folder multi_endpoint_comparison \
    --results-filename all_results.json
  ```

### Configuration Complexity

```
Single Endpoint:    ✓ Simplest   → Just endpoint + token
Multiple Endpoints: ✓✓ Medium    → 2-4 sets of credentials + comprehensive charts
```

### Environment Variables (Optional)

Set these to avoid repeating `--api-root` every time:

```bash
# In your ~/.bashrc or ~/.zshrc
export API_ROOT="https://your-workspace.cloud.databricks.com"

# Now you can omit --api-root in commands
python3 src/benchmark_llm.py \
  --endpoint my-model \
  --api-token dapi123abc
```

---

## 🎓 Common Usage Patterns

### Pattern 1: Quick Performance Check
```bash
# Run benchmark (generates charts automatically)
python3 src/benchmark_llm.py \
  --endpoint my-endpoint \
  --api-token dapi123abc \
  --api-root https://workspace.databricks.com \
  --qps-list 1.0 \
  --parallel-workers 4 \
  --output-dir my_results

# Generate PDF report
python3 src/generate_pdf_report.py --folder my_results
```

### Pattern 2: Compare Models for Procurement (4 Vendors)
```bash
# Test 4 vendors
python3 src/compare_multi_endpoints.py \
  --endpoint-1 vendor-a --endpoint-1-name "Vendor A" --api-token-1 TOKEN_A \
  --endpoint-2 vendor-b --endpoint-2-name "Vendor B" --api-token-2 TOKEN_B \
  --endpoint-3 vendor-c --endpoint-3-name "Vendor C" --api-token-3 TOKEN_C \
  --endpoint-4 vendor-d --endpoint-4-name "Vendor D" --api-token-4 TOKEN_D \
  --qps-list 0.5 1.0 2.0 \
  --parallel-workers 2 4 \
  --output-dir vendor_comparison

# Generate decision report
python3 src/generate_pdf_report.py --folder vendor_comparison
python3 src/generate_summary_stats.py --csv
```

### Pattern 3: Production Readiness Test
```bash
# Stress test at expected load
python3 src/benchmark_llm.py \
  --endpoint prod-endpoint \
  --api-token TOKEN \
  --api-root https://prod-workspace.databricks.com \
  --qps-list 5.0 \
  --parallel-workers 16 \
  --output-tokens 500 \
  --requests-per-worker 5 \
  --timeout 600

# Get detailed analysis
python3 src/generate_summary_stats.py --csv --json
```

### Pattern 4: A/B Test New Model
```bash
# Compare old vs new
python3 src/compare_multi_endpoints.py \
  --endpoint-1 current-prod-v1 \
  --endpoint-1-name "Current Production v1" \
  --api-token-1 TOKEN \
  --endpoint-2 candidate-v2 \
  --endpoint-2-name "Candidate v2" \
  --api-token-2 TOKEN \
  --qps-list 1.0 2.0 5.0 \
  --parallel-workers 4 8 \
  --output-dir ab_test_results

# Review results
python3 src/generate_pdf_report.py \
  --folder ab_test_results \
  --results-filename all_results.json
```

### Pattern 5: Daily Automated Testing
```bash
#!/bin/bash
# daily_benchmark.sh

DATE=$(date +%Y%m%d)

# Set your credentials
API_TOKEN="dapi123abc"
API_ROOT="https://workspace.databricks.com"

python3 src/compare_multi_endpoints.py \
  --endpoint-1 production-endpoint \
  --endpoint-1-name "Production" \
  --api-token-1 $API_TOKEN \
  --api-root-1 $API_ROOT \
  --endpoint-2 staging-endpoint \
  --endpoint-2-name "Staging" \
  --api-token-2 $API_TOKEN \
  --api-root-2 $API_ROOT \
  --qps-list 1.0 \
  --parallel-workers 4 \
  --output-dir "results_${DATE}"

python3 src/generate_pdf_report.py \
  --folder "results_${DATE}" \
  --results-filename all_results.json \
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

### Authentication Errors

**Error:** `HTTP 401 Unauthorized` or `HTTP 403 Forbidden`

**Solution:**
```bash
# 1. Verify your API token is correct
echo $DATABRICKS_TOKEN  # If using env var

# 2. Generate a new token:
#    Databricks Workspace → Settings → User Settings → Access Tokens → Generate New Token

# 3. Test with correct format:
python3 src/benchmark_llm.py \
  --endpoint your-endpoint \
  --api-token dapi1234567890abcdef \
  --api-root https://your-workspace.cloud.databricks.com
```

### Connection Errors

**Error:** `Connection refused`, `Cannot connect to host`, or timeout errors

**Common causes:**
1. **Wrong API root URL**
   ```bash
   # ❌ Wrong - includes /api/2.0
   --api-root https://workspace.databricks.com/api/2.0

   # ✅ Correct - just the workspace URL
   --api-root https://workspace.databricks.com
   ```

2. **Wrong endpoint name**
   ```bash
   # ❌ Wrong - full URL
   --endpoint https://workspace.databricks.com/serving-endpoints/my-model/invocations

   # ✅ Correct - just the endpoint name
   --endpoint my-model
   ```

3. **Network/VPN issues**
   - Ensure you're connected to VPN if required
   - Check firewall settings
   - Test connectivity: `curl https://your-workspace.databricks.com`

### Missing Required Parameters

**Error:** `error: the following arguments are required`

**Solutions:**

For single endpoint:
```bash
# Minimum required parameters:
python3 src/benchmark_llm.py \
  --endpoint ENDPOINT_NAME \
  --api-token YOUR_TOKEN
```

For two endpoint comparison:
```bash
# Minimum required parameters:
python3 src/compare_multi_endpoints.py \
  --endpoint-1 NAME_1 --endpoint-1-name "Display 1" --api-token-1 TOKEN_1 \
  --endpoint-2 NAME_2 --endpoint-2-name "Display 2" --api-token-2 TOKEN_2
```

For multiple endpoints:
```bash
# At least 2 endpoints required:
python3 src/compare_multi_endpoints.py \
  --endpoint-1 NAME_1 --endpoint-1-name "Model 1" --api-token-1 TOKEN_1 \
  --endpoint-2 NAME_2 --endpoint-2-name "Model 2" --api-token-2 TOKEN_2
```

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

### High Failure Rates

If you see many failed requests:
- **Reduce load:** Lower `--qps-list` values (try 0.25 or 0.5)
- **Fewer workers:** Use smaller `--parallel-workers` values (try 2 or 4)
- **Increase timeout:** Use `--timeout 600` for slower endpoints
- **Check endpoint health:** Verify endpoint is running in Databricks UI
- **Reduce retries:** If endpoint is consistently failing, use `--max-retries 1` to fail faster

### Slow Performance / No Output

If the benchmark seems stuck:
- **Normal for first request:** The first request may take 60-300 seconds (cold start)
- **Watch for progress:** Look for "Worker X: Request Y" messages
- **Check timeout:** Default is 300s - increase if needed with `--timeout 600`
- **Reduce scope:** Start with fewer combinations:
  ```bash
  # Minimal test - just one configuration
  python3 src/benchmark_llm.py \
    --endpoint test-endpoint \
    --api-token TOKEN \
    --qps-list 1.0 \
    --parallel-workers 2 \
    --output-tokens 200
  ```

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

| Scenario | Script to Use | Key Parameters |
|----------|---------------|----------------|
| Quick health check | `benchmark_llm.py` | `--endpoint --api-token` |
| Compare 2 models | `compare_multi_endpoints.py` | `--endpoint-1 --endpoint-2` + tokens |
| Evaluate 3-4 models | `compare_multi_endpoints.py` | `--endpoint-1 --endpoint-2 [--endpoint-3] [--endpoint-4]` |
| Vendor selection | `compare_multi_endpoints.py` + PDF report | All 4 endpoints + comprehensive test params |
| Production readiness | `benchmark_llm.py` | High `--qps-list` and `--parallel-workers` |
| A/B testing new model | `compare_multi_endpoints.py` | Production vs candidate endpoints |
| Daily monitoring | Automate with cron + any script | Use `--output-dir results_$(date +%Y%m%d)` |
| Cost optimization | Compare different models | Same test params across all models |
| Load/stress testing | `benchmark_llm.py` | `--qps-list 5.0 10.0 --parallel-workers 16 32` |

---

## 💡 Pro Tips

1. **Start small**: Begin with low QPS (0.5) and few workers (2)
2. **Test gradually**: Increase load step-by-step to find limits
3. **Save results**: Always use `--output-dir` to preserve data
4. **Archive reports**: Keep PDFs by date for trend analysis
5. **Use environment variables**: Set `API_ROOT` to avoid repeating workspace URL
6. **Monitor progress**: Watch for "Worker X: Request Y" messages to track progress
7. **Cold starts**: First request can take 60-300s, be patient
8. **Meaningful names**: Use descriptive `--endpoint-X-name` values for clear reports

---

## 📋 CLI Quick Reference

### Common Parameters (All Scripts)

| Parameter | Description | Example | Required |
|-----------|-------------|---------|----------|
| `--api-token` | Databricks API token (less secure) | `dapi123abc...` | ❌ No |
| `--api-token-env` | Read token from env var name | `DATABRICKS_API_TOKEN` | ❌ No |
| `--api-token-stdin` | Read token from stdin | `printf '%s\n' "$TOKEN" | ...` | ❌ No |
| `--api-root` | Workspace URL | `https://workspace.databricks.com` | ❌ No (has default) |
| `--input-tokens` | Input token sizes (list) | `500 1000 1500` | ❌ No (default: 15000) |
| `--output-tokens` | Output token sizes (list) | `200 500 1000` | ❌ No (default: 200 500 1000) |
| `--qps-list` | Queries per second (list) | `0.5 1.0 2.0` | ❌ No (default: 0.5 1.0) |
| `--parallel-workers` | Worker counts (list) | `4 6 8` | ❌ No (default: 4 6) |
| `--requests-per-worker` | Requests per worker | `5` | ❌ No (default: 5) |
| `--timeout` | Request timeout (seconds) | `300` | ❌ No (default: 300) |
| `--max-retries` | Max retry attempts | `3` | ❌ No (default: 3) |

### Script-Specific Parameters

#### Single Endpoint (`benchmark_llm.py`)
```bash
--endpoint ENDPOINT_NAME          # Required: Endpoint name
--api-token-env DATABRICKS_API_TOKEN  # Recommended token source
--api-token YOUR_TOKEN            # Less secure token source
--api-token-stdin                # Read token from stdin
--output-dir benchmark_results    # Optional: Output directory (saves JSON + generates charts)
--output-file results.json        # Optional: Save to JSON only (no charts)
```

#### Multiple Endpoints (`compare_multi_endpoints.py`)
```bash
# Endpoints 1 and 2 are REQUIRED
--endpoint-1 NAME_1               # Required: First endpoint
--endpoint-1-name "Model 1"       # Required: Display name
--api-token-1 "$API_TOKEN"        # Common pattern: shared token variable
--api-token-1-env DATABRICKS_API_TOKEN_1  # Alternative source
--api-root-1 URL_1                # Optional: Workspace URL

--endpoint-2 NAME_2               # Required: Second endpoint
--endpoint-2-name "Model 2"       # Required: Display name
--api-token-2 "$API_TOKEN"        # Common pattern: shared token variable
--api-token-2-env DATABRICKS_API_TOKEN_2  # Alternative source
--api-root-2 URL_2                # Optional: Workspace URL

# Endpoints 3 and 4 are OPTIONAL
--endpoint-3 NAME_3               # Optional: Third endpoint
--endpoint-3-name "Model 3"       # Optional: Display name
--api-token-3 "$API_TOKEN"        # Optional token for endpoint 3

--endpoint-4 NAME_4               # Optional: Fourth endpoint
--endpoint-4-name "Model 4"       # Optional: Display name
--api-token-4 "$API_TOKEN"        # Optional token for endpoint 4

--output-dir multi_comparison     # Optional: Output directory
```

### Getting Help

```bash
# Get detailed help for any script
python3 src/benchmark_llm.py --help
python3 src/compare_multi_endpoints.py --help
```

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
