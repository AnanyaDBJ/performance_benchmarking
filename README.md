# LLM Benchmark App

A full-stack Databricks App for benchmarking LLM serving endpoints. Configure tests, run benchmarks, view real-time progress, and download professional reports — all from a web UI.

---

## Features

- **Endpoint discovery** — automatically lists serving endpoints in your workspace with type classification (Pay-per-Token, Provisioned Throughput, External Model, Custom)
- **Configurable benchmarks** — select 1–4 endpoints, set input/output token sizes, QPS rates, worker counts, and more
- **Real-time progress** — SSE streaming delivers live updates as the benchmark runs
- **Interactive results** — sortable tables, latency/throughput charts, and per-endpoint summaries
- **Export** — download PDF reports, CSV summaries, or raw JSON from completed runs
- **Persistent history** — benchmark runs are saved to Lakebase (PostgreSQL) so you can revisit past results; falls back to in-memory storage when no database is configured

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  React + Vite Frontend  (TanStack Router, shadcn)│
│  src/llm_benchmark_app/ui/                       │
└────────────────────┬─────────────────────────────┘
                     │  /api/*
┌────────────────────▼─────────────────────────────┐
│  FastAPI Backend                                 │
│  src/llm_benchmark_app/backend/                  │
│    ├── benchmark/engine.py   (benchmark runner)  │
│    ├── benchmark/report.py   (PDF generation)    │
│    ├── benchmark/stats.py    (CSV/JSON export)   │
│    ├── benchmark/db.py       (Lakebase storage)  │
│    └── benchmark_router.py   (API routes)        │
└────────────────────┬─────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Databricks Workspace   │
        │  - Serving Endpoints    │
        │  - Lakebase (Postgres)  │
        └─────────────────────────┘
```

Backend serves the frontend at `/` and exposes the API at `/api`. The OpenAPI client is auto-generated from the backend schema so the frontend always stays in sync.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A Databricks workspace with serving endpoints
- (Optional) Lakebase database instance for persistent benchmark history

---

## Getting Started

### 1. Install dependencies

```bash
uv sync
```

### 2. Start development servers

```bash
uv run apx dev start
```

This starts the FastAPI backend, Vite frontend dev server, and the OpenAPI client watcher. Check status with:

```bash
uv run apx dev status
```

### 3. Open the app

Navigate to the URL printed by the dev server (typically `http://localhost:8000`).

### Other useful commands

```bash
uv run apx dev logs          # View recent logs (default: last 10 min)
uv run apx dev logs -f       # Stream logs live
uv run apx dev check         # Run TypeScript + Python lint checks
uv run apx dev stop          # Stop all dev servers
uv run apx build             # Build for production
```

---

## Deployment

The app is deployed as a Databricks App using Databricks Asset Bundles. The `databricks.yml` defines the app resource along with an optional Lakebase database instance for persistent storage.

```bash
# Deploy to Databricks
databricks bundle deploy

# Check deployed app logs
uv run apx databricks-apps-logs
```

### Database

When a Lakebase database instance is configured (via `databricks.yml` resources), benchmark runs and results are persisted across app restarts. Without it, the app runs with in-memory storage only.

---

## Usage

### Running a benchmark

1. **Select endpoints** — the endpoint selector auto-discovers serving endpoints in your workspace. Pick 1–4 endpoints to benchmark.
2. **Configure parameters** — set input/output token sizes, QPS rates, parallel worker counts, requests per worker, timeout, and retry limits.
3. **Start** — click Run to kick off the benchmark. A real-time progress bar and status messages track each test configuration.
4. **View results** — once complete, results appear in sortable tables and charts showing latency, throughput, and success rates.

### Downloading reports

From any completed benchmark run, download:

- **PDF** — professional report with charts, rankings, and executive summary
- **CSV** — spreadsheet-ready summary statistics
- **JSON** — raw benchmark data for programmatic analysis

### Benchmark history

The History tab shows all past benchmark runs. Click any run to load its full results and charts, or download its reports.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/endpoints` | List serving endpoints in the workspace |
| `GET` | `/api/benchmarks` | List all benchmark runs |
| `POST` | `/api/benchmarks` | Start a new benchmark run |
| `GET` | `/api/benchmarks/{id}` | Get run status and results |
| `POST` | `/api/benchmarks/{id}/cancel` | Cancel a running benchmark |
| `GET` | `/api/benchmarks/{id}/stream` | SSE stream of benchmark progress |
| `GET` | `/api/benchmarks/{id}/pdf` | Download PDF report |
| `GET` | `/api/benchmarks/{id}/csv` | Download CSV summary |
| `GET` | `/api/benchmarks/{id}/json` | Download raw JSON results |

---

## Project Structure

```
performance_benchmarking/
├── databricks.yml                              # Databricks Asset Bundle config
├── app.yml                                     # App entrypoint (uvicorn)
├── pyproject.toml                              # Python project & apx config
├── package.json                                # Frontend dependencies
│
└── src/llm_benchmark_app/
    ├── backend/
    │   ├── app.py                              # FastAPI app + lifespan
    │   ├── core.py                             # Dependencies & config
    │   ├── models.py                           # Pydantic request/response models
    │   ├── router.py                           # Base routes (version, health)
    │   ├── benchmark_router.py                 # Benchmark API routes
    │   └── benchmark/
    │       ├── engine.py                       # Benchmark runner with progress/cancel
    │       ├── db.py                           # Lakebase (Postgres) persistence
    │       ├── report.py                       # PDF report generation
    │       └── stats.py                        # CSV/JSON export & summary stats
    │
    └── ui/
        ├── main.tsx                            # React entry point
        ├── routes/
        │   ├── __root.tsx                      # Root layout
        │   └── index.tsx                       # Dashboard page
        ├── components/
        │   ├── benchmark/
        │   │   ├── config-panel.tsx             # Benchmark configuration form
        │   │   ├── results-panel.tsx            # Results display + history
        │   │   ├── benchmark-charts.tsx         # Latency/throughput charts
        │   │   ├── endpoint-selector.tsx        # Endpoint picker with search
        │   │   └── chip-input.tsx               # Numeric list input
        │   └── ui/                             # shadcn/ui primitives
        ├── lib/
        │   ├── benchmark-api.ts                # API client hooks & helpers
        │   └── utils.ts                        # Utilities
        └── styles/
            └── globals.css                     # Tailwind + theme
```

---

## Key Metrics

The benchmark measures each endpoint across every combination of configuration parameters:

| Metric | Description |
|--------|-------------|
| **Median Latency** | 50th percentile response time |
| **P95 Latency** | 95th percentile response time |
| **Throughput** | Tokens processed per second |
| **Success Rate** | Percentage of requests that completed without error |
| **Input / Output Tokens** | Token counts sent and received per request |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Vite, TanStack Router, shadcn/ui, Tailwind CSS, Recharts |
| Backend | FastAPI, Pydantic, uvicorn |
| Benchmark Engine | aiohttp (async HTTP), matplotlib (charts), reportlab (PDF) |
| Database | Lakebase (PostgreSQL) via psycopg + connection pooling |
| Auth | Databricks OBO (on-behalf-of) tokens |
| Deployment | Databricks Apps, Databricks Asset Bundles |
| Tooling | apx, uv, Bun |
