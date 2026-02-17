import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useBenchmarkResults,
  useBenchmarks,
  downloadUrl,
  type BenchmarkResultItem,
  type BenchmarkRunOut,
} from "@/lib/benchmark-api";
import { BenchmarkCharts } from "./benchmark-charts";
import {
  Download,
  FileText,
  FileSpreadsheet,
  FileJson,
  Clock,
  Activity,
  Zap,
  CheckCircle2,
  BarChart3,
  TableIcon,
  History,
  LayoutDashboard,
} from "lucide-react";

interface ResultsPanelProps {
  activeRunId: string | null;
  onSelectRun: (runId: string) => void;
}

export function ResultsPanel({ activeRunId, onSelectRun }: ResultsPanelProps) {
  const { data: result, isLoading } = useBenchmarkResults(activeRunId);
  const { data: runs } = useBenchmarks();

  const hasResults =
    result && result.results && result.results.length > 0;

  return (
    <div className="space-y-4 h-full flex flex-col">
      <Tabs defaultValue="summary" className="flex-1 flex flex-col">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="summary" className="text-xs">
            <LayoutDashboard className="h-3.5 w-3.5 mr-1" />
            Summary
          </TabsTrigger>
          <TabsTrigger value="charts" className="text-xs">
            <BarChart3 className="h-3.5 w-3.5 mr-1" />
            Charts
          </TabsTrigger>
          <TabsTrigger value="details" className="text-xs">
            <TableIcon className="h-3.5 w-3.5 mr-1" />
            Details
          </TabsTrigger>
          <TabsTrigger value="history" className="text-xs">
            <History className="h-3.5 w-3.5 mr-1" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="flex-1 mt-4">
          {!activeRunId && <EmptyState />}
          {activeRunId && isLoading && <LoadingSkeleton />}
          {activeRunId && hasResults && (
            <SummaryTab results={result.results} summary={result.summary} />
          )}
          {activeRunId && !isLoading && !hasResults && result && (
            <RunningState status={result.status} />
          )}
        </TabsContent>

        <TabsContent value="charts" className="flex-1 mt-4">
          {!activeRunId && <EmptyState />}
          {activeRunId && hasResults && (
            <BenchmarkCharts results={result.results} />
          )}
          {activeRunId && !hasResults && !isLoading && (
            <RunningState status={result?.status ?? "pending"} />
          )}
        </TabsContent>

        <TabsContent value="details" className="flex-1 mt-4">
          {!activeRunId && <EmptyState />}
          {activeRunId && hasResults && (
            <DetailsTab results={result.results} />
          )}
          {activeRunId && !hasResults && !isLoading && (
            <RunningState status={result?.status ?? "pending"} />
          )}
        </TabsContent>

        <TabsContent value="history" className="flex-1 mt-4">
          <HistoryTab
            runs={runs ?? []}
            activeRunId={activeRunId}
            onSelectRun={onSelectRun}
          />
        </TabsContent>
      </Tabs>

      {/* Download bar */}
      {activeRunId && hasResults && (
        <DownloadBar runId={activeRunId} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Summary Tab
// ---------------------------------------------------------------------------

function SummaryTab({
  results,
}: {
  results: BenchmarkResultItem[];
  summary: Record<string, unknown> | null;
}) {
  const totalRequests = results.reduce((s, r) => s + r.total_requests, 0);
  const successfulRequests = results.reduce(
    (s, r) => s + r.successful_requests,
    0,
  );
  const failedRequests = results.reduce((s, r) => s + r.failed_requests, 0);
  const avgLatency =
    results.reduce((s, r) => s + r.median_latency, 0) / results.length;
  const avgThroughput =
    results.reduce((s, r) => s + r.throughput, 0) / results.length;
  const bestLatency = Math.min(...results.map((r) => r.median_latency));
  const bestThroughput = Math.max(...results.map((r) => r.throughput));
  const bestLatencyEp = results.find(
    (r) => r.median_latency === bestLatency,
  )?.endpoint_name;
  const bestThroughputEp = results.find(
    (r) => r.throughput === bestThroughput,
  )?.endpoint_name;

  const successRate =
    totalRequests > 0
      ? ((successfulRequests / totalRequests) * 100).toFixed(1)
      : "0";

  // Unique endpoints
  const endpoints = [...new Set(results.map((r) => r.endpoint_name))];

  return (
    <div className="space-y-4">
      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          icon={<Clock className="h-4 w-4 text-blue-500" />}
          label="Best Latency"
          value={`${bestLatency.toFixed(3)}s`}
          sub={bestLatencyEp}
        />
        <MetricCard
          icon={<Zap className="h-4 w-4 text-amber-500" />}
          label="Best Throughput"
          value={`${bestThroughput.toFixed(0)} tok/s`}
          sub={bestThroughputEp}
        />
        <MetricCard
          icon={<CheckCircle2 className="h-4 w-4 text-green-500" />}
          label="Success Rate"
          value={`${successRate}%`}
          sub={`${successfulRequests}/${totalRequests} requests`}
        />
        <MetricCard
          icon={<Activity className="h-4 w-4 text-purple-500" />}
          label="Avg Throughput"
          value={`${avgThroughput.toFixed(0)} tok/s`}
          sub={`Avg latency: ${avgLatency.toFixed(3)}s`}
        />
      </div>

      {/* Performance ranking */}
      {endpoints.length > 1 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Performance Rankings</CardTitle>
          </CardHeader>
          <CardContent>
            <RankingTable results={results} endpoints={endpoints} />
          </CardContent>
        </Card>
      )}

      {/* Quick stats */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-y-2 text-sm">
            <span className="text-muted-foreground">Endpoints tested</span>
            <span className="font-medium">{endpoints.length}</span>
            <span className="text-muted-foreground">Total configurations</span>
            <span className="font-medium">{results.length}</span>
            <span className="text-muted-foreground">Failed requests</span>
            <span className={`font-medium ${failedRequests > 0 ? "text-destructive" : ""}`}>
              {failedRequests}
            </span>
            <span className="text-muted-foreground">Latency range</span>
            <span className="font-medium">
              {bestLatency.toFixed(3)}s &ndash;{" "}
              {Math.max(...results.map((r) => r.median_latency)).toFixed(3)}s
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3">
        <div className="flex items-center gap-2 mb-1">
          {icon}
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
        <p className="text-xl font-bold">{value}</p>
        {sub && (
          <p className="text-xs text-muted-foreground mt-0.5 truncate">
            {sub}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function RankingTable({
  results,
  endpoints,
}: {
  results: BenchmarkResultItem[];
  endpoints: string[];
}) {
  const avgMetrics = endpoints.map((ep) => {
    const epResults = results.filter((r) => r.endpoint_name === ep);
    const avgLat =
      epResults.reduce((s, r) => s + r.median_latency, 0) / epResults.length;
    const avgThr =
      epResults.reduce((s, r) => s + r.throughput, 0) / epResults.length;
    return { name: ep, avgLat, avgThr };
  });

  const byLatency = [...avgMetrics].sort((a, b) => a.avgLat - b.avgLat);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-12">#</TableHead>
          <TableHead>Endpoint</TableHead>
          <TableHead className="text-right">Avg Latency</TableHead>
          <TableHead className="text-right">Avg Throughput</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {byLatency.map((ep, i) => (
          <TableRow key={ep.name}>
            <TableCell className="font-medium">
              <Badge variant={i === 0 ? "default" : "secondary"} className="w-6 h-6 p-0 justify-center">
                {i + 1}
              </Badge>
            </TableCell>
            <TableCell className="font-medium">{ep.name}</TableCell>
            <TableCell className="text-right tabular-nums">
              {ep.avgLat.toFixed(3)}s
            </TableCell>
            <TableCell className="text-right tabular-nums">
              {ep.avgThr.toFixed(0)} tok/s
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ---------------------------------------------------------------------------
// Details Tab
// ---------------------------------------------------------------------------

function DetailsTab({ results }: { results: BenchmarkResultItem[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">All Benchmark Results</CardTitle>
      </CardHeader>
      <CardContent className="overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Endpoint</TableHead>
              <TableHead className="text-right">QPS</TableHead>
              <TableHead className="text-right">Workers</TableHead>
              <TableHead className="text-right">Out Tokens</TableHead>
              <TableHead className="text-right">Latency</TableHead>
              <TableHead className="text-right">P95</TableHead>
              <TableHead className="text-right">Throughput</TableHead>
              <TableHead className="text-right">Success</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {results.map((r, i) => (
              <TableRow key={i}>
                <TableCell className="font-medium text-xs">
                  {r.endpoint_name}
                </TableCell>
                <TableCell className="text-right tabular-nums text-xs">
                  {r.qps}
                </TableCell>
                <TableCell className="text-right tabular-nums text-xs">
                  {r.num_workers}
                </TableCell>
                <TableCell className="text-right tabular-nums text-xs">
                  {r.output_tokens}
                </TableCell>
                <TableCell className="text-right tabular-nums text-xs">
                  {r.median_latency.toFixed(3)}s
                </TableCell>
                <TableCell className="text-right tabular-nums text-xs">
                  {r.p95_latency.toFixed(3)}s
                </TableCell>
                <TableCell className="text-right tabular-nums text-xs">
                  {r.throughput.toFixed(0)}
                </TableCell>
                <TableCell className="text-right text-xs">
                  <span
                    className={
                      r.failed_requests > 0 ? "text-destructive" : "text-green-600"
                    }
                  >
                    {r.successful_requests}/{r.total_requests}
                  </span>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// History Tab
// ---------------------------------------------------------------------------

function HistoryTab({
  runs,
  activeRunId,
  onSelectRun,
}: {
  runs: BenchmarkRunOut[];
  activeRunId: string | null;
  onSelectRun: (id: string) => void;
}) {
  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <History className="h-10 w-10 text-muted-foreground/40 mb-3" />
        <p className="text-sm text-muted-foreground">
          No benchmark runs yet. Configure and run your first benchmark.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {runs.map((run) => (
        <Card
          key={run.id}
          className={`cursor-pointer transition-colors hover:bg-accent/50 ${
            run.id === activeRunId ? "border-primary" : ""
          }`}
          onClick={() => onSelectRun(run.id)}
        >
          <CardContent className="py-3">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <StatusBadge status={run.status} />
                <span className="text-xs text-muted-foreground font-mono">
                  {run.id.slice(0, 8)}
                </span>
              </div>
              <span className="text-xs text-muted-foreground">
                {new Date(run.created_at).toLocaleString()}
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              {run.endpoint_names.map((ep) => (
                <Badge key={ep} variant="outline" className="text-[10px]">
                  {ep}
                </Badge>
              ))}
              {run.result_count > 0 && (
                <span className="text-xs text-muted-foreground ml-auto">
                  {run.result_count} results
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "completed"
      ? "default"
      : status === "running"
        ? "secondary"
        : status === "failed"
          ? "destructive"
          : "outline";
  return (
    <Badge variant={variant} className="text-[10px]">
      {status}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Download Bar
// ---------------------------------------------------------------------------

function DownloadBar({ runId }: { runId: string }) {
  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-2">
          <Download className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Export Results</span>
          <div className="flex gap-2 ml-auto">
            <Button
              variant="outline"
              size="sm"
              asChild
            >
              <a href={downloadUrl(runId, "pdf")} download>
                <FileText className="h-3.5 w-3.5 mr-1" />
                PDF Report
              </a>
            </Button>
            <Button
              variant="outline"
              size="sm"
              asChild
            >
              <a href={downloadUrl(runId, "csv")} download>
                <FileSpreadsheet className="h-3.5 w-3.5 mr-1" />
                CSV
              </a>
            </Button>
            <Button
              variant="outline"
              size="sm"
              asChild
            >
              <a href={downloadUrl(runId, "json")} download>
                <FileJson className="h-3.5 w-3.5 mr-1" />
                JSON
              </a>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <BarChart3 className="h-12 w-12 text-muted-foreground/30 mb-4" />
      <h3 className="text-lg font-medium text-muted-foreground mb-1">
        No Results Yet
      </h3>
      <p className="text-sm text-muted-foreground max-w-sm">
        Select endpoints and configure your benchmark parameters, then click
        Run Benchmark to start.
      </p>
    </div>
  );
}

function RunningState({ status }: { status: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <Activity className="h-12 w-12 text-muted-foreground/30 mb-4 animate-pulse" />
      <h3 className="text-lg font-medium text-muted-foreground mb-1">
        {status === "running" ? "Benchmark Running..." : `Status: ${status}`}
      </h3>
      <p className="text-sm text-muted-foreground">
        Results will appear here when the benchmark completes.
      </p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
      <Skeleton className="h-48" />
    </div>
  );
}
