import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";
import type { BenchmarkResultItem } from "@/lib/benchmark-api";

const COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4"];

interface BenchmarkChartsProps {
  results: BenchmarkResultItem[];
}

export function BenchmarkCharts({ results }: BenchmarkChartsProps) {
  const endpoints = useMemo(
    () => [...new Set(results.map((r) => r.endpoint_name))],
    [results],
  );

  // Group results by config key
  const configs = useMemo(() => {
    const map = new Map<string, BenchmarkResultItem[]>();
    for (const r of results) {
      const key = `QPS=${r.qps} | Workers=${r.num_workers} | OutTokens=${r.output_tokens}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(r);
    }
    return map;
  }, [results]);

  const configKeys = useMemo(() => [...configs.keys()], [configs]);
  const [selectedConfig, setSelectedConfig] = useState<string | "all">("all");

  // Comparison bar charts (per-config)
  const configResults =
    selectedConfig === "all" ? results : (configs.get(selectedConfig) ?? []);

  // Aggregate per endpoint for selected config
  const perEndpoint = useMemo(() => {
    const map = new Map<string, { lats: number[]; p95s: number[]; thrs: number[]; fails: number; total: number }>();
    for (const r of configResults) {
      if (!map.has(r.endpoint_name)) {
        map.set(r.endpoint_name, { lats: [], p95s: [], thrs: [], fails: 0, total: 0 });
      }
      const e = map.get(r.endpoint_name)!;
      e.lats.push(r.median_latency);
      e.p95s.push(r.p95_latency);
      e.thrs.push(r.throughput);
      e.fails += r.failed_requests;
      e.total += r.total_requests;
    }
    return [...map.entries()].map(([name, v]) => ({
      name,
      latency: +(v.lats.reduce((a, b) => a + b, 0) / v.lats.length).toFixed(3),
      p95: +(v.p95s.reduce((a, b) => a + b, 0) / v.p95s.length).toFixed(3),
      throughput: Math.round(v.thrs.reduce((a, b) => a + b, 0) / v.thrs.length),
      failures: v.fails,
    }));
  }, [configResults]);

  // Scaling data (throughput vs workers)
  const scalingData = useMemo(() => {
    const workers = [...new Set(results.map((r) => r.num_workers))].sort(
      (a, b) => a - b,
    );
    return workers.map((w) => {
      const entry: Record<string, number | string> = { workers: w };
      for (const ep of endpoints) {
        const epResults = results.filter(
          (r) => r.endpoint_name === ep && r.num_workers === w,
        );
        if (epResults.length > 0) {
          entry[ep] = Math.round(
            epResults.reduce((s, r) => s + r.throughput, 0) / epResults.length,
          );
        }
      }
      return entry;
    });
  }, [results, endpoints]);

  return (
    <div className="space-y-4">
      {/* Config filter */}
      {configKeys.length > 1 && (
        <div className="flex flex-wrap gap-1.5">
          <Badge
            variant={selectedConfig === "all" ? "default" : "outline"}
            className="cursor-pointer text-xs"
            onClick={() => setSelectedConfig("all")}
          >
            All configs
          </Badge>
          {configKeys.map((key) => (
            <Badge
              key={key}
              variant={selectedConfig === key ? "default" : "outline"}
              className="cursor-pointer text-xs"
              onClick={() => setSelectedConfig(key)}
            >
              {key}
            </Badge>
          ))}
        </div>
      )}

      {/* Bar charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Median Latency (lower is better)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perEndpoint}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                formatter={(v) => [`${v}s`, "Median Latency"]}
              />
              <Bar dataKey="latency" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="P95 Latency (lower is better)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perEndpoint}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                formatter={(v) => [`${v}s`, "P95 Latency"]}
              />
              <Bar dataKey="p95" fill="#c0392b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Throughput (higher is better)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perEndpoint}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                formatter={(v) => [`${v} tok/s`, "Throughput"]}
              />
              <Bar dataKey="throughput" fill="#3498db" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Failed Requests (lower is better)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perEndpoint}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                formatter={(v) => [v, "Failures"]}
              />
              <Bar dataKey="failures" fill="#e84118" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Scaling chart */}
      {scalingData.length > 1 && endpoints.length >= 1 && (
        <ChartCard title="Throughput Scaling (by worker count)">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={scalingData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis
                dataKey="workers"
                tick={{ fontSize: 11 }}
                label={{
                  value: "Parallel Workers",
                  position: "insideBottom",
                  offset: -5,
                  fontSize: 12,
                }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{
                  value: "Throughput (tok/s)",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 12,
                }}
              />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Legend />
              {endpoints.map((ep, i) => (
                <Line
                  key={ep}
                  type="monotone"
                  dataKey={ep}
                  stroke={COLORS[i % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}
    </div>
  );
}

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="pt-0">{children}</CardContent>
    </Card>
  );
}
