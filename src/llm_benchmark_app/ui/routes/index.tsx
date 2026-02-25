import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ConfigPanel } from "@/components/benchmark/config-panel";
import { ResultsPanel } from "@/components/benchmark/results-panel";
import { Activity } from "lucide-react";

export const Route = createFileRoute("/")({
  component: () => <BenchmarkDashboard />,
});

function BenchmarkDashboard() {
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
          <Activity className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">LLM Benchmark</h1>
          <span className="text-xs text-muted-foreground hidden sm:inline">
            Performance testing for Databricks serving endpoints
          </span>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] xl:grid-cols-[420px_1fr] gap-6">
          {/* Left panel: Configuration */}
          <aside className="space-y-4">
            <ConfigPanel
              onBenchmarkStarted={setActiveRunId}
              activeRunId={activeRunId}
            />
          </aside>

          {/* Right panel: Results */}
          <section>
            <ResultsPanel
              activeRunId={activeRunId}
              onSelectRun={setActiveRunId}
            />
          </section>
        </div>
      </main>
    </div>
  );
}
