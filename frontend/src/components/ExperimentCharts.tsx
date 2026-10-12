"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { PerPairMetricChart } from "@/components/MetricCharts";
import type { ExperimentDetail } from "@/lib/types";

const COLORS = {
  latency: "#0f766e",
  cost: "#b45309",
  input: "#115e59",
  output: "#5c6f64",
  good: "#0f766e",
  potential: "#b45309",
};

type Props = {
  detail: ExperimentDetail;
};

export function ExperimentCharts({ detail }: Props) {
  const latency = detail.charts.latency_ms.map((d) => ({
    index: d.index,
    latency: Number(d.value.toFixed(0)),
    label: d.fit_label,
  }));
  const cost = detail.charts.cost_usd.map((d) => ({
    index: d.index,
    cost: Number((d.value * 1000).toFixed(3)),
    label: d.fit_label,
  }));
  const tokens = detail.charts.tokens.map((d) => ({
    index: d.index,
    input: d.input,
    output: d.output,
  }));
  const byLabel = detail.label_stats.map((d) => ({
    label: d.label,
    avgLatency: Number(d.avg_latency_ms.toFixed(0)),
    totalCost: Number((d.total_cost * 1000).toFixed(2)),
    pairs: d.n,
  }));
  const hasOpsCharts =
    latency.length > 0 || cost.length > 0 || tokens.length > 0;
  const hasMetrics = (detail.charts.metrics || []).length > 0;

  if (!hasOpsCharts && !hasMetrics) {
    return (
      <p className="rounded-xl border border-dashed border-paper-line bg-paper-card p-6 text-sm text-ink-mute">
        No per-pair chart series for this experiment stage. See the Frozen
        metrics page for aggregated decision results.
      </p>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {hasMetrics ? (
        <div className="lg:col-span-2">
          <PerPairMetricChart detail={detail} />
        </div>
      ) : null}

      {latency.length > 0 ? (
        <ChartCard
          title="Latency by pair"
          caption="Milliseconds per generation call"
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={latency}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d5ddd8" />
              <XAxis dataKey="index" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(value) => [`${String(value)} ms`, "Latency"]}
                labelFormatter={(label) => `Pair ${String(label)}`}
              />
              <Bar dataKey="latency" fill={COLORS.latency} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      ) : null}

      {cost.length > 0 ? (
        <ChartCard
          title="Estimated cost by pair"
          caption="USD × 1000 (mUSD) for readability"
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={cost}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d5ddd8" />
              <XAxis dataKey="index" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(value) => [`${String(value)} mUSD`, "Cost"]}
                labelFormatter={(label) => `Pair ${String(label)}`}
              />
              <Bar dataKey="cost" fill={COLORS.cost} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      ) : null}

      {tokens.length > 0 ? (
        <ChartCard title="Tokens by pair" caption="Input vs output tokens">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={tokens}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d5ddd8" />
              <XAxis dataKey="index" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="input" stackId="t" fill={COLORS.input} name="Input" />
              <Bar
                dataKey="output"
                stackId="t"
                fill={COLORS.output}
                name="Output"
                radius={[3, 3, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      ) : null}

      {byLabel.length > 0 && hasOpsCharts ? (
        <ChartCard
          title="By fit label"
          caption="Average latency (ms) and total cost (mUSD)"
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={byLabel}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d5ddd8" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar
                yAxisId="left"
                dataKey="avgLatency"
                fill={COLORS.good}
                name="Avg latency (ms)"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                yAxisId="right"
                dataKey="totalCost"
                fill={COLORS.potential}
                name="Total cost (mUSD)"
                radius={[3, 3, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      ) : null}
    </div>
  );
}

function ChartCard({
  title,
  caption,
  children,
}: {
  title: string;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-xl border border-paper-line bg-paper-card p-4 shadow-panel">
      <div className="mb-3">
        <h3 className="font-display text-lg text-ink">{title}</h3>
        <p className="text-xs text-ink-mute">{caption}</p>
      </div>
      {children}
    </section>
  );
}
