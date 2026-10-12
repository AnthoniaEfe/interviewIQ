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

import type { ExperimentDetail, MetricsDashboard } from "@/lib/types";

const COLORS = {
  m1: "#b45309",
  m2: "#0f766e",
  m3: "#115e59",
  m4: "#5c6f64",
  B0: "#2A6F97",
  B1: "#1B4332",
  T: "#E76F51",
  B2: "#E9C46A",
  S0: "#6C757D",
  S1: "#495057",
  S2: "#ADB5BD",
};

export function PerPairMetricChart({ detail }: { detail: ExperimentDetail }) {
  const rows = (detail.charts.metrics || []).map((d) => ({
    index: d.index,
    M1: d["M1.P"] == null ? null : Number(Number(d["M1.P"]).toFixed(3)),
    M2: d["M2.P"] == null ? null : Number(Number(d["M2.P"]).toFixed(3)),
    M3: d["M3.P"] == null ? null : Number(Number(d["M3.P"]).toFixed(3)),
    M4: d["M4.P"] == null ? null : Number(Number(d["M4.P"]).toFixed(3)),
  }));
  if (!rows.length) return null;
  return (
    <ChartCard
      title="Primary metrics by pair"
      caption="M1.P lower better; M2.P / M3.P / M4.P higher better"
    >
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d5ddd8" />
          <XAxis dataKey="index" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="M1" fill={COLORS.m1} name="M1.P" radius={[2, 2, 0, 0]} />
          <Bar dataKey="M2" fill={COLORS.m2} name="M2.P" radius={[2, 2, 0, 0]} />
          <Bar dataKey="M3" fill={COLORS.m3} name="M3.P" radius={[2, 2, 0, 0]} />
          <Bar dataKey="M4" fill={COLORS.m4} name="M4.P" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export function ConditionSummaryChart({
  dashboard,
}: {
  dashboard: MetricsDashboard;
}) {
  const summaries = dashboard.pilot.condition_summaries || {};
  const conditions = ["B0", "B1", "T", "B2", "S0", "S1", "S2"].filter(
    (c) => summaries[c],
  );
  if (!conditions.length) return null;
  const data = conditions.map((c) => ({
    condition: c,
    M1: summaries[c]?.["M1.P"]?.mean ?? null,
    M2: summaries[c]?.["M2.P"]?.mean ?? null,
    M3: summaries[c]?.["M3.P"]?.mean ?? null,
    M4: summaries[c]?.["M4.P"]?.mean ?? null,
  }));
  return (
    <ChartCard
      title="Pilot primary metrics by condition"
      caption="Means over primary-30 (see report for 95% CIs)"
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d5ddd8" />
          <XAxis dataKey="condition" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="M1" fill={COLORS.m1} name="M1.P" />
          <Bar dataKey="M2" fill={COLORS.m2} name="M2.P" />
          <Bar dataKey="M3" fill={COLORS.m3} name="M3.P" />
          <Bar dataKey="M4" fill={COLORS.m4} name="M4.P" />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
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
