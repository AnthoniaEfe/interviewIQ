import Link from "next/link";
import { notFound } from "next/navigation";

import { ExperimentCharts } from "@/components/ExperimentCharts";
import { PairTable } from "@/components/PairTable";
import { StatCard } from "@/components/StatCard";
import { fetchExperiment, formatMs, formatUsd } from "@/lib/api";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ id: string }>;
};

export default async function ExperimentDetailPage({ params }: Props) {
  const { id } = await params;
  let detail;
  try {
    detail = await fetchExperiment(id);
  } catch {
    notFound();
  }

  const summary = detail.summary;
  const totalQuestions = detail.pairs.reduce((sum, p) => sum + p.n_questions, 0);

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <Link
          href="/"
          className="text-sm text-accent-deep underline-offset-2 hover:underline"
        >
          ← All experiments
        </Link>
        <h1 className="font-display text-3xl tracking-tight text-ink sm:text-4xl">
          {summary.experiment_id}
        </h1>
        <p className="text-sm text-ink-mute">
          {summary.generation_mode} · {summary.model} ·{" "}
          {summary.primary_dataset || "local dataset"} · human eval:{" "}
          {summary.human_evaluation || "not_yet_evaluated"}
        </p>
      </div>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Pairs ok"
          value={`${summary.combinations_ok ?? 0}/${summary.combinations_attempted ?? 0}`}
        />
        <StatCard label="Questions" value={String(totalQuestions)} />
        <StatCard
          label="Avg latency"
          value={formatMs(summary.average_latency_ms)}
        />
        <StatCard
          label="Total cost"
          value={formatUsd(summary.total_estimated_cost_usd)}
        />
      </section>

      <ExperimentCharts detail={detail} />

      {detail.label_stats.length > 0 ? (
        <section className="overflow-hidden rounded-xl border border-paper-line bg-paper-card shadow-panel">
          <div className="border-b border-paper-line px-4 py-3">
            <h2 className="font-display text-xl text-ink">Fit label breakdown</h2>
          </div>
          <table className="min-w-full text-left text-sm">
            <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
              <tr>
                <th className="px-4 py-2">Label</th>
                <th className="px-4 py-2">Pairs</th>
                <th className="px-4 py-2">Avg latency</th>
                <th className="px-4 py-2">Avg cost</th>
                <th className="px-4 py-2">Total cost</th>
                <th className="px-4 py-2">Tokens in/out</th>
              </tr>
            </thead>
            <tbody>
              {detail.label_stats.map((row) => (
                <tr key={row.label} className="border-t border-paper-line">
                  <td className="px-4 py-2">{row.label}</td>
                  <td className="px-4 py-2">{row.n}</td>
                  <td className="px-4 py-2">{formatMs(row.avg_latency_ms)}</td>
                  <td className="px-4 py-2">{formatUsd(row.avg_cost)}</td>
                  <td className="px-4 py-2">{formatUsd(row.total_cost)}</td>
                  <td className="px-4 py-2 font-mono text-xs">
                    {row.total_input_tokens} / {row.total_output_tokens}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      <PairTable pairs={detail.pairs} />
    </div>
  );
}
