import Link from "next/link";
import { notFound } from "next/navigation";

import { ExperimentCharts } from "@/components/ExperimentCharts";
import { PairTable } from "@/components/PairTable";
import { StatCard } from "@/components/StatCard";
import { StatusPill } from "@/components/StatusPill";
import {
  fetchExperiment,
  formatMs,
  formatUsd,
  stageLabel,
} from "@/lib/api";

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
  const stage = String(summary.stage || summary.generation_mode || "");
  const isMetricsStage = [
    "metrics_freeze",
    "metrics_pilot",
    "scale_decision",
  ].includes(stage);
  const totalQuestions = detail.pairs.reduce((sum, p) => sum + p.n_questions, 0);
  const decision = (summary.decision as { verdict?: string } | undefined)?.verdict;
  const counts = summary.counts as Record<string, number> | undefined;

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <Link
          href="/"
          className="text-sm text-accent-deep underline-offset-2 hover:underline"
        >
          ← All experiments
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="font-display text-3xl tracking-tight text-ink sm:text-4xl">
            {summary.experiment_id}
          </h1>
          {summary.post_hoc ? (
            <span className="rounded bg-warn-soft px-2 py-1 text-xs text-warn">
              POST-HOC
            </span>
          ) : null}
        </div>
        <p className="text-sm text-ink-mute">
          {stageLabel(stage)} ·{" "}
          {summary.model ||
            (typeof summary.embedding_model === "string"
              ? summary.embedding_model
              : null) ||
            "—"}{" "}
          ·{" "}
          {summary.primary_dataset || "local / metrics"} · human eval:{" "}
          {summary.human_evaluation || "not_yet_evaluated"}
        </p>
        {isMetricsStage ? (
          <p className="text-sm">
            <Link
              href="/metrics"
              className="text-accent-deep underline-offset-2 hover:underline"
            >
              Open aggregated frozen metrics dashboard →
            </Link>
          </p>
        ) : null}
      </div>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {decision ? (
          <StatCard label="Decision verdict" value={String(decision)} />
        ) : (
          <StatCard
            label="Pairs ok"
            value={`${summary.combinations_ok ?? counts?.baseline_sets_M1_M4 ?? 0}/${summary.combinations_attempted ?? counts?.baseline_sets_M1_M4 ?? 0}`}
          />
        )}
        <StatCard
          label={isMetricsStage ? "Artifacts loaded" : "Questions"}
          value={String(isMetricsStage ? detail.pairs.length : totalQuestions)}
        />
        <StatCard
          label="Avg latency"
          value={formatMs(summary.average_latency_ms)}
        />
        <StatCard
          label="Total cost"
          value={formatUsd(summary.total_estimated_cost_usd)}
        />
      </section>

      {stage === "scale_decision" && summary.decision ? (
        <section className="rounded-xl border border-paper-line bg-paper-card p-5 shadow-panel">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="font-display text-xl text-ink">Scale decision</h2>
            <StatusPill status={decision || null} />
          </div>
          <ul className="mt-3 space-y-2 text-sm">
            {(
              (summary.decision as { clauses?: { id: number; text: string; holds: boolean }[] })
                .clauses || []
            ).map((c) => (
              <li
                key={c.id}
                className="flex flex-wrap items-center justify-between gap-2 border-t border-paper-line pt-2"
              >
                <span>
                  {c.id}. {c.text}
                </span>
                <StatusPill status={c.holds ? "HOLD" : "FAIL"} />
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {stage === "metrics_freeze" && counts ? (
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="M1/M4 n" value={String(counts.baseline_sets_M1_M4 ?? "—")} />
          <StatCard label="M2 n" value={String(counts.baseline_sets_M2 ?? "—")} />
          <StatCard label="M3 n" value={String(counts.baseline_sets_M3 ?? "—")} />
          <StatCard label="τₚ n" value={String(counts.tau_calibrated ?? "—")} />
        </section>
      ) : null}

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
