import Link from "next/link";

import type { ExperimentSummary } from "@/lib/types";
import { StatCard } from "@/components/StatCard";
import { fetchExperiments, formatMs, formatUsd } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  let experiments: ExperimentSummary[] = [];
  let error: string | null = null;
  try {
    experiments = await fetchExperiments();
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load experiments";
  }

  const completed = experiments.filter((e) => e.status === "completed");
  const totalCost = completed.reduce(
    (sum, e) => sum + (e.total_estimated_cost_usd || 0),
    0,
  );
  const totalPairs = completed.reduce(
    (sum, e) => sum + (e.combinations_ok || 0),
    0,
  );

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="font-display text-4xl tracking-tight text-ink">
          Experiment results
        </h1>
        <p className="max-w-2xl text-ink-mute">
          Visualize baseline and InterviewIQ runs: cost, latency, tokens, and
          generated questions. Upload new CV/JD pairs from the testing page when
          you want ad-hoc cases beyond the HF dataset.
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Experiments" value={String(experiments.length)} />
        <StatCard label="Completed" value={String(completed.length)} />
        <StatCard
          label="Pairs generated"
          value={String(totalPairs)}
          hint="Across completed runs"
        />
        <StatCard
          label="Tracked API cost"
          value={formatUsd(totalCost)}
          hint="Configured Terra pricing"
        />
      </section>

      {error ? (
        <div className="rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          Cannot reach the API at{" "}
          <code className="font-mono">
            {process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}
          </code>
          . Start the backend, then refresh. ({error})
        </div>
      ) : null}

      <section className="overflow-hidden rounded-xl border border-paper-line bg-paper-card shadow-panel">
        <div className="border-b border-paper-line px-4 py-3">
          <h2 className="font-display text-xl text-ink">Runs</h2>
        </div>
        {experiments.length === 0 && !error ? (
          <p className="p-6 text-sm text-ink-mute">
            No experiment summaries found under experiments/results/.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
                <tr>
                  <th className="px-4 py-3">Experiment</th>
                  <th className="px-4 py-3">Mode</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Pairs</th>
                  <th className="px-4 py-3">Avg latency</th>
                  <th className="px-4 py-3">Cost</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => (
                  <tr
                    key={exp.experiment_id}
                    className="border-t border-paper-line hover:bg-paper/80"
                  >
                    <td className="px-4 py-3">
                      <Link
                        href={`/experiments/${exp.experiment_id}`}
                        className="font-medium text-accent-deep underline-offset-2 hover:underline"
                      >
                        {exp.experiment_id}
                      </Link>
                      {exp.primary_dataset ? (
                        <p className="mt-1 max-w-xs truncate text-xs text-ink-mute">
                          {exp.primary_dataset}
                        </p>
                      ) : null}
                    </td>
                    <td className="px-4 py-3">{exp.generation_mode || "—"}</td>
                    <td className="px-4 py-3 font-mono text-xs">
                      {exp.model || "—"}
                    </td>
                    <td className="px-4 py-3">
                      <StatusPill status={exp.status} />
                    </td>
                    <td className="px-4 py-3">
                      {exp.combinations_ok ?? 0}/
                      {exp.combinations_attempted ?? 0}
                    </td>
                    <td className="px-4 py-3">
                      {formatMs(exp.average_latency_ms)}
                    </td>
                    <td className="px-4 py-3">
                      {formatUsd(exp.total_estimated_cost_usd)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function StatusPill({ status }: { status: string | null }) {
  const tone =
    status === "completed"
      ? "bg-accent-soft text-accent-deep"
      : status === "failed"
        ? "bg-warn-soft text-warn"
        : "bg-paper text-ink-mute";
  return (
    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${tone}`}>
      {status || "unknown"}
    </span>
  );
}
