import Link from "next/link";

import type { ExperimentSummary } from "@/lib/types";
import { StatusPill } from "@/components/StatusPill";
import { StatCard } from "@/components/StatCard";
import {
  fetchExperiments,
  formatMs,
  formatUsd,
  stageLabel,
} from "@/lib/api";

export const dynamic = "force-dynamic";

const METRIC_STAGES = new Set([
  "metrics_freeze",
  "metrics_pilot",
  "scale_decision",
]);

export default async function HomePage() {
  let experiments: ExperimentSummary[] = [];
  let error: string | null = null;
  try {
    experiments = await fetchExperiments();
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load experiments";
  }

  const completed = experiments.filter((e) => e.status === "completed");
  const metricRuns = experiments.filter((e) =>
    METRIC_STAGES.has(String(e.stage || e.generation_mode || "")),
  );
  const scale = experiments.find(
    (e) => e.stage === "scale_decision" && !e.post_hoc && e.status === "completed",
  ) || experiments.find((e) => e.stage === "scale_decision");
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
          Baseline generation, treatment pilot, and frozen automatic evaluation
          metrics. Open{" "}
          <Link href="/metrics" className="text-accent-deep underline-offset-2 hover:underline">
            Frozen metrics
          </Link>{" "}
          for the scale/diagnose decision, or{" "}
          <Link href="/human-eval" className="text-accent-deep underline-offset-2 hover:underline">
            Human eval
          </Link>{" "}
          for the assignment audit scaffold.
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Experiments" value={String(experiments.length)} />
        <StatCard label="Completed" value={String(completed.length)} />
        <StatCard
          label="Metric / decision runs"
          value={String(metricRuns.length)}
          hint="Freeze, pilot metrics, scale decision"
        />
        <StatCard
          label="Scale verdict"
          value={scale?.decision_verdict || "—"}
          hint={scale ? scale.experiment_id : "No scale_decision yet"}
        />
      </section>

      <section className="grid gap-3 sm:grid-cols-2">
        <StatCard
          label="Pairs tracked"
          value={String(totalPairs)}
          hint="Across completed runs (incl. metric n)"
        />
        <StatCard
          label="Tracked API cost"
          value={formatUsd(totalCost)}
          hint="Configured model pricing + projections"
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
                  <th className="px-4 py-3">Stage</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Pairs / n</th>
                  <th className="px-4 py-3">Avg latency</th>
                  <th className="px-4 py-3">Cost / verdict</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => {
                  const stage = exp.stage || exp.generation_mode;
                  const isMetric = METRIC_STAGES.has(String(stage || ""));
                  return (
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
                        <div className="mt-1 flex flex-wrap gap-1">
                          {exp.post_hoc ? (
                            <span className="rounded bg-warn-soft px-1.5 py-0.5 text-[10px] text-warn">
                              POST-HOC
                            </span>
                          ) : null}
                          {exp.primary_dataset ? (
                            <p className="max-w-xs truncate text-xs text-ink-mute">
                              {exp.primary_dataset}
                            </p>
                          ) : null}
                        </div>
                      </td>
                      <td className="px-4 py-3 capitalize">
                        {stageLabel(stage)}
                      </td>
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
                        {isMetric && exp.decision_verdict ? (
                          <StatusPill status={exp.decision_verdict} />
                        ) : (
                          formatUsd(exp.total_estimated_cost_usd)
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
