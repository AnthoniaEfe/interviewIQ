import Link from "next/link";

import { ConditionSummaryChart } from "@/components/MetricCharts";
import { StatCard } from "@/components/StatCard";
import { StatusPill } from "@/components/StatusPill";
import {
  fetchMetricsDashboard,
  figureUrl,
  formatCi,
  formatCny,
  formatMetric,
  formatUsd,
} from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function MetricsPage() {
  let dashboard;
  let error: string | null = null;
  try {
    dashboard = await fetchMetricsDashboard();
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load metrics dashboard";
  }

  if (error || !dashboard) {
    return (
      <div className="space-y-4">
        <h1 className="font-display text-4xl text-ink">Frozen metrics</h1>
        <div className="rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          {error || "No metrics data."} Start the API and ensure Part A/B results exist.
        </div>
      </div>
    );
  }

  const { freeze, baseline, pilot, scale, figures, limitations } = dashboard;
  const dist = baseline.distributions || {};
  const verdict = scale.verdict || "—";
  const cost = scale.cost_projection || {};

  return (
    <div className="space-y-10">
      <section className="space-y-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-accent-deep">
          Pre-registered v1
        </p>
        <h1 className="font-display text-4xl tracking-tight text-ink">
          Frozen metrics & scale decision
        </h1>
        <p className="max-w-3xl text-ink-mute">
          Alignment and diversity proxies — not question quality. Fit-label control
          was null; gap labels are noisy.{" "}
          <Link
            href="/human-eval"
            className="text-accent-deep underline-offset-2 hover:underline"
          >
            Human evaluation
          </Link>{" "}
          is separate and optional.
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Verdict" value={String(verdict)} hint="Advisory" />
        <StatCard
          label="Baseline M1/M4 n"
          value={String(baseline.counts?.baseline_sets_M1_M4 ?? "—")}
        />
        <StatCard
          label="M2/M3 n"
          value={String(baseline.counts?.baseline_sets_M2 ?? "—")}
          hint="Pairs with gap matrices"
        />
        <StatCard
          label="Projected 150-pair cost"
          value={formatCny(numberOrNull(cost.total_cny))}
          hint={formatUsd(numberOrNull(cost.total_usd))}
        />
      </section>

      {freeze ? (
        <section className="rounded-xl border border-paper-line bg-paper-card p-5 shadow-panel">
          <h2 className="font-display text-xl text-ink">Freeze integrity</h2>
          <dl className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
            <div>
              <dt className="font-mono text-[11px] uppercase text-ink-mute">
                Frozen at (UTC)
              </dt>
              <dd className="text-ink">{freeze.frozen_at_utc || "—"}</dd>
            </div>
            <div>
              <dt className="font-mono text-[11px] uppercase text-ink-mute">
                Git commit
              </dt>
              <dd className="font-mono text-xs text-ink">
                {freeze.git_commit || "—"}
              </dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="font-mono text-[11px] uppercase text-ink-mute">
                Spec SHA-256
              </dt>
              <dd className="break-all font-mono text-xs text-ink">
                {freeze.spec_sha256 || "—"}
              </dd>
            </div>
            <div>
              <dt className="font-mono text-[11px] uppercase text-ink-mute">
                τₚ calibrated
              </dt>
              <dd className="text-ink">{freeze.n_tau}</dd>
            </div>
            <div>
              <dt className="font-mono text-[11px] uppercase text-ink-mute">
                Blind to pilot questions
              </dt>
              <dd className="text-ink">
                {freeze.freeze_blind_to_pilot_questions ? "yes" : "no"}
                {freeze.note_pilot_may_have_been_inspected_previously
                  ? " (prior pilot inspection → advisory)"
                  : ""}
              </dd>
            </div>
          </dl>
          <p className="mt-3 text-xs text-ink-mute">
            Experiments: freeze{" "}
            <ExpLink id={dashboard.baseline_experiment_id} /> · pilot{" "}
            <ExpLink id={dashboard.pilot_experiment_id} /> · scale{" "}
            <ExpLink id={dashboard.scale_experiment_id} />
          </p>
        </section>
      ) : (
        <p className="rounded-xl border border-dashed border-paper-line p-4 text-sm text-ink-mute">
          No freeze file yet. Run metrics_freeze first.
        </p>
      )}

      <section className="space-y-3">
        <h2 className="font-display text-2xl text-ink">Decision clauses</h2>
        <div className="grid gap-3">
          {(scale.clauses || []).map((c) => (
            <div
              key={c.id}
              className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-paper-line bg-paper-card px-4 py-3 shadow-panel"
            >
              <div>
                <p className="font-mono text-[11px] uppercase text-ink-mute">
                  Clause {c.id}
                </p>
                <p className="text-sm text-ink">{c.text}</p>
              </div>
              <StatusPill status={c.holds ? "HOLD" : "FAIL"} />
            </div>
          ))}
        </div>
        {scale.validity_caveat ? (
          <p className="text-sm text-ink-mute">{scale.validity_caveat}</p>
        ) : null}
      </section>

      <section className="space-y-3">
        <h2 className="font-display text-2xl text-ink">Baseline distributions</h2>
        <div className="overflow-hidden rounded-xl border border-paper-line bg-paper-card shadow-panel">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
              <tr>
                <th className="px-4 py-2">Metric</th>
                <th className="px-4 py-2">Overall</th>
                <th className="px-4 py-2">Good Fit</th>
                <th className="px-4 py-2">Potential Fit</th>
              </tr>
            </thead>
            <tbody>
              {(["M1.P", "M4.P", "M2.P", "M3.P"] as const).map((m) => {
                const row = dist[m];
                return (
                  <tr key={m} className="border-t border-paper-line">
                    <td className="px-4 py-2 font-mono text-xs">{m}</td>
                    <td className="px-4 py-2 font-mono text-xs">
                      {formatCi(row?.overall)}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs">
                      {formatCi(row?.by_fit_label?.["Good Fit"])}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs">
                      {formatCi(row?.by_fit_label?.["Potential Fit"])}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <ConditionSummaryChart dashboard={dashboard} />

      <section className="space-y-3">
        <h2 className="font-display text-2xl text-ink">Win / tie / loss</h2>
        <div className="overflow-hidden rounded-xl border border-paper-line bg-paper-card shadow-panel">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
              <tr>
                <th className="px-4 py-2">Comparison</th>
                <th className="px-4 py-2">M1.P</th>
                <th className="px-4 py-2">M2.P</th>
                <th className="px-4 py-2">M3.P</th>
                <th className="px-4 py-2">M4.P</th>
              </tr>
            </thead>
            <tbody>
              {(["B0", "B1", "B2"] as const).map((b) => {
                const row = scale.win_tie_loss?.[b] || {};
                return (
                  <tr key={b} className="border-t border-paper-line">
                    <td className="px-4 py-2">T vs {b}</td>
                    {(["M1.P", "M2.P", "M3.P", "M4.P"] as const).map((m) => (
                      <td key={m} className="px-4 py-2">
                        <StatusPill status={row[m] || "—"} />
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {(scale.comparisons || []).length > 0 ? (
        <section className="space-y-3">
          <h2 className="font-display text-2xl text-ink">
            Twelve comparisons
          </h2>
          <div className="overflow-x-auto rounded-xl border border-paper-line bg-paper-card shadow-panel">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
                <tr>
                  <th className="px-3 py-2">Base</th>
                  <th className="px-3 py-2">Metric</th>
                  <th className="px-3 py-2">Δ fav</th>
                  <th className="px-3 py-2">95% CI</th>
                  <th className="px-3 py-2">p Holm</th>
                  <th className="px-3 py-2">r_rb</th>
                  <th className="px-3 py-2">Out</th>
                </tr>
              </thead>
              <tbody>
                {scale.comparisons!.map((c, i) => (
                  <tr key={`${c.baseline}-${c.metric}-${i}`} className="border-t border-paper-line">
                    <td className="px-3 py-2">{c.baseline}</td>
                    <td className="px-3 py-2 font-mono text-xs">{c.metric}</td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {formatMetric(c.mean_favourable_diff, 4)}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      [{formatMetric(c.ci_low, 3)}, {formatMetric(c.ci_high, 3)}]
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {formatMetric(c.wilcoxon_p_holm, 4)}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {formatMetric(c.rank_biserial, 3)}
                    </td>
                    <td className="px-3 py-2">
                      <StatusPill status={c.outcome} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {figures.length > 0 ? (
        <section className="space-y-3">
          <h2 className="font-display text-2xl text-ink">Figures 50–56</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {figures.map((fig) => (
              <figure
                key={fig.id}
                className="overflow-hidden rounded-xl border border-paper-line bg-paper-card shadow-panel"
              >
                <div className="border-b border-paper-line px-4 py-2">
                  <figcaption className="text-sm text-ink">{fig.title}</figcaption>
                </div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={figureUrl(fig.url)}
                  alt={fig.title}
                  className="h-auto w-full bg-paper"
                />
              </figure>
            ))}
          </div>
        </section>
      ) : null}

      {pilot.condition_summaries ? (
        <section className="space-y-3">
          <h2 className="font-display text-2xl text-ink">
            Pilot condition means
          </h2>
          <div className="overflow-x-auto rounded-xl border border-paper-line bg-paper-card shadow-panel">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
                <tr>
                  <th className="px-3 py-2">Condition</th>
                  <th className="px-3 py-2">M1.P</th>
                  <th className="px-3 py-2">M2.P</th>
                  <th className="px-3 py-2">M3.P</th>
                  <th className="px-3 py-2">M4.P</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(pilot.condition_summaries).map(([cond, metrics]) => (
                  <tr key={cond} className="border-t border-paper-line">
                    <td className="px-3 py-2 font-medium">{cond}</td>
                    {(["M1.P", "M2.P", "M3.P", "M4.P"] as const).map((m) => (
                      <td key={m} className="px-3 py-2 font-mono text-xs">
                        {formatCi(metrics[m])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="rounded-xl border border-paper-line bg-paper-card p-5 shadow-panel">
        <h2 className="font-display text-xl text-ink">Limitations</h2>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink-soft">
          {limitations.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function ExpLink({ id }: { id: string | null }) {
  if (!id) return <span>—</span>;
  return (
    <Link
      href={`/experiments/${id}`}
      className="font-mono text-xs text-accent-deep underline-offset-2 hover:underline"
    >
      {id}
    </Link>
  );
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
