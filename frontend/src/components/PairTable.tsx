"use client";

import { useMemo, useState } from "react";

import type { ExperimentPair } from "@/lib/types";
import { formatMs, formatUsd } from "@/lib/api";

type Props = {
  pairs: ExperimentPair[];
};

export function PairTable({ pairs }: Props) {
  const [selected, setSelected] = useState<string | null>(pairs[0]?.file ?? null);
  const active = useMemo(
    () => pairs.find((p) => p.file === selected) ?? null,
    [pairs, selected],
  );

  if (!pairs.length) {
    return (
      <p className="rounded-xl border border-dashed border-paper-line bg-paper-card p-6 text-sm text-ink-mute">
        No pair results in this experiment.
      </p>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="overflow-hidden rounded-xl border border-paper-line bg-paper-card shadow-panel">
        <div className="border-b border-paper-line px-4 py-3">
          <h3 className="font-display text-lg text-ink">Pairs</h3>
          <p className="text-xs text-ink-mute">
            Select a row to inspect generated questions
          </p>
        </div>
        <div className="max-h-[420px] overflow-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="sticky top-0 bg-paper-card font-mono text-[11px] uppercase tracking-wide text-ink-mute">
              <tr>
                <th className="px-3 py-2">Pair</th>
                <th className="px-3 py-2">Fit</th>
                <th className="px-3 py-2">Latency</th>
                <th className="px-3 py-2">Cost</th>
                <th className="px-3 py-2">Qs</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((pair) => {
                const id = pair.pair_id || pair.profile_id || pair.file;
                const short = id.length > 18 ? `${id.slice(0, 16)}…` : id;
                const activeRow = pair.file === selected;
                return (
                  <tr
                    key={pair.file}
                    onClick={() => setSelected(pair.file)}
                    className={`cursor-pointer border-t border-paper-line transition ${
                      activeRow ? "bg-accent-soft/70" : "hover:bg-paper"
                    }`}
                  >
                    <td className="px-3 py-2 font-mono text-xs text-ink">{short}</td>
                    <td className="px-3 py-2 text-ink-soft">
                      {pair.fit_label || "—"}
                    </td>
                    <td className="px-3 py-2">{formatMs(pair.latency_ms)}</td>
                    <td className="px-3 py-2">{formatUsd(pair.estimated_cost)}</td>
                    <td className="px-3 py-2">{pair.n_questions}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-paper-line bg-paper-card p-4 shadow-panel">
        {active ? (
          <>
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-accent-deep">
              Generated questions
            </p>
            <h3 className="mt-1 break-all font-display text-lg text-ink">
              {active.pair_id || active.profile_id}
            </h3>
            <p className="mt-1 text-xs text-ink-mute">
              {active.fit_label || "unlabeled"} · {formatMs(active.latency_ms)} ·{" "}
              {formatUsd(active.estimated_cost)}
            </p>
            <ol className="mt-4 max-h-[360px] space-y-3 overflow-auto pr-1">
              {active.questions.map((q, index) => (
                <li key={index} className="text-sm leading-relaxed text-ink-soft">
                  <span className="mr-2 font-mono text-xs text-ink-mute">
                    {index + 1}.
                  </span>
                  {q.question}
                </li>
              ))}
            </ol>
          </>
        ) : (
          <p className="text-sm text-ink-mute">Select a pair.</p>
        )}
      </div>
    </div>
  );
}
