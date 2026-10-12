"use client";

import { useMemo, useState, useTransition } from "react";

import type { AssignmentAuditRow, QuestionRatingsPayload } from "@/lib/types";
import {
  saveAssignmentAudit,
  saveQuestionRatings,
} from "@/lib/api";

type Props = {
  auditPath: string;
  instructions: string;
  initialRows: AssignmentAuditRow[];
  initialRatings: QuestionRatingsPayload;
};

export function HumanEvalPanel({
  auditPath,
  instructions,
  initialRows,
  initialRatings,
}: Props) {
  const [rows, setRows] = useState(initialRows);
  const [ratingsNote, setRatingsNote] = useState(initialRatings.note || "");
  const [raterId, setRaterId] = useState(initialRatings.rater_id || "");
  const [status, setStatus] = useState(initialRatings.status || "not_started");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  const filled = useMemo(
    () =>
      rows.filter((r) => ["0", "1"].includes(String(r.human_judges_related || "").trim()))
        .length,
    [rows],
  );

  function updateRow(index: number, patch: Partial<AssignmentAuditRow>) {
    setRows((prev) =>
      prev.map((row, i) => (i === index ? { ...row, ...patch } : row)),
    );
  }

  function onSaveAudit() {
    setError(null);
    setMessage(null);
    startTransition(async () => {
      try {
        const res = await saveAssignmentAudit(rows);
        setMessage(
          `Assignment audit saved (${res.n_filled}/${res.n_rows} judged).`,
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Save failed");
      }
    });
  }

  function onSaveScaffold() {
    setError(null);
    setMessage(null);
    startTransition(async () => {
      try {
        await saveQuestionRatings({
          status,
          note: ratingsNote,
          rater_id: raterId || null,
          ratings: initialRatings.ratings || [],
        });
        setMessage("Question-rating scaffold metadata saved.");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Save failed");
      }
    });
  }

  return (
    <div className="space-y-8">
      {error ? (
        <div className="rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          {error}
        </div>
      ) : null}
      {message ? (
        <div className="rounded-xl border border-accent/20 bg-accent-soft px-4 py-3 text-sm text-accent-deep">
          {message}
        </div>
      ) : null}

      <section className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-2xl text-ink">
              Assignment audit (V5)
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-ink-mute">{instructions}</p>
            <p className="mt-1 font-mono text-xs text-ink-mute">
              {auditPath} · filled {filled}/{rows.length}
            </p>
          </div>
          <button
            type="button"
            onClick={onSaveAudit}
            disabled={pending || rows.length === 0}
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-deep disabled:opacity-50"
          >
            {pending ? "Saving…" : "Save audit judgments"}
          </button>
        </div>

        {rows.length === 0 ? (
          <p className="rounded-xl border border-dashed border-paper-line p-6 text-sm text-ink-mute">
            No assignment_audit.csv yet. Run metrics_freeze to generate the blank sheet.
          </p>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-paper-line bg-paper-card shadow-panel">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-paper font-mono text-[11px] uppercase tracking-wide text-ink-mute">
                <tr>
                  <th className="px-3 py-2">#</th>
                  <th className="px-3 py-2">Question</th>
                  <th className="px-3 py-2">Requirement</th>
                  <th className="px-3 py-2">Linked</th>
                  <th className="px-3 py-2">Related?</th>
                  <th className="px-3 py-2">Notes</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={`${row.pair_id}-${index}`} className="border-t border-paper-line align-top">
                    <td className="px-3 py-2 font-mono text-xs text-ink-mute">
                      {index + 1}
                    </td>
                    <td className="max-w-xs px-3 py-2 text-ink-soft">
                      {row.question_text}
                    </td>
                    <td className="max-w-[12rem] px-3 py-2 text-ink-soft">
                      {row.requirement_text}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {String(row.linked)}
                    </td>
                    <td className="px-3 py-2">
                      <select
                        className="rounded border border-paper-line bg-paper px-2 py-1 text-sm"
                        value={row.human_judges_related ?? ""}
                        onChange={(e) =>
                          updateRow(index, {
                            human_judges_related: e.target.value,
                          })
                        }
                      >
                        <option value="">—</option>
                        <option value="1">1 related</option>
                        <option value="0">0 not</option>
                      </select>
                    </td>
                    <td className="px-3 py-2">
                      <input
                        className="w-40 rounded border border-paper-line bg-paper px-2 py-1 text-sm"
                        value={row.notes ?? ""}
                        onChange={(e) =>
                          updateRow(index, { notes: e.target.value })
                        }
                        placeholder="optional"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-xl border border-paper-line bg-paper-card p-5 shadow-panel">
        <h2 className="font-display text-2xl text-ink">
          Question quality ratings (scaffold)
        </h2>
        <p className="mt-2 text-sm text-ink-mute">
          Reserved for eventual human evaluation of question quality. Automatic
          frozen metrics (M1–M4) remain alignment/diversity proxies and are not
          replaced by these ratings. Pair-level rating UI can be wired here later
          without changing the frozen metric definitions.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="font-mono text-[11px] uppercase text-ink-mute">
              Status
            </span>
            <select
              className="mt-1 w-full rounded border border-paper-line bg-paper px-3 py-2"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            >
              <option value="not_started">not_started</option>
              <option value="in_progress">in_progress</option>
              <option value="complete">complete</option>
            </select>
          </label>
          <label className="block text-sm">
            <span className="font-mono text-[11px] uppercase text-ink-mute">
              Rater id
            </span>
            <input
              className="mt-1 w-full rounded border border-paper-line bg-paper px-3 py-2"
              value={raterId}
              onChange={(e) => setRaterId(e.target.value)}
              placeholder="optional initials / id"
            />
          </label>
          <label className="block text-sm sm:col-span-2">
            <span className="font-mono text-[11px] uppercase text-ink-mute">
              Protocol note
            </span>
            <textarea
              className="mt-1 min-h-[100px] w-full rounded border border-paper-line bg-paper px-3 py-2"
              value={ratingsNote}
              onChange={(e) => setRatingsNote(e.target.value)}
            />
          </label>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={onSaveScaffold}
            disabled={pending}
            className="rounded-md border border-accent bg-white px-4 py-2 text-sm font-medium text-accent-deep transition hover:bg-accent-soft disabled:opacity-50"
          >
            Save scaffold metadata
          </button>
          <p className="text-xs text-ink-mute">
            Stored ratings count: {(initialRatings.ratings || []).length} (empty
            until a protocol is launched)
          </p>
        </div>
      </section>
    </div>
  );
}
