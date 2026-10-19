"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { API_BASE } from "@/lib/api";

type Draft = {
  item_id: string;
  relevance_x: number | null;
  coverage_x: number | null;
  non_redundancy_x: number | null;
  relevance_y: number | null;
  coverage_y: number | null;
  non_redundancy_y: number | null;
  preference: string | null;
  comment: string | null;
  elapsed_ms: number | null;
};

type Item = {
  item_id: string;
  job_summary: string;
  candidate_summary: string;
  job_text: string;
  candidate_text: string;
  set_x: string[];
  set_y: string[];
};

type SessionItem = { item: Item; draft: Draft | null; complete: boolean };

const CRITERIA = [
  ["relevance", "Relevance"],
  ["coverage", "Coverage"],
  ["non_redundancy", "Non-redundancy"],
] as const;

const FORBIDDEN_KEYS = new Set([
  "condition",
  "condition_id",
  "arm",
  "treatment",
  "baseline",
  "pair_id",
  "profile_id",
  "jd_id",
]);
const FORBIDDEN_VALUES = new Set(["T", "B1", "treatment", "profile-aware"]);

function hasConditionLeak(node: unknown): boolean {
  if (Array.isArray(node)) return node.some(hasConditionLeak);
  if (node && typeof node === "object") {
    for (const [key, child] of Object.entries(node as Record<string, unknown>)) {
      if (FORBIDDEN_KEYS.has(key) || hasConditionLeak(child)) return true;
    }
    return false;
  }
  return typeof node === "string" && FORBIDDEN_VALUES.has(node);
}

const EMPTY = (itemId: string): Draft => ({
  item_id: itemId,
  relevance_x: null,
  coverage_x: null,
  non_redundancy_x: null,
  relevance_y: null,
  coverage_y: null,
  non_redundancy_y: null,
  preference: null,
  comment: "",
  elapsed_ms: 0,
});

export function StudyClient({ token, mode }: { token: string; mode: string }) {
  const [ack, setAck] = useState(false);
  const [showRubric, setShowRubric] = useState(false);
  const [items, setItems] = useState<SessionItem[]>([]);
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<number>(Date.now());
  const [background, setBackground] = useState({
    hr_or_interviewing_experience: "" as "" | "yes" | "no",
    years: "",
  });

  const load = useCallback(async () => {
    const response = await fetch(
      `${API_BASE}/api/v1/human-eval/session?token=${encodeURIComponent(token)}&mode=${mode}`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      setError(await response.text());
      return;
    }
    const payload = await response.json();
    if (hasConditionLeak(payload)) {
      setError("This link returned identifying labels. Stop and contact the study owner.");
      return;
    }
    const rows = payload.items as SessionItem[];
    setItems(rows);
    const resume = payload.resume_item_id as string | null;
    const resumeIndex = resume
      ? rows.findIndex((row) => row.item.item_id === resume)
      : 0;
    const next = resumeIndex >= 0 ? resumeIndex : 0;
    setIndex(next);
    const current = rows[next];
    setDraft(current?.draft ? { ...EMPTY(current.item.item_id), ...current.draft } : current ? EMPTY(current.item.item_id) : null);
    setStarted(Date.now());
  }, [token, mode]);

  useEffect(() => {
    const seen = window.localStorage.getItem(`study-ack:${token}`);
    if (seen === "1") setAck(true);
    void load();
  }, [load, token]);

  const current = items[index];
  const nComplete = items.filter((row) => row.complete).length;

  const save = useCallback(
    async (nextDraft: Draft) => {
      const elapsed = (nextDraft.elapsed_ms ?? 0) + (Date.now() - started);
      const body = { ...nextDraft, elapsed_ms: elapsed };
      const response = await fetch(
        `${API_BASE}/api/v1/human-eval/ratings?token=${encodeURIComponent(token)}&mode=${mode}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      if (!response.ok) {
        setError(await response.text());
        return;
      }
      const saved = await response.json();
      setItems((rows) =>
        rows.map((row) =>
          row.item.item_id === nextDraft.item_id
            ? { ...row, draft: body, complete: Boolean(saved.complete) }
            : row,
        ),
      );
    },
    [mode, started, token],
  );

  useEffect(() => {
    if (!draft) return;
    const handle = window.setTimeout(() => {
      void save(draft);
    }, 400);
    return () => window.clearTimeout(handle);
  }, [draft, save]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) return;
      if (event.key === "ArrowRight") {
        setIndex((value) => Math.min(items.length - 1, value + 1));
      }
      if (event.key === "ArrowLeft") {
        setIndex((value) => Math.max(0, value - 1));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items.length]);

  useEffect(() => {
    const row = items[index];
    if (!row) return;
    setDraft(row.draft ? { ...EMPTY(row.item.item_id), ...row.draft } : EMPTY(row.item.item_id));
    setStarted(Date.now());
    // Reload the editor only when the rater moves between items.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index]);

  const done = useMemo(
    () => items.length > 0 && items.every((row) => row.complete),
    [items],
  );

  if (error) {
    return <p className="mx-auto max-w-xl p-6 text-warn">{error}</p>;
  }
  if (!ack) {
    return (
      <main className="mx-auto max-w-xl space-y-4 p-6">
        <h1 className="font-display text-3xl text-ink">Interview question rating</h1>
        <p className="text-ink-soft">
          You will compare two sets of interview questions for the same person and job.
          Ratings are stored under a rater id only. They are used for a thesis study.
          You can stop and return with the same link; progress is saved on this server.
        </p>
        <button
          type="button"
          className="rounded-md bg-accent px-4 py-3 text-white"
          onClick={() => {
            window.localStorage.setItem(`study-ack:${token}`, "1");
            setAck(true);
          }}
        >
          Continue
        </button>
      </main>
    );
  }
  if (!current || !draft) {
    return <p className="p-6 text-ink-mute">Loading the next pair…</p>;
  }
  if (done && index >= items.length) {
    return (
      <main className="mx-auto max-w-xl space-y-3 p-6">
        <h1 className="font-display text-3xl">Thank you</h1>
        <p>Every assigned pair has a complete rating. You can close this page.</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl space-y-4 p-4 md:p-6">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm text-ink-mute">
          Pair {index + 1} of {items.length} · {nComplete} complete
        </p>
        <div className="h-2 min-w-[8rem] flex-1 rounded bg-paper-line">
          <div
            className="h-2 rounded bg-accent"
            style={{ width: `${items.length ? (nComplete / items.length) * 100 : 0}%` }}
          />
        </div>
        <p className="text-sm text-ink-mute">Time left is not estimated until the pilot is measured.</p>
      </div>
      <p className="text-sm text-ink-mute">
        Progress is saved automatically. Use this same link to come back later.
      </p>
      <section className="grid gap-3 rounded-lg border border-paper-line bg-paper-card p-4">
        <label className="text-sm">
          Optional: HR or interviewing experience
          <select
            className="ml-2 rounded border border-paper-line px-2 py-1"
            value={background.hr_or_interviewing_experience}
            onChange={async (event) => {
              const value = event.target.value as "" | "yes" | "no";
              setBackground((prev) => ({ ...prev, hr_or_interviewing_experience: value }));
              if (!value) return;
              await fetch(
                `${API_BASE}/api/v1/human-eval/background?token=${encodeURIComponent(token)}&mode=${mode}`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    hr_or_interviewing_experience: value === "yes",
                    years: background.years === "" ? null : Number(background.years),
                  }),
                },
              );
            }}
          >
            <option value="">Prefer not to say</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </label>
        <button type="button" className="text-left text-sm text-accent-deep underline" onClick={() => setShowRubric((value) => !value)}>
          {showRubric ? "Hide rubric" : "Show rubric"}
        </button>
        {showRubric ? (
          <p className="text-sm text-ink-soft">
            Relevance, coverage, and non-redundancy are each 1 to 5. 1 is a poor set on that
            criterion, 3 is mixed, 5 is strong. Then choose Set X, Set Y, or no preference.
          </p>
        ) : null}
      </section>
      <section className="grid gap-4 md:grid-cols-2">
        <article className="rounded-lg border border-paper-line bg-paper-card p-4">
          <h2 className="font-display text-xl">Job</h2>
          <p>{current.item.job_summary}</p>
          <details>
            <summary>Full job text</summary>
            <pre className="whitespace-pre-wrap text-sm">{current.item.job_text}</pre>
          </details>
        </article>
        <article className="rounded-lg border border-paper-line bg-paper-card p-4">
          <h2 className="font-display text-xl">Candidate</h2>
          <p>{current.item.candidate_summary}</p>
          <details>
            <summary>Full candidate text</summary>
            <pre className="whitespace-pre-wrap text-sm">{current.item.candidate_text}</pre>
          </details>
        </article>
      </section>
      <section className="grid gap-4 md:grid-cols-2">
        {(["x", "y"] as const).map((side) => (
          <article key={side} className="space-y-3 rounded-lg border border-paper-line bg-paper-card p-4">
            <h2 className="font-display text-2xl">Set {side.toUpperCase()}</h2>
            <ol className="list-decimal space-y-2 pl-5">
              {(side === "x" ? current.item.set_x : current.item.set_y).map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ol>
            {CRITERIA.map(([key, label]) => (
              <fieldset key={key}>
                <legend className="text-sm">{label}</legend>
                <div className="flex flex-wrap gap-2">
                  {[1, 2, 3, 4, 5].map((score) => {
                    const field = `${key}_${side}` as keyof Draft;
                    const selected = draft[field] === score;
                    return (
                      <button
                        key={score}
                        type="button"
                        aria-pressed={selected}
                        className={`h-11 w-11 rounded-md border ${selected ? "bg-accent text-white" : "bg-white"}`}
                        onClick={() =>
                          setDraft((prev) => (prev ? { ...prev, [field]: score } : prev))
                        }
                      >
                        {score}
                      </button>
                    );
                  })}
                </div>
              </fieldset>
            ))}
          </article>
        ))}
      </section>
      <fieldset className="flex flex-wrap gap-2">
        <legend className="w-full text-sm">Which set is the better interview overall?</legend>
        {[
          ["X", "Set X"],
          ["Y", "Set Y"],
          ["no_preference", "No preference"],
        ].map(([value, label]) => (
          <button
            key={value}
            type="button"
            aria-pressed={draft.preference === value}
            className={`rounded-md border px-4 py-3 ${draft.preference === value ? "bg-accent text-white" : "bg-white"}`}
            onClick={() => setDraft((prev) => (prev ? { ...prev, preference: value } : prev))}
          >
            {label}
          </button>
        ))}
      </fieldset>
      <label className="block text-sm">
        Optional comment
        <textarea
          className="mt-1 w-full rounded-md border border-paper-line p-2"
          value={draft.comment ?? ""}
          onChange={(event) =>
            setDraft((prev) => (prev ? { ...prev, comment: event.target.value } : prev))
          }
        />
      </label>
      <div className="flex gap-2">
        <button type="button" className="rounded-md border px-4 py-3" onClick={() => setIndex((value) => Math.max(0, value - 1))}>
          Back
        </button>
        <button
          type="button"
          className="rounded-md border px-4 py-3"
          onClick={() => setIndex((value) => Math.min(items.length, value + 1))}
        >
          Forward
        </button>
      </div>
      {done ? (
        <p className="rounded-md bg-accent-soft p-4">All pairs are complete. Thank you. You can close this page.</p>
      ) : null}
    </main>
  );
}
