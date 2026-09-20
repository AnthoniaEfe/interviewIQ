"use client";

import { useState } from "react";
import type { FormEvent } from "react";

import { fetchUploads, uploadPairJson } from "@/lib/api";
import type { UploadMeta } from "@/lib/types";

type Props = {
  initialUploads: UploadMeta[];
};

export function UploadForm({ initialUploads }: Props) {
  const [resume, setResume] = useState("");
  const [job, setJob] = useState("");
  const [seniority, setSeniority] = useState("mid");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState(initialUploads);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await uploadPairJson({
        resume_text: resume,
        job_description_text: job,
        seniority,
        label: label || undefined,
      });
      setMessage(`${result.upload_id} stored. ${result.note}`);
      setResume("");
      setJob("");
      setLabel("");
      setUploads(await fetchUploads());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
      <form
        onSubmit={onSubmit}
        className="space-y-4 rounded-xl border border-paper-line bg-paper-card p-5 shadow-panel"
      >
        <div>
          <h2 className="font-display text-2xl text-ink">Upload CV / JD pair</h2>
          <p className="mt-1 text-sm text-ink-mute">
            Stores text for testing. Does not call Terra yet — generation stays on
            the experiment runner until the InterviewIQ pipeline API is wired.
          </p>
        </div>

        <label className="block text-sm">
          <span className="font-medium text-ink-soft">Resume / CV</span>
          <textarea
            required
            minLength={20}
            value={resume}
            onChange={(e) => setResume(e.target.value)}
            rows={10}
            className="mt-1 w-full rounded-lg border border-paper-line bg-paper px-3 py-2 font-mono text-xs text-ink outline-none ring-accent focus:ring-2"
            placeholder="Paste candidate CV text…"
          />
        </label>

        <label className="block text-sm">
          <span className="font-medium text-ink-soft">Job description</span>
          <textarea
            required
            minLength={20}
            value={job}
            onChange={(e) => setJob(e.target.value)}
            rows={10}
            className="mt-1 w-full rounded-lg border border-paper-line bg-paper px-3 py-2 font-mono text-xs text-ink outline-none ring-accent focus:ring-2"
            placeholder="Paste job description text…"
          />
        </label>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="font-medium text-ink-soft">Seniority</span>
            <select
              value={seniority}
              onChange={(e) => setSeniority(e.target.value)}
              className="mt-1 w-full rounded-lg border border-paper-line bg-paper px-3 py-2 text-sm outline-none ring-accent focus:ring-2"
            >
              <option value="junior">junior</option>
              <option value="mid">mid</option>
              <option value="senior">senior</option>
            </select>
          </label>
          <label className="block text-sm">
            <span className="font-medium text-ink-soft">Optional label</span>
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              className="mt-1 w-full rounded-lg border border-paper-line bg-paper px-3 py-2 text-sm outline-none ring-accent focus:ring-2"
              placeholder="manual-test"
            />
          </label>
        </div>

        <button
          type="submit"
          disabled={busy}
          className="rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white transition hover:bg-accent-deep disabled:opacity-60"
        >
          {busy ? "Saving…" : "Store pair"}
        </button>

        {message ? (
          <p className="rounded-lg bg-accent-soft px-3 py-2 text-sm text-accent-deep">
            {message}
          </p>
        ) : null}
        {error ? (
          <p className="rounded-lg bg-warn-soft px-3 py-2 text-sm text-warn">{error}</p>
        ) : null}
      </form>

      <aside className="rounded-xl border border-paper-line bg-paper-card p-5 shadow-panel">
        <h3 className="font-display text-lg text-ink">Recent uploads</h3>
        <p className="mt-1 text-xs text-ink-mute">
          Files land in experiments/datasets/uploads/
        </p>
        <ul className="mt-4 space-y-3">
          {uploads.length === 0 ? (
            <li className="text-sm text-ink-mute">No uploads yet.</li>
          ) : (
            uploads.slice(0, 12).map((item) => (
              <li
                key={item.upload_id}
                className="rounded-lg border border-paper-line bg-paper px-3 py-2"
              >
                <p className="font-mono text-xs text-ink">{item.upload_id}</p>
                <p className="mt-1 text-xs text-ink-mute">
                  {item.seniority}
                  {item.label ? ` · ${item.label}` : ""} · {item.resume_chars} /{" "}
                  {item.job_chars} chars
                </p>
              </li>
            ))
          )}
        </ul>
      </aside>
    </div>
  );
}
