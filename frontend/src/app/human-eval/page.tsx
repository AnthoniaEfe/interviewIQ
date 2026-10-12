import Link from "next/link";

import { HumanEvalPanel } from "@/components/HumanEvalPanel";
import { StatCard } from "@/components/StatCard";
import type { AssignmentAuditRow, QuestionRatingsPayload } from "@/lib/types";
import {
  fetchAssignmentAudit,
  fetchQuestionRatings,
} from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function HumanEvalPage() {
  let auditError: string | null = null;
  let ratingsError: string | null = null;
  let audit: {
    path: string;
    n_rows: number;
    n_filled: number;
    rows: AssignmentAuditRow[];
    instructions: string;
  } = {
    path: "experiments/metrics/assignment_audit.csv",
    n_rows: 0,
    n_filled: 0,
    rows: [],
    instructions: "",
  };
  let ratings: QuestionRatingsPayload = {
    schema_version: 1,
    status: "not_started",
    note: "",
    ratings: [],
  };

  try {
    audit = await fetchAssignmentAudit();
  } catch (err) {
    auditError = err instanceof Error ? err.message : "Failed to load audit";
  }
  try {
    ratings = await fetchQuestionRatings();
  } catch (err) {
    ratingsError = err instanceof Error ? err.message : "Failed to load ratings";
  }

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-accent-deep">
          Optional · not frozen metrics
        </p>
        <h1 className="font-display text-4xl tracking-tight text-ink">
          Human evaluation
        </h1>
        <p className="max-w-3xl text-ink-mute">
          Fill the V5 assignment audit when ready, and keep question-quality
          ratings separate from the automatic M1–M4 scale decision. See{" "}
          <Link
            href="/metrics"
            className="text-accent-deep underline-offset-2 hover:underline"
          >
            Frozen metrics
          </Link>{" "}
          for the pre-registered automatic results.
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-3">
        <StatCard label="Audit rows" value={String(audit.n_rows)} />
        <StatCard
          label="Judged"
          value={`${audit.n_filled}/${audit.n_rows || 0}`}
        />
        <StatCard
          label="Quality ratings status"
          value={ratings.status || "not_started"}
        />
      </section>

      {auditError || ratingsError ? (
        <div className="rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          {auditError || ratingsError}. Is the API running?
        </div>
      ) : null}

      <HumanEvalPanel
        auditPath={audit.path}
        instructions={
          audit.instructions ||
          "Judge whether each question is related to the listed requirement (1/0)."
        }
        initialRows={audit.rows}
        initialRatings={ratings}
      />
    </div>
  );
}
