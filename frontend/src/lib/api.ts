import type {
  AssignmentAuditRow,
  ExperimentDetail,
  ExperimentSummary,
  MetricsDashboard,
  QuestionRatingsPayload,
  UploadMeta,
} from "./types";

/**
 * Resolve the FastAPI base URL.
 *
 * - Browser / Vercel: must be the *public* RunPod (or other) backend URL via
 *   NEXT_PUBLIC_API_URL — never localhost when the UI is deployed remotely.
 * - Server components inside Docker can optionally use INTERNAL_API_URL
 *   (e.g. http://backend:8000) while the browser still uses the public URL.
 */
function resolveApiBase(): string {
  const internal =
    typeof window === "undefined"
      ? process.env.INTERNAL_API_URL?.trim()
      : undefined;
  const publicUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
  const raw = (internal || publicUrl || "http://127.0.0.1:8000").replace(
    /\/$/,
    "",
  );

  if (
    typeof window !== "undefined" &&
    process.env.NODE_ENV === "production" &&
    /^(https?:\/\/)?(localhost|127\.0\.0\.1)(:|\/|$)/i.test(raw)
  ) {
    console.error(
      "[InterviewIQ] NEXT_PUBLIC_API_URL points at localhost in production. " +
        "Set it to your public RunPod FastAPI URL and redeploy.",
    );
  }
  return raw;
}

export const API_BASE = resolveApiBase();

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const data = await getJson<{ experiments: ExperimentSummary[] }>(
    "/api/v1/experiments",
  );
  return data.experiments;
}

export async function fetchExperiment(
  experimentId: string,
): Promise<ExperimentDetail> {
  return getJson<ExperimentDetail>(`/api/v1/experiments/${experimentId}`);
}

export async function fetchMetricsDashboard(): Promise<MetricsDashboard> {
  return getJson<MetricsDashboard>("/api/v1/experiments/metrics/dashboard");
}

export function figureUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http")) return pathOrUrl;
  return `${API_BASE}${pathOrUrl}`;
}

export async function fetchAssignmentAudit(): Promise<{
  path: string;
  n_rows: number;
  n_filled: number;
  rows: AssignmentAuditRow[];
  instructions: string;
}> {
  return getJson("/api/v1/experiments/human-eval/assignment-audit");
}

export async function saveAssignmentAudit(rows: AssignmentAuditRow[]) {
  return putJson<{ status: string; n_rows: number; n_filled: number }>(
    "/api/v1/experiments/human-eval/assignment-audit",
    { rows },
  );
}

export async function fetchQuestionRatings(): Promise<QuestionRatingsPayload> {
  return getJson("/api/v1/experiments/human-eval/question-ratings");
}

export async function saveQuestionRatings(payload: Partial<QuestionRatingsPayload>) {
  return putJson<{ status: string; data: QuestionRatingsPayload }>(
    "/api/v1/experiments/human-eval/question-ratings",
    payload,
  );
}

export async function fetchUploads(): Promise<UploadMeta[]> {
  const data = await getJson<{ uploads: UploadMeta[] }>(
    "/api/v1/experiments/uploads/list",
  );
  return data.uploads;
}

export async function uploadPairJson(payload: {
  resume_text: string;
  job_description_text: string;
  seniority: string;
  label?: string;
}): Promise<{ upload_id: string; note: string }> {
  const response = await fetch(`${API_BASE}/api/v1/experiments/uploads/json`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Upload failed: ${response.status}`);
  }
  return response.json();
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `$${value.toFixed(value >= 0.1 ? 3 : 4)}`;
}

export function formatCny(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `¥${value.toFixed(2)}`;
}

export function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

export function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatCi(ci?: {
  mean?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
} | null): string {
  if (!ci || ci.mean == null) return "—";
  const low = ci.ci_low == null ? "?" : ci.ci_low.toFixed(3);
  const high = ci.ci_high == null ? "?" : ci.ci_high.toFixed(3);
  return `${ci.mean.toFixed(3)} [${low}, ${high}]`;
}

export function stageLabel(stage: string | null | undefined): string {
  if (!stage) return "—";
  return stage.replace(/_/g, " ");
}
