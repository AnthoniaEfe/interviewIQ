import type { ExperimentDetail, ExperimentSummary, UploadMeta } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

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

export function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}
