export type ExperimentSummary = {
  experiment_id: string;
  status: string | null;
  generation_mode: string | null;
  primary_dataset: string | null;
  model: string | null;
  provider: string | null;
  combinations_attempted: number | null;
  combinations_ok: number | null;
  combinations_failed: number | null;
  total_estimated_cost_usd: number | null;
  average_latency_ms: number | null;
  human_evaluation: string | null;
  final_question_count_requested: number | null;
};

export type ExperimentPair = {
  file: string;
  profile_id: string | null;
  jd_id: string | null;
  pair_id: string | null;
  fit_label: string | null;
  seniority: string | null;
  model: string | null;
  n_questions: number;
  questions: { question: string }[];
  resume_text: string | null;
  job_description_text: string | null;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_tokens: number | null;
  estimated_cost: number | null;
  fallback_used: boolean;
  generated_at: string | null;
};

export type LabelStat = {
  label: string;
  n: number;
  total_cost: number;
  total_latency_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_cost: number;
  avg_latency_ms: number;
};

export type ExperimentDetail = {
  summary: Record<string, unknown> & ExperimentSummary;
  pairs: ExperimentPair[];
  label_stats: LabelStat[];
  charts: {
    latency_ms: {
      index: number;
      pair_id: string | null;
      fit_label: string;
      value: number;
    }[];
    cost_usd: {
      index: number;
      pair_id: string | null;
      fit_label: string;
      value: number;
    }[];
    tokens: {
      index: number;
      pair_id: string | null;
      fit_label: string;
      input: number;
      output: number;
    }[];
  };
};

export type UploadMeta = {
  upload_id: string;
  seniority: string;
  label: string | null;
  resume_chars: number;
  job_chars: number;
  created_at: string;
  status: string;
  note: string;
};
