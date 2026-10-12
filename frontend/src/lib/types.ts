export type ExperimentSummary = {
  experiment_id: string;
  status: string | null;
  stage?: string | null;
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
  spec_sha256?: string | null;
  decision_verdict?: string | null;
  post_hoc?: boolean;
};

export type PairMetrics = {
  "M1.P"?: number | null;
  "M2.P"?: number | null;
  "M3.P"?: number | null;
  "M4.P"?: number | null;
  "M1.S1"?: number | null;
  "M3.S2"?: number | null;
  tau?: number | null;
};

export type ExperimentPair = {
  file: string;
  profile_id: string | null;
  jd_id: string | null;
  pair_id: string | null;
  fit_label: string | null;
  seniority: string | null;
  model: string | null;
  condition?: string | null;
  n_questions: number;
  questions: { question: string }[];
  metrics?: PairMetrics | null;
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
    metrics?: {
      index: number;
      pair_id: string | null;
      fit_label: string;
      "M1.P"?: number | null;
      "M2.P"?: number | null;
      "M3.P"?: number | null;
      "M4.P"?: number | null;
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

export type MetricCi = {
  mean: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n?: number | null;
};

export type MetricsDashboard = {
  freeze: {
    path: string;
    frozen_at_utc: string | null;
    git_commit: string | null;
    spec_sha256: string | null;
    n_tau: number;
    baseline_result_dirs?: string[];
    freeze_blind_to_pilot_questions?: boolean;
    note_pilot_may_have_been_inspected_previously?: boolean;
    post_hoc?: boolean;
  } | null;
  baseline_experiment_id: string | null;
  pilot_experiment_id: string | null;
  scale_experiment_id: string | null;
  baseline: {
    counts?: Record<string, number> | null;
    distributions?: Record<
      string,
      { overall?: MetricCi; by_fit_label?: Record<string, MetricCi> }
    > | null;
    validity?: Record<string, unknown> | null;
    primary30_noise_floor?: Record<string, { mean_abs_replicate_diff?: number | null }> | null;
    spec_sha256?: string | null;
    embedding_model?: string | null;
  };
  pilot: {
    condition_summaries?: Record<string, Record<string, MetricCi & { n_pairs?: number }>> | null;
    n_primary_pairs?: number | null;
    pilot_dirs?: Record<string, string | null> | null;
    freeze_note?: string | null;
  };
  scale: {
    verdict?: string | null;
    advisory?: boolean;
    clauses?: {
      id: number;
      text: string;
      holds: boolean;
      wins?: number;
      losses?: number;
      wins_M2_M3?: number;
      projected_cost_cny?: number | null;
      scale_budget_cny?: number | null;
    }[];
    win_tie_loss?: Record<string, Record<string, string>> | null;
    comparisons?: {
      baseline: string;
      metric: string;
      mean_favourable_diff: number | null;
      ci_low: number | null;
      ci_high: number | null;
      wilcoxon_p: number | null;
      wilcoxon_p_holm: number | null;
      rank_biserial: number | null;
      outcome: string;
      n_pairs: number;
    }[];
    cost_projection?: Record<string, unknown> | null;
    diagnostics?: Record<string, unknown> | null;
    noise_floor_V4?: Record<string, unknown> | null;
    validity_caveat?: string | null;
    post_hoc?: boolean;
    scale_budget_cny?: number | null;
  };
  figures: { id: string; filename: string; url: string; title: string }[];
  limitations: string[];
};

export type AssignmentAuditRow = {
  pair_id: string;
  question_text: string;
  requirement_text: string;
  rank: string;
  cosine: string;
  linked: string;
  human_judges_related: string;
  notes: string;
};

export type QuestionRating = {
  pair_id?: string;
  condition?: string;
  question_index?: number;
  question_text?: string;
  relevance?: number | null;
  clarity?: number | null;
  specificity?: number | null;
  notes?: string;
};

export type QuestionRatingsPayload = {
  schema_version: number;
  status: string;
  note: string;
  updated_at_utc?: string;
  rater_id?: string | null;
  ratings: QuestionRating[];
};
