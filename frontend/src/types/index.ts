export interface Run {
  run_id: string;
  experiment_name: string;
  run_name: string;
  path: string;
  status: 'running' | 'completed' | 'failed' | 'stopped' | 'unknown';
  config: RunConfig;
  has_metrics: boolean;
  has_gif: boolean;
  has_repair: boolean;
  checkpoints_count: number;
  archived: boolean;
}

export interface RunConfig {
  experiment_name: string;
  seed: number;
  num_agents: number;
  num_landmarks: number;
  num_env_steps: number;
  episode_length: number;
  n_rollout_threads: number;
  eval_interval: number;
  disable_messages: boolean;
  eval_disable_messages: boolean;
  eval_noise_std: number;
  use_eval: boolean;
}

export interface RepairConfig {
  checkpoint_name: string;
  mirror_scope: 'partner_full' | 'partner' | 'all';
  repair_target: 'auto' | 'embedding' | 'comm' | 'lora' | 'full' | 'noncomm';
  controller: 'causal' | 'reward_only';
  trigger_mode?: 'meta_causal' | 'and_gate';
  cost_lambda?: number;
  causal_attribution_threshold?: number;
  severe_reward_ratio?: number;
  holdout_episodes?: number;
  lora_rank?: number;
  accept_comm_recovery?: number;
  measure_episodes: number;
  repair_iters: number;
  seed: number;
}

export interface IterationLoss {
  iter: number;
  policy_loss: number;
  value_loss: number;
}

export interface RepairVerdict {
  has_logs: boolean;
  detector_fired: boolean | null;
  decision_type: 'FRUGAL_ABSTENTION' | 'CAUSAL_CONFIRMED' | 'EMERGENCY_TRIGGER' | 'TRIGGERED' | 'NOT_DEGRADED' | 'NO_RUNS' | 'UNKNOWN';
  decision_summary: string | null;
  trigger_mode: string;
  reward_drop_ratio: number | null;
  causal_attribution_ratio: number | null;
  causal_evidence_score: number | null;
  expected_recovery: number | null;
  compute_cost: number | null;
  roi: number | null;
  roi_positive: boolean | null;
  baseline_reward: number | null;
  baseline_comm_effect: number | null;
  baseline_value_sens?: number | null;
  baseline_kl?: number | null;
  degraded_reward: number | null;
  degraded_comm_effect: number | null;
  degraded_value_sens?: number | null;
  degraded_kl?: number | null;
  repaired_reward: number | null;
  repaired_comm_effect: number | null;
  repaired_value_sens?: number | null;
  repaired_kl?: number | null;
  iteration_losses?: IterationLoss[];
  repair_target: string | null;
  accepted: boolean | null;
  accepted_target: string | null;
  reward_recovery_pct: number | null;
  comm_recovery_pct: number | null;
  retention_loss_pct: number | null;
  heldout_validation: 'CONFIRMED' | 'FAILED' | 'NONE';
  heldout_reward_recovery: number | null;
  trainable_params?: number | null;
  total_params?: number | null;
  escalation_steps?: EscalationStep[];
}

export interface EscalationStep {
  target: string;
  status: 'RUNNING' | 'ACCEPTED' | 'ROLLBACK' | 'REJECTED';
  trainable_params: number;
  pct_params: number;
  reward_recovery: number | null;
  comm_recovery: number | null;
}

export interface RunTrajectoryData {
  run_id: string;
  num_agents: number;
  num_landmarks: number;
  seed: number;
  latest_comm_effect: number | null;
  latest_kl: number | null;
  latest_reward: number | null;
  repaired_reward: number | null;
  has_repair: boolean;
  is_live_run: boolean;
  landmarks: { id: number; x: number; y: number }[];
  steps: {
    step: number;
    agents: {
      id: number;
      x: number;
      y: number;
      vx: number;
      vy: number;
      emitted_token: number;
      target_lm: number;
    }[];
  }[];
}

export interface BenchmarkConfig {
  scales: string[];
  arms: string[];
  seeds: number[];
  repair_iters: number;
  measure_episodes: number;
  holdout_episodes: number;
}

export interface BenchmarkStatus {
  running: boolean;
  pid: number | null;
  logs: string;
}

export interface BenchmarkPlotInfo {
  id: string;
  filename: string;
  url: string;
  title: string;
}

export interface BenchmarkSummary {
  has_data: boolean;
  rows: Record<string, any>[];
  metrics_summary_md: string;
  plots: BenchmarkPlotInfo[];
}

export interface SystemInfo {
  cuda_available: boolean;
  device_count: number;
  device_name: string;
  torch_version: string;
  active_run_id: string | null;
  process_type: 'training' | 'repair' | null;
  status: 'busy' | 'idle';
}

export interface MetricRow {
  step?: number;
  episode?: number;
  reward?: number;
  eval_reward?: number;
  causal_influence_kl_mean?: number;
  causal_influence_value_sensitivity_mean?: number;
  comm_effect?: number;
  [key: string]: any;
}

export interface CheckpointInfo {
  name: string;
  path: string;
  step: number;
  is_best: boolean;
}

export interface TrajectoryStep {
  step: number;
  agents: {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    emitted_token: number; // 0 - 4
    perceived_partner_x?: number; // perturbed position
    perceived_partner_y?: number;
  }[];
  landmarks: {
    id: number;
    x: number;
    y: number;
  }[];
}
