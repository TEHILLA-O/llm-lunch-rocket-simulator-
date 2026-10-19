export type RewardParts = {
  distance: number;
  speed: number;
  upright: number;
  landing_bonus: number;
  crash_penalty: number;
};

export type Telemetry = {
  x_m: number;
  altitude_m: number;
  vx_mps: number;
  vy_mps: number;
  speed_mps: number;
  tilt_rad: number;
  tilt_deg: number;
  signed_angle_rad: number;
  omega_radps: number;
  action: number | null;
  action_name: string;
  step_reward: number;
  episode_reward: number;
  distance_m: number;
  landing_status: string;
  controller: string;
  left_contact: boolean;
  right_contact: boolean;
  step: number;
};

export type Scene = {
  x: number;
  y: number;
  angle: number;
  body: number[][];
  legs: { hip: number[]; foot: number[]; contact: boolean }[];
  flames: Record<string, { active: boolean; origin: number[]; direction: number[] }>;
  outcome: string | null;
  pad: { x: number; half_width: number; y: number };
};

export type Frame = {
  type: "frame";
  telemetry: Telemetry;
  reward_parts: RewardParts;
  geometry: Scene;
  cumulative_reward: number;
  trail: number[][];
  done: boolean;
  paused: boolean;
  speed: number;
  model_label: string;
};

export type Hello = {
  type: "hello";
  session_id: string;
  controller: string;
  model_label: string;
  reward_mode: string;
  reward_label: string;
  demo: string | null;
  dt: number;
  terrain: { x: number; y: number }[];
  pad: { x: number; half_width: number };
  env_version: string;
};

export type Experiment = {
  id: string;
  name: string;
  seed: number;
  status: string;
  env_version: string;
  config: {
    reward: { mode: string; label?: string; distance_weight: number; speed_weight: number; upright_weight: number };
    physics: { gravity: number; main_accel: number; attitude_torque: number };
    milestones?: number[];
  };
  timesteps_target: number;
  timesteps_completed: number;
  error: string | null;
  created_at: string;
};

export type Checkpoint = {
  id: string;
  experiment_id: string;
  timesteps: number;
  path: string;
  metrics: { milestone?: number | null; evaluation?: Record<string, number> } | null;
  experiment_name?: string;
};

export type TrajectorySummary = {
  id: string;
  controller: string;
  label: string;
  outcome: string;
  seed: number | null;
  summary: {
    episode_reward?: number;
    physics_steps?: number;
    final_distance_m?: number;
    final_speed_mps?: number;
  };
  created_at: string;
};

export type TrajectoryStep = {
  step: number;
  observation: number[];
  raw: Record<string, number | boolean>;
  action: number | null;
  action_name: string | null;
  reward: number;
  reward_parts: RewardParts;
  cumulative_reward: number;
  distance_m: number;
  speed_mps: number;
  terminated: boolean;
  truncated: boolean;
  outcome: string | null;
  scene?: Scene;
};

export type Trajectory = {
  id?: string;
  controller: string;
  label: string;
  outcome: string;
  steps: TrajectoryStep[];
  summary?: TrajectorySummary["summary"];
  config?: { physics?: Record<string, number> };
  terrain?: { x: number; y: number }[];
  pad?: { x: number; half_width: number };
};

export type Analytics = {
  episodes: {
    timestep: number;
    episode_reward: number;
    episode_length: number;
    outcome: string;
    final_distance_m: number | null;
    final_speed_mps: number | null;
  }[];
  rolling_reward: { timestep: number; value: number }[];
  success_rate: { timestep: number; value: number }[];
  crash_rate: { timestep: number; value: number }[];
  mean_episode_length: number | null;
  mean_landing_speed_mps: number | null;
  mean_final_distance_m: number | null;
  timesteps_completed: number;
  landing_count: number;
  evaluations: {
    id: string;
    label: string;
    timesteps: number | null;
    metrics: Record<string, number | string>;
  }[];
};
