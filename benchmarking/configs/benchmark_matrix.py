"""
Benchmark Configuration Matrix for MACPPO Causal Emergent Communication.
Defines scenarios, agent scales, control arms, and evaluation parameters.
"""

from typing import Dict, Any, List

# Scenario & Agent Scale Configurations
SCENARIO_SCALES: Dict[str, Dict[str, Any]] = {
    "2a3l": {
        "name": "2Agents_3Landmarks",
        "scenario_name": "simple_spread",
        "num_agents": 2,
        "num_landmarks": 3,
        "description": "Minimal cooperative navigation baseline with low agent contention.",
        "default_train_steps": 500_000,
        "mirror_scope": "partner_full",
    },
    "3a4l": {
        "name": "3Agents_4Landmarks",
        "scenario_name": "simple_spread",
        "num_agents": 3,
        "num_landmarks": 4,
        "description": "Standard agentic coordination contention; communication is vital.",
        "default_train_steps": 500_000,
        "mirror_scope": "partner_full",
    },
    "4a5l": {
        "name": "4Agents_5Landmarks",
        "scenario_name": "simple_spread",
        "num_agents": 4,
        "num_landmarks": 5,
        "description": "High-density multi-agent contention; requires robust protocol routing.",
        "default_train_steps": 750_000,
        "mirror_scope": "partner_full",
    },
    "4a6l": {
        "name": "4Agents_6Landmarks",
        "scenario_name": "simple_spread",
        "num_agents": 4,
        "num_landmarks": 6,
        "description": "Over-complete landmark spread under high coordination requirements.",
        "default_train_steps": 750_000,
        "mirror_scope": "partner_full",
    },
}

# Control Arms for Peer-Review Credibility
CONTROL_ARMS: Dict[str, Dict[str, Any]] = {
    "causal_adaptive": {
        "name": "Causal Adaptive Repair (Ours)",
        "controller": "causal",
        "repair_target": None,  # Rule-based auto selection: embedding -> comm -> lora -> full
        "description": "Smart causal trigger (Reward + Comm collapse) with parameter-efficient escalation and fresh-episode held-out validation.",
    },
    "naive_reward_only": {
        "name": "Naive Reward-Only Trigger",
        "controller": "reward_only",
        "repair_target": "lora",
        "description": "Ablation: Triggers repair whenever reward drops >= 20%, agnostic to communication status.",
    },
    "noncomm_repair": {
        "name": "Non-Comm Parameter Repair",
        "controller": "causal",
        "repair_target": "noncomm",
        "description": "Ablation: Freezes all communication parameters; adapts only the motor action output layer.",
    },
    "no_repair": {
        "name": "No-Repair Baseline",
        "controller": "causal",
        "repair_target": "none",
        "description": "Lower bound: Perturbed environment without any online intervention.",
    },
}

# Default Evaluation Hyperparameters
EVAL_CONFIG: Dict[str, Any] = {
    "default_seeds": [1, 2, 3, 4, 5],
    "measure_episodes": 6,
    "repair_iters": 15,
    "lora_rank": 4,
    "lora_alpha": 8.0,
    "n_rollout_threads": 4,
    "n_eval_rollout_threads": 1,
    "reward_drop_threshold": 0.10,
    "accept_reward_recovery": 0.30,
    "accept_comm_recovery": 0.15,
}
