import json
import csv
from pathlib import Path

orig_p = Path("experimentation/results/benchmark_results.json")
dest_json = Path("benchmarking/data/results/benchmark_master.json")
dest_csv = Path("benchmarking/data/results/benchmark_master.csv")

orig_records = []
if orig_p.exists():
    with open(orig_p, "r") as f:
        orig_records = json.load(f)

existing_records = []
if dest_json.exists():
    try:
        with open(dest_json, "r") as f:
            existing_records = json.load(f)
    except Exception:
        existing_records = []

unified = []
# Normalize and import all 20 existing benchmark records
for r in orig_records:
    scale = "2a3l" if r.get("num_agents") == 2 else "3a4l"
    arm = "causal_adaptive" if "Causal" in r.get("controller_type", "") else "naive_reward_only"
    norm = {
        "run_id": f"{scale}_seed{r.get('seed')}_{arm}",
        "scale": scale,
        "scale_name": r.get("config_name", f"{r.get('num_agents')}Agents_{r.get('num_landmarks')}Landmarks"),
        "num_agents": r.get("num_agents"),
        "num_landmarks": r.get("num_landmarks"),
        "seed": r.get("seed"),
        "arm": arm,
        "arm_name": "Causal Adaptive Repair (Ours)" if arm == "causal_adaptive" else "Naive Reward-Only Trigger",
        "checkpoint": r.get("selected_checkpoint", "checkpoint_best"),
        "baseline_reward": r.get("baseline_reward"),
        "baseline_no_msg_reward": r.get("baseline_no_msg_reward"),
        "baseline_comm_effect": r.get("baseline_comm_effect"),
        "baseline_value_sens": r.get("baseline_value_sens"),
        "baseline_kl": r.get("baseline_kl"),
        "degraded_reward": r.get("degraded_reward"),
        "degraded_no_msg_reward": r.get("degraded_no_msg_reward"),
        "degraded_comm_effect": r.get("degraded_comm_effect"),
        "degraded_value_sens": r.get("degraded_value_sens"),
        "degraded_kl": r.get("degraded_kl"),
        "detector_fired": r.get("detector_fired", False),
        "reward_drop_ratio": r.get("reward_drop_ratio"),
        "comm_drop_ratio": r.get("comm_drop_ratio"),
        "repair_target": r.get("repair_target", "none"),
        "repaired_reward": r.get("repaired_reward"),
        "repaired_comm_effect": r.get("repaired_comm_effect"),
        "repaired_value_sens": r.get("repaired_value_sens"),
        "repaired_kl": r.get("repaired_kl"),
        "reward_recovery_pct": r.get("reward_recovery_pct"),
        "comm_recovery_pct": r.get("comm_recovery_pct"),
        "repair_decision": r.get("repair_decision", "NO_REPAIR_TRIGGERED"),
        "heldout_validation": r.get("heldout_validation", "N/A"),
        "heldout_reward_recovery": r.get("heldout_reward_recovery"),
        "heldout_comm_recovery": r.get("heldout_comm_recovery"),
        "normal_reward_retention_loss": r.get("normal_reward_retention_loss")
    }
    unified.append(norm)

def reconcile_record(norm):
    """
    Ensure mathematical reconciliation across recovery percentages:
    recovery = (repaired - degraded) / (baseline - degraded)
    Fixes discrepancies where recorded percentage diverges from reported raw rewards.
    """
    if (norm.get("repaired_reward") is not None and
        norm.get("baseline_reward") is not None and
        norm.get("degraded_reward") is not None):
        lost_rew = norm["baseline_reward"] - norm["degraded_reward"]
        if abs(lost_rew) > 1e-6:
            norm["reward_recovery_pct"] = round(
                ((norm["repaired_reward"] - norm["degraded_reward"]) / lost_rew) * 100.0, 1
            )

    if (norm.get("repaired_comm_effect") is not None and
        norm.get("baseline_comm_effect") is not None and
        norm.get("degraded_comm_effect") is not None):
        lost_comm = norm["baseline_comm_effect"] - norm["degraded_comm_effect"]
        if lost_comm > 1e-6:
            norm["comm_recovery_pct"] = round(
                ((norm["repaired_comm_effect"] - norm["degraded_comm_effect"]) / lost_comm) * 100.0, 1
            )
    return norm

# Reconcile all records
unified = [reconcile_record(r) for r in unified]
existing_records = [reconcile_record(r) for r in existing_records]

master_dict = {r["run_id"]: r for r in unified}
for r in existing_records:
    master_dict[r["run_id"]] = r
unified = list(master_dict.values())

dest_json.parent.mkdir(parents=True, exist_ok=True)
with open(dest_json, "w") as f:
    json.dump(unified, f, indent=2)

if unified:
    fieldnames = list(dict.fromkeys([k for r in unified for k in r.keys()]))
    with open(dest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(unified)

print(f"Successfully consolidated {len(unified)} records into {dest_json} and {dest_csv}")
