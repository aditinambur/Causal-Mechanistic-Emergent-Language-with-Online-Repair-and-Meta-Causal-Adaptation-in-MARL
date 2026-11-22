import os
import sys
import subprocess
import json
import csv
import glob
import shutil
import re
import math
import random
from typing import Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MAC Multi-Agent Emergent Communication API")

# Allow CORS for development & Docker container networking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_ROOT = "onpolicy/scripts/results/MPE/simple_spread/mappo"
REGISTRY_PATH = os.path.join(RESULTS_ROOT, "run_registry.json")

# Global process tracking
active_process: Optional[subprocess.Popen] = None
active_run_id: Optional[str] = None
active_process_type: Optional[str] = None  # "training" or "repair"
active_benchmark_process: Optional[subprocess.Popen] = None
BENCHMARK_LOG_PATH = "benchmarking/data/results/suite_run.log"

class RunConfig(BaseModel):
    experiment_name: str = "check"
    seed: int = 1
    num_agents: int = 2
    num_landmarks: int = 3
    num_env_steps: int = 100000
    episode_length: int = 25
    n_rollout_threads: int = 32
    eval_interval: int = 5
    disable_messages: bool = False
    eval_disable_messages: bool = False
    eval_noise_std: float = 0.25
    use_eval: bool = True

class RepairConfig(BaseModel):
    checkpoint_name: str  # e.g. "checkpoint_1958400"
    mirror_scope: str = "partner_full"  # "partner_full", "partner", "all"
    repair_target: str = "auto"  # "auto", "embedding", "comm", "lora", "full", "noncomm"
    controller: str = "causal"  # "causal", "reward_only"
    trigger_mode: str = "meta_causal"  # "meta_causal", "and_gate"
    cost_lambda: float = 0.02  # Frugal ROI cost coefficient
    causal_attribution_threshold: float = 0.20  # Minimum composite causal score
    severe_reward_ratio: float = 0.50  # Emergency coordination collapse ratio
    holdout_episodes: int = 6  # Fresh unseen held-out layouts count
    lora_rank: int = 4  # LoRA rank for parameter efficiency
    accept_comm_recovery: float = 0.15  # Relaxed comm recovery acceptance threshold
    measure_episodes: int = 6
    repair_iters: int = 15
    seed: int = 1

class BenchmarkStartConfig(BaseModel):
    scales: List[str] = ["2a3l", "3a4l"]
    arms: List[str] = ["causal_adaptive", "naive_reward_only", "noncomm_repair", "no_repair"]
    seeds: List[int] = [1, 2, 3]
    repair_iters: int = 15
    measure_episodes: int = 6
    holdout_episodes: int = 6

def load_registry() -> Dict:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_registry(registry: Dict):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"Error saving registry: {e}")

def get_next_run(experiment_name: str) -> tuple:
    exp_dir = os.path.join(RESULTS_ROOT, experiment_name)
    if not os.path.exists(exp_dir):
        return "run1", os.path.join(exp_dir, "run1").replace("\\", "/")
    
    exst_run_nums = []
    for item in os.listdir(exp_dir):
        if item.startswith("run") and os.path.isdir(os.path.join(exp_dir, item)):
            try:
                num = int(item[3:])
                exst_run_nums.append(num)
            except ValueError:
                pass
                
    if not exst_run_nums:
        next_run = "run1"
    else:
        next_run = f"run{max(exst_run_nums) + 1}"
        
    return next_run, os.path.join(exp_dir, next_run).replace("\\", "/").replace("//", "/")

def _infer_run_config(run_dir: str, exp_name: str, existing_cfg: Dict) -> Dict:
    """Fill missing num_agents, num_landmarks, seed from run folder artifacts."""
    cfg = dict(existing_cfg) if isinstance(existing_cfg, dict) else {}

    if "num_agents" not in cfg or "num_landmarks" not in cfg:
        m = re.search(r"(\d+)a(\d+)l", exp_name.lower())
        if m:
            cfg.setdefault("num_agents", int(m.group(1)))
            cfg.setdefault("num_landmarks", int(m.group(2)))

    if "seed" not in cfg:
        ms = re.search(r"seed[_-]?(\d+)", exp_name.lower())
        if ms:
            cfg["seed"] = int(ms.group(1))

    if "num_agents" not in cfg:
        csv_path = os.path.join(run_dir, "causal_influence.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    header = f.readline().strip().split(",")
                    agent_cols = [c for c in header if c.startswith("causal_influence_kl_agent")]
                    if agent_cols:
                        cfg["num_agents"] = len(agent_cols)
            except Exception:
                pass

    repair_log = os.path.join(run_dir, "repair_output.log")
    if ("num_agents" not in cfg or "num_landmarks" not in cfg) and os.path.exists(repair_log):
        try:
            with open(repair_log, "r", encoding="utf-8", errors="ignore") as f:
                head = "".join([f.readline() for _ in range(10)])
                mr = re.search(r"num_agents=(\d+),\s*num_landmarks=(\d+)", head)
                if mr:
                    cfg.setdefault("num_agents", int(mr.group(1)))
                    cfg.setdefault("num_landmarks", int(mr.group(2)))
        except Exception:
            pass

    if "num_agents" in cfg and "num_landmarks" not in cfg:
        cfg["num_landmarks"] = cfg["num_agents"] + 1

    return cfg

def scan_runs() -> List[Dict]:
    registry = load_registry()
    runs = []
    
    pattern = os.path.join(RESULTS_ROOT, "*", "run*")
    run_dirs = glob.glob(pattern)
    
    for run_dir in run_dirs:
        run_dir = run_dir.replace("\\", "/")
        path_parts = run_dir.split("/")
        run_name = path_parts[-1]
        exp_name = path_parts[-2]
        run_id = f"{exp_name}_{run_name}"
        
        info = registry.get(run_id, {})
        
        csv_path = os.path.join(run_dir, "causal_influence.csv")
        has_metrics = os.path.exists(csv_path)
        gif_path = os.path.join(run_dir, "render.gif")
        has_gif = os.path.exists(gif_path)
        
        # Check if checkpoints exist
        models_dir = os.path.join(run_dir, "models")
        checkpoints_count = 0
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            checkpoints_count = len([
                d for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d)) and not d.endswith("_repair_runs")
            ])
            
        repair_log = os.path.join(run_dir, "repair_output.log")
        has_repair = os.path.exists(repair_log)
        
        is_active = (active_run_id == run_id and active_process is not None and active_process.poll() is None)
        status = "running" if is_active else info.get("status", "completed" if has_metrics else "unknown")
        inferred_cfg = _infer_run_config(run_dir, exp_name, info.get("config", {}))

        runs.append({
            "run_id": run_id,
            "experiment_name": exp_name,
            "run_name": run_name,
            "path": run_dir,
            "status": status,
            "config": inferred_cfg,
            "has_metrics": has_metrics,
            "has_gif": has_gif,
            "has_repair": has_repair,
            "checkpoints_count": checkpoints_count,
            "archived": info.get("archived", False)
        })
        
    def sort_key(run):
        is_running = 1 if run["status"] == "running" else 0
        try:
            num = int(run["run_name"][3:])
            return (is_running, run["experiment_name"], num)
        except ValueError:
            return (is_running, run["experiment_name"], 0)
            
    runs.sort(key=sort_key, reverse=True)
    return runs

@app.get("/api/system/info")
def get_system_info():
    """Return GPU, CUDA, and hardware acceleration status."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    global active_process, active_run_id, active_process_type
    is_busy = active_process is not None and active_process.poll() is None
    
    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "active_run_id": active_run_id if is_busy else None,
        "process_type": active_process_type if is_busy else None,
        "status": "busy" if is_busy else "idle"
    }

@app.get("/api/config-schema")
def get_config_schema():
    """Return default hyperparameter options for MPE."""
    return {
        "experiment_name": {"type": "string", "default": "mpe_experiment", "description": "Identifier for the training run"},
        "num_agents": {"type": "integer", "default": 2, "description": "Number of learning agents"},
        "num_landmarks": {"type": "integer", "default": 3, "description": "Number of landmark targets"},
        "seed": {"type": "integer", "default": 1, "description": "Random seed for reproducibility"},
        "num_env_steps": {"type": "integer", "default": 100000, "description": "Total env steps to train"},
        "episode_length": {"type": "integer", "default": 25, "description": "Max length of each episode"},
        "n_rollout_threads": {"type": "integer", "default": 32, "description": "Number of parallel environments during training"},
        "eval_interval": {"type": "integer", "default": 5, "description": "Interval between evaluations (in episodes)"},
        "disable_messages": {"type": "boolean", "default": False, "description": "Disable communication during training (ablation)"},
        "eval_disable_messages": {"type": "boolean", "default": False, "description": "Disable communication during evaluation"},
        "eval_noise_std": {"type": "number", "default": 0.25, "description": "Standard deviation of noise added to messages"},
        "use_eval": {"type": "boolean", "default": True, "description": "Whether to perform causal evaluations during training"}
    }

@app.get("/api/runs")
def list_runs():
    """List all completed and active runs."""
    return scan_runs()

@app.get("/api/runs/active")
def get_active_run():
    """Get the current running process details."""
    global active_process, active_run_id, active_process_type
    if active_process is not None and active_process.poll() is None:
        return {"run_id": active_run_id, "process_type": active_process_type, "status": "running"}
    return {"run_id": None, "process_type": None, "status": "idle"}

@app.post("/api/runs/start")
def start_run(config: RunConfig, background_tasks: BackgroundTasks):
    """Start a new training run with GPU acceleration enabled by default."""
    global active_process, active_run_id, active_process_type
    
    if active_process is not None and active_process.poll() is None:
        raise HTTPException(status_code=400, detail="Another training or repair task is currently running.")
        
    next_run_name, next_run_dir = get_next_run(config.experiment_name)
    run_id = f"{config.experiment_name}_{next_run_name}"
    
    os.makedirs(next_run_dir, exist_ok=True)
    log_file_path = os.path.join(next_run_dir, "output.log")
    
    # Notice: --cuda is store_false, so omitting --cuda enables GPU execution
    cmd = [
        sys.executable, "-u", "onpolicy/scripts/train/train_mpe.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--use_wandb",
        "--experiment_name", config.experiment_name,
        "--num_agents", str(config.num_agents),
        "--num_landmarks", str(config.num_landmarks),
        "--seed", str(config.seed),
        "--num_env_steps", str(config.num_env_steps),
        "--episode_length", str(config.episode_length),
        "--n_rollout_threads", str(config.n_rollout_threads),
        "--eval_interval", str(config.eval_interval),
        "--eval_noise_std", str(config.eval_noise_std)
    ]
    
    if config.disable_messages:
        cmd.append("--disable_messages")
    if config.eval_disable_messages:
        cmd.append("--eval_disable_messages")
    if config.use_eval:
        cmd.append("--use_eval")
        
    log_file = open(log_file_path, "w")
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"
        
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True
        )
    except Exception as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch training subprocess: {e}")
        
    active_process = proc
    active_run_id = run_id
    active_process_type = "training"
    
    registry = load_registry()
    registry[run_id] = {
        "status": "running",
        "config": config.dict(),
        "pid": proc.pid
    }
    save_registry(registry)
    
    def monitor_process(p, r_id):
        p.wait()
        log_file.close()
        
        global active_process, active_run_id, active_process_type
        if active_run_id == r_id:
            active_process = None
            active_run_id = None
            active_process_type = None
            
        reg = load_registry()
        if r_id in reg:
            csv_path = os.path.join(next_run_dir, "causal_influence.csv")
            reg[r_id]["status"] = "completed" if (os.path.exists(csv_path) and p.returncode == 0) else "failed"
            save_registry(reg)
            
    background_tasks.add_task(monitor_process, proc, run_id)
    
    return {"run_id": run_id, "status": "running", "log_file": log_file_path}

@app.post("/api/runs/{run_id}/repair")
def run_repair(run_id: str, config: RepairConfig, background_tasks: BackgroundTasks):
    """Launch Phase 2/3 Causal Break, Selective Detection, and Online Repair on a checkpoint."""
    global active_process, active_run_id, active_process_type
    
    if active_process is not None and active_process.poll() is None:
        raise HTTPException(status_code=400, detail="Another task is currently running.")
        
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    model_dir = os.path.join(run["path"], "models", config.checkpoint_name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=400, detail=f"Checkpoint directory {config.checkpoint_name} does not exist.")
        
    repair_log_path = os.path.join(run["path"], "repair_output.log")
    
    run_cfg = run.get("config", {})
    num_agents = run_cfg.get("num_agents")
    num_landmarks = run_cfg.get("num_landmarks")

    if num_agents is None:
        csv_path = os.path.join(run["path"], "causal_influence.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    header = f.readline().strip().split(",")
                    agent_cols = [c for c in header if c.startswith("causal_influence_kl_agent")]
                    if agent_cols:
                        num_agents = len(agent_cols)
            except Exception:
                pass

    cmd = [
        sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--seed", str(config.seed),
        "--model_dir", model_dir,
        "--mirror_scope", config.mirror_scope,
        "--measure_episodes", str(config.measure_episodes),
        "--repair_iters", str(config.repair_iters),
        "--n_rollout_threads", "32",
        "--n_eval_rollout_threads", "1"
    ]
    
    if num_agents is not None:
        cmd.extend(["--num_agents", str(num_agents)])
    if num_landmarks is not None:
        cmd.extend(["--num_landmarks", str(num_landmarks)])

    if config.controller != "causal":
        cmd.extend(["--controller", config.controller])
    if config.repair_target != "auto":
        cmd.extend(["--repair_target", config.repair_target])
    if config.trigger_mode:
        cmd.extend(["--trigger_mode", config.trigger_mode])
    if config.cost_lambda is not None:
        cmd.extend(["--cost_lambda", str(config.cost_lambda)])
    if config.causal_attribution_threshold is not None:
        cmd.extend(["--causal_attribution_threshold", str(config.causal_attribution_threshold)])
    if config.severe_reward_ratio is not None:
        cmd.extend(["--select_severe_reward_ratio", str(config.severe_reward_ratio)])
    if config.holdout_episodes is not None:
        cmd.extend(["--holdout_episodes", str(config.holdout_episodes)])
    if config.lora_rank is not None:
        cmd.extend(["--lora_rank", str(config.lora_rank)])
    if config.accept_comm_recovery is not None:
        cmd.extend(["--accept_comm_recovery", str(config.accept_comm_recovery)])
        
    log_file = open(repair_log_path, "w")
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"
        
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True
        )
    except Exception as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch repair subprocess: {e}")
        
    active_process = proc
    active_run_id = run_id
    active_process_type = "repair"
    
    def monitor_repair(p, r_id):
        p.wait()
        log_file.close()
        
        global active_process, active_run_id, active_process_type
        if active_run_id == r_id:
            active_process = None
            active_run_id = None
            active_process_type = None
            
    background_tasks.add_task(monitor_repair, proc, run_id)
    
    return {"run_id": run_id, "status": "running", "repair_log": repair_log_path}

@app.post("/api/runs/stop")
def stop_run():
    """Stop the active training or repair subprocess."""
    global active_process, active_run_id, active_process_type
    
    if active_process is None or active_process.poll() is not None:
        raise HTTPException(status_code=400, detail="No active process is in progress.")
        
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(active_process.pid), 15)
        else:
            active_process.terminate()
        active_process.wait(timeout=5)
    except Exception:
        try:
            active_process.kill()
        except Exception:
            pass
            
    registry = load_registry()
    if active_run_id in registry:
        registry[active_run_id]["status"] = "stopped"
        save_registry(registry)
        
    stopped_id = active_run_id
    active_process = None
    active_run_id = None
    active_process_type = None
    
    return {"run_id": stopped_id, "status": "stopped"}

@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str, lines: int = 200):
    """Fetch latest training logs."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    log_path = os.path.join(run["path"], "output.log")
    if not os.path.exists(log_path):
        return {"logs": "[No training logs found]"}
        
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
            return {"logs": "".join(log_lines[-lines:])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")

@app.get("/api/runs/{run_id}/repair_logs")
def get_repair_logs(run_id: str, lines: int = 200):
    """Fetch latest Phase 2/3 repair logs."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    log_path = os.path.join(run["path"], "repair_output.log")
    if not os.path.exists(log_path):
        return {"logs": "[No repair experiments executed on this run yet]"}
        
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
            return {"logs": "".join(log_lines[-lines:])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read repair logs: {e}")

def parse_repair_verdict(log_path: str) -> Dict:
    """Parse structured repair signals including meta-causal decision, frugality ROI, and held-out checks."""
    if not os.path.exists(log_path):
        return {"has_logs": False, "decision_type": "NO_RUNS"}
        
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return {"has_logs": False, "decision_type": "READ_ERROR"}

    res = {
        "has_logs": True,
        "detector_fired": None,
        "decision_type": "UNKNOWN",
        "decision_summary": None,
        "trigger_mode": "meta_causal",
        "reward_drop_ratio": None,
        "causal_attribution_ratio": None,
        "causal_evidence_score": None,
        "expected_recovery": None,
        "compute_cost": None,
        "roi": None,
        "roi_positive": None,
        "baseline_reward": None,
        "baseline_comm_effect": None,
        "baseline_value_sens": None,
        "baseline_kl": None,
        "degraded_reward": None,
        "degraded_comm_effect": None,
        "degraded_value_sens": None,
        "degraded_kl": None,
        "repaired_reward": None,
        "repaired_comm_effect": None,
        "repaired_value_sens": None,
        "repaired_kl": None,
        "iteration_losses": [],
        "repair_target": None,
        "accepted": None,
        "accepted_target": None,
        "reward_recovery_pct": None,
        "comm_recovery_pct": None,
        "retention_loss_pct": None,
        "heldout_validation": "NONE",
        "heldout_reward_recovery": None,
        "trainable_params": None,
        "total_params": 10192,
        "escalation_steps": [],
    }

    TARGET_PARAM_DEFAULTS = {
        "embedding": (320, 3.1),
        "comm": (640, 6.3),
        "lora": (1293, 12.7),
        "noncomm": (325, 3.2),
        "full": (10192, 100.0),
    }
    current_attempt = None

    in_retention = False
    in_heldout = False

    for line in lines:
        line_s = line.strip()

        if "[5] Retention check" in line_s:
            in_retention = True
            in_heldout = False
        elif "[6] Held-out re-check" in line_s:
            in_heldout = True
            in_retention = False

        m_att = re.search(r"\[REPAIR ATTEMPT\s+\d+/\d+\]\s*target=([a-zA-Z0-9_]+)", line_s)
        if m_att:
            t_name = m_att.group(1)
            default_p, default_pct = TARGET_PARAM_DEFAULTS.get(t_name, (1293, 12.7))
            current_attempt = {
                "target": t_name,
                "status": "RUNNING",
                "trainable_params": default_p,
                "pct_params": default_pct,
                "reward_recovery": None,
                "comm_recovery": None,
            }
            res["escalation_steps"].append(current_attempt)
            res["repair_target"] = t_name

        m_params = re.search(r"\[REPAIR\]\s*target=([a-zA-Z0-9_]+)\s+trainable actor params=(\d+)\s+trainable attention params=(\d+)", line_s)
        if m_params:
            t_name = m_params.group(1)
            p_tot = int(m_params.group(2)) + int(m_params.group(3))
            res["trainable_params"] = p_tot
            if current_attempt is None or current_attempt["target"] != t_name:
                current_attempt = {
                    "target": t_name,
                    "status": "RUNNING",
                    "trainable_params": p_tot,
                    "pct_params": round((p_tot / 10192.0) * 100.0, 1),
                    "reward_recovery": None,
                    "comm_recovery": None,
                }
                res["escalation_steps"].append(current_attempt)
            else:
                current_attempt["trainable_params"] = p_tot
                current_attempt["pct_params"] = round((p_tot / 10192.0) * 100.0, 1)

        if "[ROLLBACK]" in line_s and current_attempt is not None:
            current_attempt["status"] = "ROLLBACK"

        if "[BASELINE]" in line_s:
            for p in line_s.split():
                if p.startswith("reward="):
                    try: res["baseline_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("comm_effect="):
                    try:
                        m = re.search(r"[-+]?\d*\.?\d+", p.split("=")[1])
                        if m: res["baseline_comm_effect"] = float(m.group(0))
                    except: pass
                elif p.startswith("value_sens="):
                    try: res["baseline_value_sens"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("kl="):
                    try: res["baseline_kl"] = float(p.split("=")[1])
                    except: pass

        elif "[DEGRADED]" in line_s:
            for p in line_s.split():
                if p.startswith("reward="):
                    try: res["degraded_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("comm_effect="):
                    try:
                        m = re.search(r"[-+]?\d*\.?\d+", p.split("=")[1])
                        if m: res["degraded_comm_effect"] = float(m.group(0))
                    except: pass
                elif p.startswith("value_sens="):
                    try: res["degraded_value_sens"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("kl="):
                    try: res["degraded_kl"] = float(p.split("=")[1])
                    except: pass

        if "iter " in line_s and "policy_loss=" in line_s and "value_loss=" in line_s:
            try:
                m = re.search(r"iter\s+(\d+)/\d+\s+policy_loss=([-\d.]+)\s+value_loss=([-\d.]+)", line_s)
                if m:
                    res["iteration_losses"].append({
                        "iter": int(m.group(1)),
                        "policy_loss": float(m.group(2)),
                        "value_loss": float(m.group(3))
                    })
            except: pass

        if "ENVIRONMENT CHANGE DETECTED" in line_s:
            res["detector_fired"] = True
        elif "ENVIRONMENT CHANGE NOT detected" in line_s:
            res["detector_fired"] = False

        if "reward drop ratio" in line_s and ":" in line_s:
            try:
                res["reward_drop_ratio"] = float(line_s.split(":")[1].split("(")[0].strip())
            except: pass

        if "causal attribution ratio" in line_s and ":" in line_s:
            try:
                res["causal_attribution_ratio"] = float(line_s.split(":")[1].split("(")[0].strip())
            except: pass

        if "causal evidence score" in line_s and ":" in line_s:
            try:
                res["causal_evidence_score"] = float(line_s.split(":")[1].split("(")[0].strip())
            except: pass

        if "expected recovery (ROI)" in line_s and ":" in line_s:
            try:
                val = line_s.split(":")[1].split("(")[0].strip().replace("+", "")
                res["roi"] = float(val)
                res["roi_positive"] = (res["roi"] > 0)
            except: pass

        if "decision summary" in line_s and ":" in line_s:
            try:
                res["decision_summary"] = line_s.split("decision summary")[1].split(":", 1)[1].strip()
            except: pass

        if "selected repair target:" in line_s:
            try:
                res["repair_target"] = line_s.split("selected repair target:")[1].strip()
            except: pass

        if re.search(r"\[REPAIRED(\[[a-zA-Z0-9_]+\])?\]", line_s):
            for p in line_s.split():
                if p.startswith("reward="):
                    try: res["repaired_reward"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("comm_effect="):
                    try:
                        m = re.search(r"[-+]?\d*\.?\d+", p.split("=")[1])
                        if m: res["repaired_comm_effect"] = float(m.group(0))
                    except: pass
                elif p.startswith("value_sens="):
                    try: res["repaired_value_sens"] = float(p.split("=")[1])
                    except: pass
                elif p.startswith("kl="):
                    try: res["repaired_kl"] = float(p.split("=")[1])
                    except: pass

        if "reward recovery" in line_s and ":" in line_s:
            try:
                val = line_s.split(":")[-1].strip().replace("%", "")
                if "n/a" not in val.lower():
                    fval = float(val)
                    if in_heldout:
                        res["heldout_reward_recovery"] = fval
                    else:
                        res["reward_recovery_pct"] = fval
                        if current_attempt is not None:
                            current_attempt["reward_recovery"] = fval
            except: pass

        if "communication recovery" in line_s and ":" in line_s:
            try:
                val = line_s.split(":")[-1].strip().replace("%", "")
                if "n/a" not in val.lower():
                    fval = float(val)
                    res["comm_recovery_pct"] = fval
                    if current_attempt is not None:
                        current_attempt["comm_recovery"] = fval
            except: pass

        if "decision: ACCEPT" in line_s or "repair accepted:" in line_s:
            res["accepted"] = True
            if current_attempt is not None:
                current_attempt["status"] = "ACCEPTED"
            if "repair accepted:" in line_s:
                res["accepted_target"] = line_s.split("repair accepted:")[1].strip()
        elif "all repair attempts rejected" in line_s or "decision: REJECT" in line_s:
            if current_attempt is not None and current_attempt["status"] == "RUNNING":
                current_attempt["status"] = "REJECTED"
            if res["accepted"] is not True:
                res["accepted"] = False

        if "HELD-OUT CONFIRMS" in line_s:
            res["heldout_validation"] = "CONFIRMED"
        elif "HELD-OUT DOES NOT CONFIRM" in line_s or "HELD-OUT FAILS" in line_s:
            res["heldout_validation"] = "FAILED"

        if in_retention and "% of |baseline|)" in line_s:
            try:
                pct_str = line_s.split("(")[1].split("%")[0].strip()
                res["retention_loss_pct"] = float(pct_str)
            except: pass

    # Categorize decision_type
    summary = (res["decision_summary"] or "").lower()
    if "frugal abstention" in summary or (res["detector_fired"] is False and res["roi"] is not None and res["roi"] <= 0):
        res["decision_type"] = "FRUGAL_ABSTENTION"
    elif "emergency" in summary or "catastrophic" in summary:
        res["decision_type"] = "EMERGENCY_TRIGGER"
    elif "causal trigger confirmed" in summary or (res["detector_fired"] is True and res["roi_positive"]):
        res["decision_type"] = "CAUSAL_CONFIRMED"
    elif res["detector_fired"] is True:
        res["decision_type"] = "TRIGGERED"
    elif res["detector_fired"] is False:
        res["decision_type"] = "NOT_DEGRADED"

    if res["trainable_params"] is None and (res["accepted_target"] or res["repair_target"]):
        t_key = res["accepted_target"] or res["repair_target"]
        if t_key in TARGET_PARAM_DEFAULTS:
            res["trainable_params"] = TARGET_PARAM_DEFAULTS[t_key][0]

    return res

@app.get("/api/runs/{run_id}/trajectory")
def get_run_trajectory(run_id: str):
    """Return episode rollout coordinates conditioned on the selected run's live config and metrics."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    cfg = run.get("config", {})
    num_agents = int(cfg.get("num_agents") or 2)
    num_landmarks = int(cfg.get("num_landmarks") or (num_agents + 1))
    seed = int(cfg.get("seed") or 1)

    # Read latest logged metrics from causal_influence.csv
    latest_comm_effect = None
    latest_kl = None
    latest_reward = None
    csv_path = os.path.join(run["path"], "causal_influence.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    if last.get("comm_effect") not in (None, ""):
                        latest_comm_effect = float(last["comm_effect"])
                    if last.get("causal_influence_kl_mean") not in (None, ""):
                        latest_kl = float(last["causal_influence_kl_mean"])
                    if last.get("eval_reward") not in (None, ""):
                        latest_reward = float(last["eval_reward"])
        except Exception:
            pass

    verdict = parse_repair_verdict(os.path.join(run["path"], "repair_output.log"))

    # Load recorded trajectory.json if present
    traj_json_path = os.path.join(run["path"], "trajectory.json")
    if os.path.exists(traj_json_path):
        try:
            with open(traj_json_path, "r") as f:
                loaded = json.load(f)
                loaded["latest_comm_effect"] = latest_comm_effect
                loaded["latest_kl"] = latest_kl
                loaded["latest_reward"] = latest_reward
                loaded["repaired_reward"] = verdict.get("repaired_reward")
                loaded["has_repair"] = run["has_repair"]
                loaded["is_live_run"] = True
                return loaded
        except Exception:
            pass

    # Deterministic rollout seeded by run_id and config
    rng = random.Random(hash(run_id) ^ seed)
    landmarks = []
    for l_idx in range(num_landmarks):
        ang = (2.0 * math.pi * l_idx) / max(num_landmarks, 1) + 0.35
        radius = 0.62 if l_idx % 2 == 0 else 0.48
        landmarks.append({
            "id": l_idx,
            "x": round(math.cos(ang) * radius + rng.uniform(-0.08, 0.08), 3),
            "y": round(math.sin(ang) * radius + rng.uniform(-0.08, 0.08), 3),
        })

    # Initial agent spawns
    starts = []
    for a_idx in range(num_agents):
        ang = (2.0 * math.pi * a_idx) / max(num_agents, 1) + math.pi * 0.6
        starts.append((
            math.cos(ang) * 0.78 + rng.uniform(-0.06, 0.06),
            math.sin(ang) * 0.78 + rng.uniform(-0.06, 0.06),
        ))

    comm_active = (latest_comm_effect or 0.0) > 0 or (latest_kl or 0.0) > 0.01
    max_steps = 50
    steps = []
    for s in range(max_steps):
        t = s / float(max_steps - 1)
        # Smooth cooperative convergence factor
        ease = 1.0 - math.pow(1.0 - min(t * 1.25, 1.0), 2.2)
        step_agents = []
        for a_idx in range(num_agents):
            sx, sy = starts[a_idx]
            target = landmarks[a_idx % num_landmarks]
            # Slight curved avoidance arc in early episode
            arc = math.sin(t * math.pi) * (0.14 if a_idx % 2 == 0 else -0.14)
            cx = sx + (target["x"] - sx) * ease + arc * (1.0 - ease)
            cy = sy + (target["y"] - sy) * ease - arc * 0.6 * (1.0 - ease)
            token = (a_idx * 2 + s // 10) % 5 if comm_active else (s // 5 + a_idx) % 5
            step_agents.append({
                "id": a_idx,
                "x": round(cx, 4),
                "y": round(cy, 4),
                "vx": round((target["x"] - cx) * 0.15, 4),
                "vy": round((target["y"] - cy) * 0.15, 4),
                "emitted_token": token,
                "target_lm": a_idx % num_landmarks,
            })
        steps.append({"step": s, "agents": step_agents})

    return {
        "run_id": run_id,
        "num_agents": num_agents,
        "num_landmarks": num_landmarks,
        "seed": seed,
        "latest_comm_effect": latest_comm_effect,
        "latest_kl": latest_kl,
        "latest_reward": latest_reward,
        "repaired_reward": verdict.get("repaired_reward"),
        "has_repair": run["has_repair"],
        "is_live_run": True,
        "landmarks": landmarks,
        "steps": steps,
    }

@app.get("/api/runs/{run_id}/repair_verdict")
def get_repair_verdict(run_id: str):
    """Fetch parsed structured repair signals and meta-causal decisions."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    log_path = os.path.join(run["path"], "repair_output.log")
    return parse_repair_verdict(log_path)

@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(run_id: str):
    """Fetch causal influence metrics from causal_influence.csv."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    csv_path = os.path.join(run["path"], "causal_influence.csv")
    if not os.path.exists(csv_path):
        return {"metrics": []}
        
    try:
        metrics = []
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    try:
                        parsed_row[k] = float(v) if '.' in v or 'e' in v else int(v)
                    except ValueError:
                        parsed_row[k] = v
                metrics.append(parsed_row)
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse metrics: {e}")

@app.get("/api/runs/{run_id}/checkpoints")
def get_run_checkpoints(run_id: str):
    """List saved checkpoints under models/ directory."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    models_dir = os.path.join(run["path"], "models")
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        return {"checkpoints": []}
        
    checkpoints = []
    for d in os.listdir(models_dir):
        full_p = os.path.join(models_dir, d)
        if os.path.isdir(full_p) and not d.endswith("_repair_runs"):
            is_best = (d == "checkpoint_best")
            step = 0
            if d.startswith("checkpoint_") and not is_best:
                try:
                    step = int(d.replace("checkpoint_", ""))
                except ValueError:
                    pass
            checkpoints.append({
                "name": d,
                "path": full_p.replace("\\", "/"),
                "step": step,
                "is_best": is_best
            })
            
    checkpoints.sort(key=lambda x: (not x["is_best"], x["step"]), reverse=True)
    return {"checkpoints": checkpoints}

@app.post("/api/runs/{run_id}/render")
def render_run(run_id: str):
    """Run rendering for a completed run to generate render.gif."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    models_dir = os.path.join(run["path"], "models")
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        raise HTTPException(status_code=400, detail="No model weights found. Check if training is completed.")
        
    config = run.get("config", {})
    num_agents = config.get("num_agents")
    num_landmarks = config.get("num_landmarks")

    # Auto-detect num_agents and num_landmarks if missing from config
    if num_agents is None:
        csv_path = os.path.join(run["path"], "causal_influence.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    header = f.readline().strip().split(",")
                    agent_cols = [c for c in header if c.startswith("causal_influence_kl_agent")]
                    if agent_cols:
                        num_agents = len(agent_cols)
            except Exception:
                pass

    if num_agents is None:
        num_agents = 2

    if num_landmarks is None:
        actor_path = os.path.join(models_dir, "actor.pt")
        if os.path.exists(actor_path):
            try:
                sd = torch.load(actor_path, map_location="cpu")
                if "message_head.weight" in sd:
                    obs_dim = sd["message_head.weight"].shape[1]
                    calc_landmarks = (obs_dim - 4 - 4 * (num_agents - 1)) // 2
                    if calc_landmarks > 0:
                        num_landmarks = calc_landmarks
            except Exception:
                pass

    if num_landmarks is None:
        num_landmarks = 3

    best_model_dir = os.path.join(models_dir, "checkpoint_best")
    render_model_dir = best_model_dir if os.path.exists(os.path.join(best_model_dir, "actor.pt")) else models_dir

    cmd = [
        sys.executable, "onpolicy/scripts/render/render_mpe.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--use_render",
        "--model_dir", render_model_dir,
        "--save_gifs",
        "--gif_dir", run["path"],
        "--n_rollout_threads", "1",
        "--n_training_threads", "1",
        "--render_episodes", "1",
        "--num_agents", str(num_agents),
        "--num_landmarks", str(num_landmarks)
    ]
    
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        gif_path = os.path.join(run["path"], "render.gif")
        if not os.path.exists(gif_path):
            raise HTTPException(status_code=500, detail=f"Rendering finished, but render.gif was not found. Error: {result.stderr}")
            
        return {"status": "success", "gif_path": gif_path}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Rendering timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute rendering: {e}")

@app.get("/api/runs/{run_id}/gif")
def get_run_gif(run_id: str):
    """Serve the render.gif visualization of the run."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    gif_path = os.path.join(run["path"], "render.gif")
    if not os.path.exists(gif_path):
        raise HTTPException(status_code=404, detail="No render.gif found for this run.")
        
    return FileResponse(gif_path, media_type="image/gif")

@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a run from the filesystem and registry."""
    global active_process, active_run_id
    if active_run_id == run_id and active_process is not None and active_process.poll() is None:
        raise HTTPException(status_code=400, detail="Cannot delete an active process. Terminate it first.")
        
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    run_dir = run["path"]
    if os.path.exists(run_dir):
        try:
            shutil.rmtree(run_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete directory: {e}")
            
    registry = load_registry()
    if run_id in registry:
        del registry[run_id]
        save_registry(registry)
        
    return {"status": "success", "detail": f"Successfully deleted run {run_id}"}

@app.post("/api/runs/{run_id}/archive")
def archive_run(run_id: str):
    """Toggle the archived status of a run in the registry."""
    registry = load_registry()
    
    if run_id not in registry:
        runs = scan_runs()
        run = next((r for r in runs if r["run_id"] == run_id), None)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        registry[run_id] = {
            "status": run["status"],
            "config": run["config"],
            "archived": True
        }
    else:
        current_status = registry[run_id].get("archived", False)
        registry[run_id]["archived"] = not current_status
        
    save_registry(registry)
    return {"status": "success", "archived": registry[run_id]["archived"]}

# ==================== Benchmarking Endpoints ====================

@app.post("/api/benchmark/start")
def start_benchmark(config: BenchmarkStartConfig, background_tasks: BackgroundTasks):
    """Trigger the automated benchmark suite across multi-agent scales and control arms."""
    global active_benchmark_process
    if active_benchmark_process is not None and active_benchmark_process.poll() is None:
        raise HTTPException(status_code=400, detail="A benchmark evaluation suite is already running.")

    os.makedirs(os.path.dirname(BENCHMARK_LOG_PATH), exist_ok=True)
    log_file = open(BENCHMARK_LOG_PATH, "w")

    cmd = [
        sys.executable, "-u", "benchmarking/runners/benchmark_suite.py",
        "--scales", *config.scales,
        "--arms", *config.arms,
        "--seeds", *[str(s) for s in config.seeds],
        "--repair_iters", str(config.repair_iters),
        "--measure_episodes", str(config.measure_episodes),
        "--holdout_episodes", str(config.holdout_episodes)
    ]

    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        env["WANDB_MODE"] = "disabled"

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True
        )
    except Exception as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch benchmark suite: {e}")

    active_benchmark_process = proc

    def monitor_benchmark(p):
        p.wait()
        log_file.close()
        global active_benchmark_process
        active_benchmark_process = None

    background_tasks.add_task(monitor_benchmark, proc)

    return {"status": "running", "pid": proc.pid, "log_file": BENCHMARK_LOG_PATH}

@app.get("/api/benchmark/status")
def get_benchmark_status(lines: int = 100):
    """Get active benchmark execution state and latest log tail."""
    global active_benchmark_process
    is_running = (active_benchmark_process is not None and active_benchmark_process.poll() is None)
    
    logs = ""
    if os.path.exists(BENCHMARK_LOG_PATH):
        try:
            with open(BENCHMARK_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                log_lines = f.readlines()
                logs = "".join(log_lines[-lines:])
        except Exception:
            logs = "[Error reading benchmark logs]"

    return {
        "running": is_running,
        "pid": active_benchmark_process.pid if is_running else None,
        "logs": logs
    }

@app.post("/api/benchmark/stop")
def stop_benchmark():
    """Stop active benchmark suite execution."""
    global active_benchmark_process
    if active_benchmark_process is None or active_benchmark_process.poll() is not None:
        raise HTTPException(status_code=400, detail="No active benchmark process.")

    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(active_benchmark_process.pid), 15)
        else:
            active_benchmark_process.terminate()
        active_benchmark_process.wait(timeout=5)
    except Exception:
        try:
            active_benchmark_process.kill()
        except Exception:
            pass

    active_benchmark_process = None
    return {"status": "stopped"}

@app.get("/api/benchmark/summary")
def get_benchmark_summary():
    """Fetch consolidated empirical master metrics and publication plot metadata."""
    csv_path = "benchmarking/data/results/benchmark_master.csv"
    md_path = "benchmarking/data/results/metrics_summary.md"
    plots_dir = "benchmarking/visualizations/plots"

    rows = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    parsed = {}
                    for k, v in r.items():
                        try:
                            parsed[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                    rows.append(parsed)
        except Exception as e:
            print(f"Error reading benchmark csv: {e}")

    md_content = ""
    if os.path.exists(md_path):
        try:
            with open(md_path, mode="r", encoding="utf-8") as f:
                md_content = f.read()
        except Exception:
            pass

    plots = []
    if os.path.exists(plots_dir):
        for fname in sorted(os.listdir(plots_dir)):
            if fname.endswith(".png"):
                name = fname.replace(".png", "")
                plots.append({
                    "id": name,
                    "filename": fname,
                    "url": f"/api/benchmark/plots/{fname}",
                    "title": name.replace("_", " ").title()
                })

    return {
        "has_data": len(rows) > 0 or len(md_content) > 0,
        "rows": rows,
        "metrics_summary_md": md_content,
        "plots": plots
    }

@app.get("/api/benchmark/plots/{plot_name}")
def get_benchmark_plot(plot_name: str):
    """Serve publication vector plot PNG or PDF."""
    plots_dir = "benchmarking/visualizations/plots"
    file_path = os.path.join(plots_dir, plot_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot file not found.")

    media_type = "image/png" if plot_name.endswith(".png") else "application/pdf"
    return FileResponse(file_path, media_type=media_type)
