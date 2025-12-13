import json
import time
import uuid
import subprocess
import tempfile
import os
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.dkb.client_sqlite import DKBClient

# --- Helper: Worker Function ---

def _worker_train_task(
    bp_path: str, 
    cfg_json: str, 
    result_path: str,
    device_id: str,
    timeout: int
) -> Tuple[int, str, str]:
    """
    Runs the training subprocess with a specific device assignment.
    """
    # Force the subprocess to use the assigned GPU
    env = os.environ.copy()
    if "cuda" in device_id:
        # Extract ID (e.g. "cuda:0" -> "0")
        gpu_idx = device_id.split(":")[-1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    
    # We update the config to use the generic 'cuda' since visible devices is set
    # or keep 'cpu' if that was passed.
    
    cmd = [
        "python", "-m", "src.trainer.train_worker", 
        "--cfg", cfg_json,
        "--out", result_path 
    ]
    
    try:
        proc = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            env=env
        )
        return (proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as te:
        return (124, "", f"TIMEOUT: {te}")
    except Exception as e:
        return (1, "", str(e))

# --- Main Runner ---

def run_campaign_parallel(
    blueprints: List[Dict[str, Any]],
    dkb_path: str = "dkb.sqlite",
    base_device: str = "cuda", # "cpu" or "cuda"
    max_workers: int = 2,
    low_fidelity_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    retries: int = 1,
    per_task_timeout: int = 300,
    post_metrics_runs: int = 10
) -> Dict[str, Any]:
    
    dkb = DKBClient(dkb_path)
    summary = {"archs": []}
    
    # 1. Setup Device Pool
    manager = multiprocessing.Manager()
    device_queue = manager.Queue()
    
    import torch
    if "cuda" in base_device and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # If max_workers > gpu_count, we overlap (2 jobs per GPU)
        # or we limit max_workers to gpu_count. Let's cycle devices.
        for i in range(max_workers):
            gpu_id = i % gpu_count
            device_queue.put(f"cuda:{gpu_id}")
    else:
        for _ in range(max_workers):
            device_queue.put("cpu")

    # 2. Prepare Tasks
    tasks = []
    run_group_id = str(uuid.uuid4())[:8]
    
    print(f"🚀 Starting Campaign {run_group_id} | Workers: {max_workers} | Blueprints: {len(blueprints)}")

    for bp in blueprints:
        name = bp.get("name") or f"arch_{int(time.time())}"
        arch_id = dkb.add_architecture(name, bp)
        
        run_id = str(uuid.uuid4())[:8]
        train_cfg = dict(low_fidelity_cfg or {
            "use_synthetic": True, "epochs": 1, "batch_size": 64, "lr": 0.01, "patience": 1
        })
        
        # Log dir specific to this run
        log_dir = Path(f"./logs/campaign_{run_group_id}/{run_id}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        cfg = {
            "blueprint": bp, 
            "data": {"root": "./data"}, 
            "train": train_cfg, 
            "device": "cuda" if "cuda" in base_device else "cpu", # Logic handled by env var
            "log_dir": str(log_dir), 
            "seed": seed
        }
        
        tasks.append({
            "arch_id": arch_id, 
            "run_id": run_id, 
            "bp": bp, 
            "cfg": cfg,
            "log_dir": log_dir
        })

    # 3. Execute Parallel
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        
        # Submit all tasks
        for t in tasks:
            device_id = device_queue.get() # Block until a device is available? 
            # Actually, ProcessPool handles the blocking. We just need to assign round-robin 
            # if we submit all at once. But device_queue needs to be managed inside the future?
            # Better approach for ProcessPool: Pass the queue to the worker, have worker pop/push.
            # But we are using subprocess. Let's just pre-assign round robin here for simplicity.
            # (Strict resource locking requires a smarter scheduler, but round-robin works for simple cases).
            
            # Re-put device for next task (round robin)
            device_queue.put(device_id) 

            # Create temp files
            bp_path = t["log_dir"] / "blueprint.json"
            bp_path.write_text(json.dumps(t["bp"], indent=2))
            
            result_path = t["log_dir"] / "result.json"
            cfg_json = json.dumps(t["cfg"])
            
            fut = exe.submit(
                _worker_train_task, 
                str(bp_path), 
                cfg_json, 
                str(result_path),
                device_id,
                per_task_timeout
            )
            futures[fut] = (t, bp_path, result_path, device_id, retries)

        # 4. Monitor Progress
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Training Models"):
            t, bp_path, result_path, used_device, task_retries = futures[fut]
            
            try:
                rc, out, err = fut.result()
            except Exception as e:
                rc, out, err = 1, "", str(e)

            # Retry Logic
            if rc != 0 and task_retries > 0:
                print(f"⚠️  Task {t['run_id']} failed on {used_device}. Retrying...")
                # Simple synchronous retry (blocking the pool slightly, but safe)
                rc, out, err = _worker_train_task(
                    str(bp_path), json.dumps(t["cfg"]), str(result_path), used_device, per_task_timeout
                )

            # Parse Results
            res_data = {}
            if rc == 0 and result_path.exists():
                try:
                    res_data = json.loads(result_path.read_text())
                except:
                    pass
            
            # Record to DB
            _record_result_to_dkb(dkb, t, res_data, rc, err, post_metrics_runs)
            
            summary["archs"].append({
                "arch_id": t["arch_id"], 
                "status": "COMPLETED" if rc == 0 else "FAILED",
                "val_acc": res_data.get("best_val_acc")
            })

    dkb.close()
    return summary

def _record_result_to_dkb(dkb, task, res, rc, err, post_runs):
    """Helper to write results to DB and run post-metrics."""
    arch_id = task["arch_id"]
    run_id = task["run_id"]
    
    start_ts = time.time()
    end_ts = time.time() # Approximation
    
    # 1. Add Trial
    try:
        ckpt = res.get("best_checkpoint")
        trial_id = dkb.add_trial(
            arch_id, run_id, task["cfg"], 
            checkpoint_path=ckpt, start_ts=start_ts, end_ts=end_ts
        )
        status = "COMPLETED" if rc == 0 else "FAILED"
        dkb.update_trial_result(trial_id, status=status, best_acc=res.get("best_val_acc"))
    except Exception as e:
        print(f"DB Error: {e}")
        return

    # 2. Add Metrics
    if rc == 0:
        try:
            lat = res.get("latency_ms", {})
            dkb.add_metrics(
                trial_id, -1, 
                val_acc=float(res.get("best_val_acc", 0)), 
                val_loss=float(res.get("best_val_loss", 0)),
                latency_cpu_ms=lat.get("cpu"),
                latency_cuda_ms=lat.get("cuda")
            )
        except: pass