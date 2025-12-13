import uuid
import time
import json
import logging
import multiprocessing
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.dkb.client_sqlite import DKBClient

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("orchestrator")

# Dynamic Import of Trainer
try:
    from src.trainer.train import train_one_model
except ImportError:
    # Mock for testing if trainer is missing
    def train_one_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(1) 
        return {
            "best_val_acc": 0.5, 
            "best_checkpoint": "mock.pth", 
            "params": 1000, 
            "flops": 1e6,
            "latency_ms": {"cpu": 1.0, "cuda": 0.5}
        }

def _worker_process(
    bp: Dict[str, Any], 
    dkb_path: str, 
    train_cfg: Dict, 
    base_log_dir: str, 
    device_queue: Optional[multiprocessing.Queue]
) -> Dict[str, Any]:
    """
    Isolated worker function to handle a single blueprint lifecycle.
    """
    worker_device = "cpu"
    if device_queue:
        try:
            worker_device = device_queue.get(timeout=5)
        except Exception:
            pass

    run_id = str(uuid.uuid4())[:8]
    log_dir = Path(base_log_dir) / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    dkb = DKBClient(dkb_path)
    result_summary = {"status": "FAILED", "error": None, "arch_id": None}

    try:
        # 1. Register Architecture
        name = bp.get("name") or f"arch_{run_id}"
        arch_id = dkb.add_architecture(name, bp)
        result_summary["arch_id"] = arch_id

        # 2. Start Trial
        cfg = {
            "blueprint": bp,
            "data": {"root": "./data"},
            "train": train_cfg,
            "device": worker_device,
            "log_dir": str(log_dir),
            "seed": 42
        }
        
        trial_id = dkb.add_trial(arch_id, run_id, cfg, start_ts=time.time())
        logger.info(f"Started Trial {trial_id} (Arch {arch_id}) on {worker_device}")

        # 3. Train
        try:
            train_result = train_one_model(cfg)
            status = "COMPLETED"
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
            traceback.print_exc()
            dkb.update_trial_result(trial_id, status="FAILED")
            result_summary["error"] = str(e)
            return result_summary

        # 4. Update DB
        best_acc = float(train_result.get("best_val_acc", 0.0) or 0.0)
        ckpt_path = train_result.get("best_checkpoint")
        
        dkb.update_trial_result(trial_id, status="COMPLETED", best_acc=best_acc, ckpt_path=ckpt_path)
        
        # Log metrics
        lat = train_result.get("latency_ms") or {}
        dkb.add_metrics(
            trial_id, 
            epoch=-1, # Final summary
            metrics={
                "val_acc": best_acc,
                "val_loss": float(train_result.get("best_val_loss", 0.0) or 0.0),
                "latency_cpu_ms": lat.get("cpu"),
                "latency_cuda_ms": lat.get("cuda")
            }
        )

        result_summary["status"] = "COMPLETED"
        result_summary["val_acc"] = best_acc

    except Exception as e:
        logger.error(f"Worker process crash: {e}")
        result_summary["error"] = str(e)
    finally:
        dkb.close()
        if device_queue:
            device_queue.put(worker_device) # Return GPU to pool

    return result_summary

def run_campaign(
    blueprints: List[Dict[str, Any]],
    dkb_path: str = "dkb.sqlite",
    low_fidelity_cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    max_workers: int = 1,
    log_dir: str = "./logs/campaign"
) -> Dict[str, Any]:

    # Setup Config
    default_train = {
        "epochs": 1, 
        "batch_size": 64, 
        "use_synthetic": False,
        "patience": 3
    }
    train_cfg = {**default_train, **(low_fidelity_cfg or {})}
    
    # Setup Device Pool
    manager = multiprocessing.Manager()
    device_queue = manager.Queue()
    
    if "cuda" in device:
        # If user passes "cuda", we check count. If "cuda:0", just one.
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                device_queue.put(f"cuda:{i}")
        else:
            device_queue.put("cpu")
    else:
        # Put 'cpu' for as many workers as we have
        for _ in range(max_workers):
            device_queue.put("cpu")

    logger.info(f"Starting Campaign with {len(blueprints)} candidates using {max_workers} workers.")

    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _worker_process, 
                bp, dkb_path, train_cfg, log_dir, device_queue
            ) 
            for bp in blueprints
        ]

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            logger.info(f"Candidate finished: {res['status']} (Acc: {res.get('val_acc', 'N/A')})")

    return {"results": results, "total": len(blueprints)}