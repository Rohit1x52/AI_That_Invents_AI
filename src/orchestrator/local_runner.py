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
from src.evolution.dna import architecture_dna, diff_dna
from src.agents.critic import critique_blueprint


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
        name = bp.get("name", f"arch_{run_id}")
        parent_id = bp.get("_parent_arch_id")

        # ===============================
        # 1. Register Architecture
        # ===============================
        arch_id = dkb.add_architecture(
            name=name,
            blueprint=bp,
            features={
                "est_params": bp.get("est_params"),
                "est_flops": bp.get("est_flops"),
            }
        )
        result_summary["arch_id"] = arch_id

        # ===============================
        # 2. DNA (BEFORE training)
        # ===============================
        dna = architecture_dna(bp)
        dkb.add_dna(arch_id, dna)

        # ===============================
        # 3. Start Trial
        # ===============================
        cfg = {
            "blueprint": bp,
            "data": {"root": "./data"},
            "train": train_cfg,
            "device": worker_device,
            "log_dir": str(log_dir),
            "seed": 42,
        }

        trial_id = dkb.add_trial(
            arch_id=arch_id,
            run_id=run_id,
            cfg=cfg,
            start_ts=time.time()
        )

        logger.info(f"Training Arch {arch_id} on {worker_device}")

        # ===============================
        # 4. Train
        # ===============================
        res = train_one_model(cfg)

        best_acc = float(res.get("best_val_acc", 0.0))
        params = res.get("params")
        latency_cpu = (res.get("latency_ms") or {}).get("cpu")

        dkb.update_trial_result(
            trial_id,
            status="COMPLETED",
            best_acc=best_acc,
            ckpt_path=res.get("best_checkpoint")
        )

        dkb.add_metrics(
            trial_id,
            epoch=-1,
            metrics={
                "val_acc": best_acc,
                "latency_cpu_ms": latency_cpu,
            }
        )

        # ===============================
        # 5. CRITIC (AFTER training)
        # ===============================
        critic = critique_blueprint(
            bp=bp,
            dna=dna,
            metrics={
                "val_acc": best_acc,
                "params": params,
                "latency_cpu_ms": latency_cpu,
            },
            mode="heuristic"  # or "llm"
        )

        dkb.add_critic_score(arch_id, critic)

        # ===============================
        # 6. Genealogy
        # ===============================
        if parent_id:
            parent = dkb.get_architecture(parent_id)
            if parent and parent.get("dna_json"):
                delta = diff_dna(parent["dna_json"], dna)
                mutation = bp.get("_mutation", {})
                dkb.add_genealogy(
                    child_arch_id=arch_id,
                    parent_arch_id=parent_id,
                    mutation_type=mutation.get("type", "unknown"),
                    mutation_reason=mutation.get("reason", "unknown"),
                    delta_dna=delta
                )

        result_summary.update({
            "status": "COMPLETED",
            "val_acc": best_acc
        })

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        result_summary["error"] = str(e)

    finally:
        dkb.close()
        if device_queue:
            device_queue.put(worker_device)

    return result_summary
