import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.trainer.train import train_one_model
from src.dkb.client_sqlite import DKBClient

def run_campaign(blueprints: List[Dict[str, Any]],
                 dkb_path: str = "dkb.sqlite",
                 low_fidelity_cfg: Optional[Dict[str, Any]] = None,
                 device: str = "cpu",
                 seed: int = 42) -> Dict[str, Any]:
    dkb = DKBClient(dkb_path)
    summary = {"archs": []}
    for bp in blueprints:
        name = bp.get("name", f"arch_{int(time.time())}")
        # optional: compute quick features (params/flops) - trainer does better
        arch_id = dkb.add_architecture(name, bp, features={})
        run_id = str(uuid.uuid4())[:8]
        cfg = {
            "blueprint": bp,
            "data": {"root": "./data"},
            "train": dict(low_fidelity_cfg or {"use_synthetic": True, "epochs": 1, "batch_size": 64, "lr": 0.01, "patience": 1}),
            "device": device,
            "log_dir": f"./logs/campaign_{run_id}",
            "seed": seed
        }
        start_ts = time.time()
        try:
            res = train_one_model(cfg)
        except Exception as e:
            # on failure insert trial with null metrics and continue
            end_ts = time.time()
            trial_id = dkb.add_trial(arch_id, run_id, cfg, checkpoint_path=None, start_ts=start_ts, end_ts=end_ts)
            summary["archs"].append({"arch_id": arch_id, "trial_id": trial_id, "error": str(e)})
            continue

        end_ts = time.time()
        # add trial row
        trial_id = dkb.add_trial(arch_id, run_id, cfg, checkpoint_path=res.get("best_checkpoint"), start_ts=start_ts, end_ts=end_ts)
        # add a final metric row (epoch=-1 for summary)
        lat = res.get("latency_ms", {})
        cpu_lat = lat.get("cpu") if isinstance(lat, dict) else None
        cuda_lat = lat.get("cuda") if isinstance(lat, dict) else None
        dkb.add_metrics(trial_id, epoch=-1, val_acc=res.get("best_val_acc", 0.0), val_loss=0.0, latency_cpu_ms=cpu_lat, latency_cuda_ms=cuda_lat)
        summary["archs"].append({"arch_id": arch_id, "trial_id": trial_id, "result": res})
    dkb.close()
    return summary
