import uuid
import time
import json
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.dkb.client_sqlite import DKBClient

try:
    from src.trainer.train import train_one_model
except Exception:
    try:
        from src.trainer.train import train as _train
        def train_one_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
            _train(cfg)  
            return {"best_val_acc": 0.0, "best_checkpoint": None, "params": None, "latency_ms": None, "flops": None, "log_dir": cfg.get("log_dir"), "seed": cfg.get("seed")}
    except Exception:
        raise RuntimeError("train_one_model not available; ensure src.trainer.train.train_one_model or train_one_model exists")

def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return "<unserializable>"


def run_campaign(blueprints: List[Dict[str, Any]],
                 dkb_path: str = "dkb.sqlite",
                 low_fidelity_cfg: Optional[Dict[str, Any]] = None,
                 device: str = "cpu",
                 seed: int = 42,
                 post_metrics_runs: int = 30) -> Dict[str, Any]:
    dkb = DKBClient(dkb_path)
    summary: Dict[str, Any] = {"archs": []}

    for bp in blueprints:
        name = bp.get("name") or f"arch_{int(time.time())}"
        print(f"[orchestrator] Starting blueprint: {name}")

        try:
            arch_id = dkb.add_architecture(name, bp, features={})
        except Exception as e:
            print(f"[ERROR] Failed to insert architecture into DKB: {e}")
            summary["archs"].append({"arch_id": None, "trial_id": None, "error": f"dkb.insert_failed: {e}"})
            continue

        run_id = str(uuid.uuid4())[:8]
        default_low = {"use_synthetic": True, "epochs": 1, "batch_size": 64, "lr": 0.01, "patience": 1}
        train_cfg = dict(low_fidelity_cfg or default_low)

        cfg = {
            "blueprint": bp,
            "data": {"root": "./data"},
            "train": train_cfg,
            "device": device,
            "log_dir": f"./logs/campaign_{run_id}",
            "seed": seed
        }

        start_ts = time.time()
        try:
            print(f"[orchestrator] Training arch_id={arch_id} run_id={run_id} ...")
            res = train_one_model(cfg)
            if not isinstance(res, dict):
                res = {"best_val_acc": None, "best_checkpoint": None, "params": None, "latency_ms": None, "flops": None, "log_dir": cfg["log_dir"], "seed": seed}
        except Exception as e:
            end_ts = time.time()
            try:
                trial_id = dkb.add_trial(arch_id, run_id, cfg, checkpoint_path=None, start_ts=start_ts, end_ts=end_ts)
            except Exception as ex:
                trial_id = None
                print(f"[WARN] Failed to add failed trial to DKB: {ex}")
            summary["archs"].append({"arch_id": arch_id, "trial_id": trial_id, "error": f"train_failed: {e}"})
            print(f"[orchestrator] Training failed for arch_id={arch_id}: {e}")
            continue

        end_ts = time.time()
        checkpoint_path = res.get("best_checkpoint")
        try:
            trial_id = dkb.add_trial(arch_id, run_id, cfg, checkpoint_path=checkpoint_path, start_ts=start_ts, end_ts=end_ts)
        except Exception as e:
            trial_id = None
            print(f"[WARN] Failed to add trial to DKB: {e}")

        try:
            lat = res.get("latency_ms", {})
            cpu_lat = lat.get("cpu") if isinstance(lat, dict) else None
            cuda_lat = lat.get("cuda") if isinstance(lat, dict) else None
            best_acc = float(res.get("best_val_acc", 0.0) or 0.0)
            dkb.add_metrics(trial_id, epoch=-1, val_acc=best_acc, val_loss=float(res.get("best_val_loss", 0.0) or 0.0), latency_cpu_ms=cpu_lat, latency_cuda_ms=cuda_lat)
        except Exception as e:
            print(f"[WARN] Failed to add summary metrics to DKB for trial {trial_id}: {e}")

        summary["archs"].append({"arch_id": arch_id, "trial_id": trial_id, "result": res})

        try:
            blueprint_path = Path(cfg["log_dir"]) / "blueprint.json"
            blueprint_path.parent.mkdir(parents=True, exist_ok=True)
            blueprint_path.write_text(json.dumps(bp, indent=2))

            cmd = [
                "python", "-m", "src.orchestrator.post_metrics",
                "--dkb", dkb_path,
                "--arch_id", str(arch_id),
                "--trial_id", str(trial_id),
                "--blueprint", str(blueprint_path),
                "--device", device,
                "--runs", str(max(10, int(post_metrics_runs)))
            ]
            print(f"[orchestrator] Calling post-metrics: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[WARN] Post-metrics subprocess failed for arch_id={arch_id}, trial_id={trial_id}: {e}")

    dkb.close()
    return summary
