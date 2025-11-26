import json, time, uuid, subprocess, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from src.dkb.client_sqlite import DKBClient

# worker function uses train_worker subprocess
def _worker_train_task(bp_path: str, cfg_json: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    cmd = [ "python", "-m", "src.trainer.train_worker", "--cfg", cfg_json ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as te:
        return (124, "", f"TIMEOUT: {te}")

def run_campaign_parallel(blueprints: List[Dict[str, Any]],
                          dkb_path: str = "dkb.sqlite",
                          device: str = "cpu",
                          max_workers: int = 2,
                          low_fidelity_cfg: Optional[Dict[str, Any]] = None,
                          seed: int = 42,
                          retries: int = 1,
                          per_task_timeout: int = 300,
                          post_metrics_runs: int = 20) -> Dict[str, Any]:
    dkb = DKBClient(dkb_path)
    summary = {"archs": []}
    tasks = []
    for bp in blueprints:
        name = bp.get("name") or f"arch_{int(time.time())}"
        arch_id = dkb.add_architecture(name, bp, features={})
        run_id = str(uuid.uuid4())[:8]
        train_cfg = dict(low_fidelity_cfg or {"use_synthetic": True, "epochs": 1, "batch_size": 64, "lr": 0.01, "patience": 1})
        cfg = {"blueprint": bp, "data": {"root": "./data"}, "train": train_cfg, "device": device, "log_dir": f"./logs/campaign_{run_id}", "seed": seed}
        tasks.append({"arch_id": arch_id, "bp": bp, "cfg": cfg, "run_id": run_id})

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for t in tasks:
            cfg_json = json.dumps(t["cfg"])
            tmpdir = tempfile.mkdtemp(prefix=f"bp_{t['arch_id']}_")
            bp_path = Path(tmpdir) / "blueprint.json"
            bp_path.write_text(json.dumps(t["bp"], indent=2))
            # Submit worker task to executor (worker runs subprocess)
            fut = exe.submit(_worker_train_task, str(bp_path), cfg_json, per_task_timeout)
            futures[fut] = (t, bp_path, retries)

        for fut in as_completed(futures):
            t, bp_path, task_retries = futures[fut]
            arch_id = t["arch_id"]
            run_id = t["run_id"]
            try:
                rc, out, err = fut.result()
            except Exception as e:
                rc, out, err = 1, "", str(e)

            trial_id = None
            # record trial (best-effort)
            start_ts = int(time.time())
            end_ts = int(time.time())
            try:
                trial_id = dkb.add_trial(arch_id, run_id, t["cfg"], checkpoint_path=None, start_ts=start_ts, end_ts=end_ts)
            except Exception as e:
                print(f"[parallel] failed adding trial: {e}")

            # retry logic
            attempts = 0
            ok = (rc == 0)
            while not ok and attempts < task_retries:
                attempts += 1
                backoff = 2 ** attempts
                print(f"[parallel] retrying arch {arch_id} attempt={attempts}/{task_retries} after {backoff}s")
                time.sleep(backoff)
                rc, out, err = _worker_train_task(str(bp_path), json.dumps(t["cfg"]), per_task_timeout)
                ok = (rc == 0)

            if ok:
                # parse JSON from stdout if present
                res = {"best_val_acc": None, "best_checkpoint": None, "params": None, "latency_ms": None, "flops": None, "log_dir": t["cfg"]["log_dir"], "seed": seed}
                try:
                    last_line = out.strip().splitlines()[-1] if out.strip().splitlines() else ""
                    if last_line.startswith("{") and last_line.endswith("}"):
                        res = json.loads(last_line)
                except Exception:
                    pass

                try:
                    cpu_lat = res.get("latency_ms", {}).get("cpu") if isinstance(res.get("latency_ms"), dict) else None
                    dkb.add_metrics(trial_id, epoch=-1, val_acc=float(res.get("best_val_acc") or 0.0), val_loss=float(res.get("best_val_loss") or 0.0), latency_cpu_ms=cpu_lat, latency_cuda_ms=None)
                except Exception as e:
                    print(f"[parallel] failed to add metrics: {e}")

                # post metrics
                try:
                    cmd = [
                        "python", "-m", "src.orchestrator.post_metrics",
                        "--dkb", dkb_path,
                        "--arch_id", str(arch_id),
                        "--trial_id", str(trial_id),
                        "--blueprint", str(bp_path),
                        "--device", device,
                        "--runs", str(post_metrics_runs)
                    ]
                    subprocess.run(cmd, check=False, timeout=per_task_timeout)
                except Exception as e:
                    print(f"[parallel] post_metrics error: {e}")

                summary["archs"].append({"arch_id": arch_id, "trial_id": trial_id, "result": res})
            else:
                print(f"[parallel] worker failed for arch {arch_id}, rc={rc}, stderr={err}")
                summary["archs"].append({"arch_id": arch_id, "trial_id": trial_id, "error": f"worker_failed_rc={rc}", "stderr": err})

    dkb.close()
    return summary
