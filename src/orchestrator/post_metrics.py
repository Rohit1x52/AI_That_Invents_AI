import argparse
import json
import sqlite3
import time
import statistics
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

# local imports (Phase-1)
try:
    from src.codegen.renderer import render_blueprint
except Exception:
    render_blueprint = None

# local DKB client import (if present)
try:
    from src.dkb.client_sqlite import DKBClient
except Exception:
    DKBClient = None


def compute_num_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def compute_flops_fvcore(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Optional[int]:
    """
    Try computing FLOPs with fvcore FlopCountAnalysis.
    Returns integer FLOPs or None if not available or fails.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            inp = torch.randn((1, *input_shape), device=next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu")
            flops = FlopCountAnalysis(model, inp)
            total = flops.total()
            # FlopCountAnalysis returns numbers like 1e9 as int-like or a torch int; coerce to int
            return int(total)
    except Exception:
        return None


def compute_flops_fallback(est_params: int) -> int:
    """
    Crude fallback: estimate FLOPs from params using a simple multiplier.
    This is intentionally conservative and fast.
    """
    # typical conv nets FLOPs roughly a few x params; choose 20 as placeholder
    return int(max(0, est_params * 20))


def measured_latency_ms(model: torch.nn.Module, input_shape: Tuple[int, ...], device: str = "cpu", runs: int = 50, warmup: int = 10) -> Optional[float]:
    """
    Measure average latency in milliseconds for one forward pass.
    Returns mean latency in ms or None on failure.
    """
    try:
        model.to(device)
        model.eval()
        inp = torch.randn((1, *input_shape), device=device)
        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inp)
            times = []
            for _ in range(runs):
                start = time.time()
                _ = model(inp)
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        return float(statistics.mean(times) * 1000.0)
    except Exception:
        return None


def update_architecture_row(dkb_path: str, arch_id: int, params: Optional[int], flops: Optional[int], summary: Optional[Dict[str, Any]]):
    """
    Update the architectures table with params, flops, and summary JSON.
    """
    conn = sqlite3.connect(dkb_path)
    cur = conn.cursor()
    updates = []
    args = []
    if params is not None:
        updates.append("params = ?")
        args.append(int(params))
    if flops is not None:
        updates.append("flops = ?")
        args.append(int(flops))
    if summary is not None:
        updates.append("summary = ?")
        args.append(json.dumps(summary))
    if updates:
        args.append(int(arch_id))
        sql = f"UPDATE architectures SET {', '.join(updates)} WHERE id = ?"
        cur.execute(sql, tuple(args))
        conn.commit()
    conn.close()


def add_summary_metrics_to_trial(dkb_path: str, trial_id: int, val_acc: float, val_loss: float, latency_cpu_ms: Optional[float], latency_cuda_ms: Optional[float]):
    """
    Use DKBClient.add_metrics if available; otherwise insert directly into metrics table.
    We use epoch = -1 for summary rows (same convention used in orchestrator).
    """
    if DKBClient is not None:
        try:
            dkb = DKBClient(dkb_path)
            dkb.add_metrics(trial_id, epoch=-1, val_acc=val_acc, val_loss=val_loss, latency_cpu_ms=latency_cpu_ms, latency_cuda_ms=latency_cuda_ms)
            dkb.close()
            return
        except Exception:
            # fallback to direct SQL
            pass

    # fallback direct SQL insert
    conn = sqlite3.connect(dkb_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO metrics (trial_id, epoch, val_acc, val_loss, latency_cpu_ms, latency_cuda_ms) VALUES (?, ?, ?, ?, ?, ?)",
        (trial_id, -1, float(val_acc), float(val_loss), latency_cpu_ms, latency_cuda_ms)
    )
    conn.commit()
    conn.close()


def load_model_from_blueprint(blueprint, device="cpu"):
    if render_blueprint is None:
        raise RuntimeError("render_blueprint not available (src.codegen.renderer missing)")
    # If blueprint is passed as string (JSON), parse it
    if isinstance(blueprint, str):
        blueprint = json.loads(blueprint)
    # If blueprint is a dict, convert to Blueprint object
    if isinstance(blueprint, dict):
        from src.codegen.blueprint import Blueprint
        blueprint = Blueprint.from_dict(blueprint)
    model = render_blueprint(blueprint)
    model.to(device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dkb", type=str, required=True, help="Path to dkb sqlite file")
    p.add_argument("--arch_id", type=int, required=True, help="Architecture ID in architectures table to update")
    p.add_argument("--trial_id", type=int, required=False, help="Trial ID to append metrics for (optional)")
    p.add_argument("--blueprint", type=str, required=True, help="Path to blueprint JSON (used to render model)")
    p.add_argument("--device", type=str, default="cpu", help="Device for latency measurement: cpu or cuda")
    p.add_argument("--runs", type=int, default=50, help="Latency measurement runs")
    p.add_argument("--warmup", type=int, default=10, help="Latency measurement warmup iterations")
    args = p.parse_args()

    dkb_path = args.dkb
    arch_id = args.arch_id
    trial_id = args.trial_id
    blueprint_path = Path(args.blueprint)
    device = args.device
    runs = args.runs
    warmup = args.warmup

    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint JSON not found: {blueprint_path}")

    blueprint = json.loads(blueprint_path.read_text())

    print("Rendering model from blueprint...")
    try:
        model = load_model_from_blueprint(blueprint, device=device)
    except Exception as e:
        print("ERROR rendering blueprint -> model:", e)
        raise

    print("Computing true number of parameters...")
    try:
        true_params = compute_num_params(model)
    except Exception as e:
        print("Failed to compute params:", e)
        true_params = None

    print("Attempting FLOPs calculation (fvcore)...")
    true_flops = compute_flops_fvcore(model, tuple(blueprint.get("input_shape", [3, 32, 32])))
    if true_flops is None:
        print("fvcore not available or failed -> using fallback FLOPs estimator")
        true_flops = compute_flops_fallback(true_params or 0)

    print(f"Measured/estimated FLOPs: {true_flops}")

    print(f"Measuring latency on device={device} (runs={runs}, warmup={warmup}) ...")
    try:
        lat_ms = measured_latency_ms(model, tuple(blueprint.get("input_shape", [3, 32, 32])), device=device, runs=runs, warmup=warmup)
    except Exception:
        lat_ms = None

    # For CUDA case, also try GPU latency if device == 'cuda' but also measure cpu if possible
    cpu_lat = None
    cuda_lat = None
    if device.startswith("cuda"):
        cuda_lat = lat_ms
        # optionally also measure CPU latency (move model temporarily)
        try:
            cpu_lat = measured_latency_ms(model, tuple(blueprint.get("input_shape", [3, 32, 32])), device="cpu", runs=max(10, runs//5), warmup=5)
        except Exception:
            cpu_lat = None
    else:
        cpu_lat = lat_ms

    # Build summary
    summary = {
        "true_params": true_params,
        "true_flops": true_flops,
        "latency_cpu_ms": cpu_lat,
        "latency_cuda_ms": cuda_lat,
        "measured_at": time.time(),
    }

    print("Updating DKB with true metrics...")
    try:
        update_architecture_row(dkb_path, arch_id, params=true_params, flops=true_flops, summary=summary)
    except Exception as e:
        print("Failed to update architectures table:", e)
        raise

    # Optionally append a summary metrics row to trial (epoch=-1)
    if trial_id is not None:
        # note: we don't have val_acc/val_loss here; set to 0.0 placeholders if not known
        val_acc = 0.0
        val_loss = 0.0
        # try to read previous summary metrics from DKB metrics for this trial (if present), else leave 0.0
        try:
            if DKBClient is not None:
                dkb = DKBClient(dkb_path)
                # try to find latest metrics for trial_id
                mets = dkb.get_metrics_for_trial(trial_id)
                if mets:
                    # prefer the last logged val_acc/val_loss if epoch -1 or highest epoch
                    last = mets[-1]
                    val_acc = float(last.get("val_acc", 0.0) or 0.0)
                    val_loss = float(last.get("val_loss", 0.0) or 0.0)
                dkb.close()
        except Exception:
            pass

        try:
            add_summary_metrics_to_trial(dkb_path, trial_id, val_acc=val_acc, val_loss=val_loss, latency_cpu_ms=cpu_lat, latency_cuda_ms=cuda_lat)
        except Exception as e:
            print("Warning: failed to insert summary metrics row:", e)

    print("Post-metrics update complete. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
