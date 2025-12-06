import argparse
import json
import math
import sys
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.eval.flops_utils import compute_flops
from src.eval.latency import measured_latency
from src.dkb.client_sqlite import DKBClient
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint

def generate_synthetic_blueprint(seed: int = 0) -> Dict:
    random.seed(seed)
    depths = [1, 2, 3]
    filters = [16, 32, 64, 128]
    return {
        "name": f"synthetic_{seed}",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "backbone": "convnet",
        "stages": [
            {"type": "conv_block", "filters": random.choice(filters), "depth": random.choice(depths), "kernel": 3, "stride": 1},
            {"type": "conv_block", "filters": random.choice(filters), "depth": random.choice(depths), "kernel": 3, "stride": 2},
            {"type": "conv_block", "filters": random.choice(filters), "depth": random.choice(depths), "kernel": 3, "stride": 2},
        ]
    }

def collect_samples(dkb_path: str, limit: int = 15) -> List[Dict]:
    samples = []
    if os.path.exists(dkb_path):
        try:
            dkb = DKBClient(dkb_path)
            arches = dkb.query_architectures()
            dkb.close()
            for a in arches:
                bp = a.get("blueprint_json")
                if isinstance(bp, str):
                    try:
                        bp = json.loads(bp)
                    except Exception:
                        continue
                if bp:
                    samples.append(bp)
        except Exception as e:
            print(f"Warning: Could not read DKB ({e}). Using synthetic data.")

    needed = limit - len(samples)
    if needed > 0:
        print(f"Generating {needed} synthetic samples to reach limit...")
        for i in range(needed):
            samples.append(generate_synthetic_blueprint(seed=i))

    return samples[:limit]

def compute_flops_safe(bp: Dict) -> float:
    try:
        shape = tuple(bp.get("input_shape", [3, 32, 32]))
        return float(compute_flops(bp, shape))
    except Exception:
        total = 0.0
        c_in = bp.get("input_shape", [3, 32, 32])[0]
        H = bp.get("input_shape", [3, 32, 32])[1]
        W = bp.get("input_shape", [3, 32, 32])[2]
        for s in bp.get("stages", []):
            c_out = s.get("filters", 32)
            k = s.get("kernel", s.get("kernel_size", 3))
            d = s.get("depth", 1)
            stride = s.get("stride", 1)
            H = max(1, H // stride)
            W = max(1, W // stride)
            total += (k * k * c_in * c_out * H * W * d)
            c_in = c_out
        return float(total)

def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def measure_robust(bp: Dict, device: str, runs: int = 20) -> float:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  Skipping {device} (CUDA not available)")
        return -1.0

    try:
        bp_obj = Blueprint.from_dict(bp)
        model = render_blueprint(bp_obj)

        shape = tuple(bp.get("input_shape", [3, 32, 32]))
        batch = 1  

        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            ms = measured_latency(model, shape, device=device, runs=runs, warmup=5)
        return float(ms)
    except Exception as e:
        print(f"  Measurement failed: {e}")
        return -1.0

def fit_robust_linear(flops: List[float], ms: List[float], device_name: str) -> Tuple[float, float]:
    x_arr = np.array(flops) / 1e9 
    y_arr = np.array(ms)

    X = x_arr.reshape(-1, 1)

    coef = None
    intercept = None
    method = "LeastSquares"
    try:
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        base = LinearRegression()
        reg = RANSACRegressor(base_estimator=base, min_samples=max(2, int(0.5 * len(x_arr))), random_state=0)
        reg.fit(X, y_arr)
        if hasattr(reg.estimator_, "coef_"):
            coef = float(reg.estimator_.coef_[0])
            intercept = float(reg.estimator_.intercept_)
            method = "RANSAC"
        else:
            raise RuntimeError("RANSAC returned non-linear estimator")
    except Exception:
        A = np.vstack([x_arr, np.ones(len(x_arr))]).T
        sol, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        coef, intercept = float(sol[0]), float(sol[1])
        method = "LeastSquares"

    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(x_arr, y_arr, label="Measured", zorder=3)
        xs = np.linspace(x_arr.min(), x_arr.max(), 100)
        plt.plot(xs, coef * xs + intercept, color="red", label=f"Fit ({method})", zorder=2)
        plt.xlabel("Complexity (GFLOPs)")
        plt.ylabel("Latency (ms)")
        plt.title(f"Profile: {device_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs("plots", exist_ok=True)
        fname = f"plots/profile_{sanitize_name(device_name)}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Warning: could not save plot ({e})")

    if coef is None or coef <= 1e-6:
        gflops_per_sec = float(100.0)
    else:
        gflops_per_sec = float(max(1.0, 1000.0 / coef))

    return float(gflops_per_sec), float(intercept)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dkb", default="dkb.sqlite", help="Path to database")
    p.add_argument("--out", default="device_profiles.json", help="Output JSON path")
    p.add_argument("--devices", nargs="+", default=["cpu"], help="Devices to profile (e.g. cpu cuda:0)")
    p.add_argument("--limit", type=int, default=15, help="Number of models to test")
    p.add_argument("--runs", type=int, default=20, help="Latency runs per model")
    p.add_argument("--seed", type=int, default=0, help="Random seed for synthetic generation")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = collect_samples(args.dkb, args.limit)
    print(f"Collected {len(samples)} samples (Mixed DKB/Synthetic)")

    profiles = {}

    for device in args.devices:
        print(f"\n--- Profiling {device} ---")
        flops_data = []
        ms_data = []

        for i, bp in enumerate(samples):
            flops = compute_flops_safe(bp)
            ms = measure_robust(bp, device, runs=args.runs)

            if ms > 0:
                print(f"  [{i+1}/{len(samples)}] GFLOPs: {flops/1e9:.4f} | Latency: {ms:.3f} ms")
                flops_data.append(flops)
                ms_data.append(ms)
            else:
                print(f"  [{i+1}/{len(samples)}] measurement failed or skipped.")

        if len(flops_data) < 2:
            print(f"  Error: Not enough successful measurements for {device}. Using defaults.")
            profiles[device] = {"gflops_per_sec": 50.0, "overhead_ms": 1.0}
            continue

        speed, overhead = fit_robust_linear(flops_data, ms_data, device)
        print(f"  Result: {speed:.2f} GFLOPS/s | Overhead: {overhead:.3f} ms")
        profiles[device] = {"gflops_per_sec": speed, "overhead_ms": max(0.0, overhead)}

    Path(args.out).write_text(json.dumps(profiles, indent=2))
    print(f"\nSaved profiles to {args.out}")
    print("Check 'plots/' folder for regression visualizations.")

if __name__ == "__main__":
    main()
