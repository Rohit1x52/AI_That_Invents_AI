import argparse, json, math, statistics, os
import sys
from pathlib import Path
from typing import List
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.eval.flops_utils import compute_flops
from src.eval.latency import measured_latency 
from src.dkb.client_sqlite import DKBClient

def collect_samples_from_dkb(dkb_path: str, limit: int = 10):
    dkb = DKBClient(dkb_path)
    arches = dkb.query_architectures()
    dkb.close()
    samples = []
    for a in arches[:limit]:
        bp = a.get("blueprint_json")
        if isinstance(bp, str):
            try:
                bp = json.loads(bp)
            except Exception:
                bp = None
        if bp:
            samples.append(bp)
    return samples

def compute_flops_safe(bp):
    try:
        return compute_flops(bp, tuple(bp.get("input_shape",[3,32,32])))
    except Exception:
        # fallback: crude conv-based estimate
        est = 0
        in_ch = bp.get("input_shape",[3])[0]
        for s in bp.get("stages", []):
            out_ch = int(s.get("filters", 32))
            k = int(s.get("kernel", 3))
            d = int(s.get("depth", 1))
            est += (k*k) * in_ch * out_ch * d
            in_ch = out_ch
        return int(est * 2)  # approximate multiplier

def measure_for_bp(bp, device, runs=20, warmup=5):
    # build model from blueprint and run measured_latency on device
    from src.codegen.blueprint import Blueprint
    from src.codegen.renderer import render_blueprint
    import torch
    bp_obj = Blueprint.from_dict(bp)
    model = render_blueprint(bp_obj)
    inp_shape = tuple(bp.get("input_shape",[3,32,32]))
    # measured_latency expects model and input_shape
    ms = measured_latency(model, inp_shape, device=device, runs=runs, warmup=warmup)
    return ms

def fit_linear(flops_list, ms_list):
    # solve ms = a * flops + b (flops in GFLOPs)
    X = np.vstack([np.array(flops_list)/1e9, np.ones(len(flops_list))]).T
    y = np.array(ms_list)
    # linear least squares
    coef, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
    # coef is ms per GFLOP => convert to GFLOPS/s = 1/coef
    if coef <= 0:
        gflops_per_sec = 50.0
    else:
        gflops_per_sec = 1.0 / coef
    overhead_ms = float(intercept)
    return float(gflops_per_sec), float(overhead_ms)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dkb", default="dkb.sqlite")
    p.add_argument("--out", default="device_profiles.json")
    p.add_argument("--devices", nargs="+", default=["cpu"])
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    samples = collect_samples_from_dkb(args.dkb, limit=args.limit)
    if not samples:
        print("No blueprints found in DKB or empty DKB. Please populate DKB or provide sample blueprints.")
        return

    result = {}
    for device in args.devices:
        flops_list = []
        ms_list = []
        print(f"Calibrating device: {device} on {len(samples)} samples")
        for bp in samples:
            try:
                flops = compute_flops_safe(bp)
                ms = measure_for_bp(bp, device=device, runs=args.runs, warmup=args.warmup)
                print(f" sample flops={flops}, ms={ms:.3f}")
                flops_list.append(flops)
                ms_list.append(ms)
            except Exception as e:
                print("  sample failed:", e)
        if len(flops_list) >= 2:
            gflops, overhead = fit_linear(flops_list, ms_list)
            result[device] = {"gflops_per_sec": gflops, "overhead_ms": overhead}
        else:
            result[device] = {"gflops_per_sec": 50.0, "overhead_ms": 1.0}

    Path(args.out).write_text(json.dumps(result, indent=2))
    print("Wrote device profiles to", args.out)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
