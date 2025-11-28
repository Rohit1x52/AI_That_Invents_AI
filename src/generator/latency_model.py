from typing import Dict, Tuple
from src.eval.flops_utils import compute_flops
import math

DEVICE_PROFILES = {
    "cpu": {"gflops_per_sec": 50.0, "overhead_ms": 0.5},  
    "cuda": {"gflops_per_sec": 1000.0, "overhead_ms": 0.2}, 
    "mobile_v1": {"gflops_per_sec": 5.0, "overhead_ms": 2.0}
}

def flops_to_ms(flops: int, device:str="cpu") -> float:
    profile = DEVICE_PROFILES.get(device, DEVICE_PROFILES["cpu"])
    gflops = flops / 1e9
    base_sec = gflops / profile["gflops_per_sec"]
    ms = base_sec * 1000.0 + profile.get("overhead_ms", 1.0)
    return max(0.01, ms)

def estimate_latency_from_blueprint(bp: Dict, device: str = "cpu", input_shape: Tuple[int,int,int]=None) -> Dict[str, float]:
    try:
        flops = compute_flops(bp, tuple(bp.get("input_shape", input_shape or [3,32,32])))
        est_ms = flops_to_ms(flops, device=device)
        return {"est_flops": int(flops), "est_latency_ms": float(est_ms)}
    except Exception:
        from src.generator.param_predictor import rule_based_param_estimate
        params = rule_based_param_estimate(bp)
        est_ms = max(0.5, params * 2e-6)
        est_flops = int(params * 20) 
        return {"est_flops": est_flops, "est_latency_ms": float(est_ms)}
