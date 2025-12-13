import json
import os
import math
from typing import Dict, Tuple, Optional

# Default fallback profiles (used if calibration file is missing)
DEFAULT_PROFILES = {
    "cpu": {"gflops_per_sec": 40.0, "overhead_ms": 1.0, "layer_overhead_ms": 0.05},
    "cuda": {"gflops_per_sec": 2000.0, "overhead_ms": 0.5, "layer_overhead_ms": 0.01},
    "mobile": {"gflops_per_sec": 10.0, "overhead_ms": 5.0, "layer_overhead_ms": 0.1}
}

# Global cache for loaded profiles
_LOADED_PROFILES = {}

def load_device_profiles(path: str = "device_profiles.json") -> Dict[str, Dict]:
    global _LOADED_PROFILES
    if _LOADED_PROFILES:
        return _LOADED_PROFILES
        
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                _LOADED_PROFILES = json.load(f)
        except Exception:
            _LOADED_PROFILES = DEFAULT_PROFILES
    else:
        _LOADED_PROFILES = DEFAULT_PROFILES
    return _LOADED_PROFILES

def calculate_theoretical_latency(
    flops: int, 
    params: int,
    layer_count: int, 
    device: str = "cpu",
    profile_path: str = "device_profiles.json"
) -> float:
    profiles = load_device_profiles(profile_path)
    
    # Fuzzy match device name (e.g. 'cuda:0' -> 'cuda')
    profile = None
    for k in profiles:
        if k in device:
            profile = profiles[k]
            break
    if not profile:
        profile = profiles.get("cpu", DEFAULT_PROFILES["cpu"])

    # 1. Compute Time (Math / Speed)
    gflops = flops / 1e9
    speed = profile.get("gflops_per_sec", 50.0)
    compute_ms = (gflops / speed) * 1000.0

    # 2. Memory Time (Approximate Memory Bandwidth bottleneck)
    # Heuristic: 1ms per 10MB of params on standard bus
    memory_ms = (params * 4 / 1e6) / 100.0 

    # 3. Overhead (Kernel Launch + System)
    base_overhead = profile.get("overhead_ms", 0.5)
    layer_cost = profile.get("layer_overhead_ms", 0.0) * layer_count
    
    # Total Latency (Compute and Memory overlap, so we take Max + Overhead)
    # In reality, it's complex, but Max(Compute, Mem) + Overhead is a solid estimator.
    estimated_ms = max(compute_ms, memory_ms) + base_overhead + layer_cost
    
    return max(0.1, estimated_ms)

def estimate_latency_from_blueprint(
    bp: Dict, 
    device: str = "cpu", 
    input_shape: Tuple[int, ...] = None
) -> Dict[str, float]:
    
    flops = 0
    params = 0
    layers = 0
    
    # 1. Try Accurate FLOPs Calculation
    try:
        from src.eval.flops_utils import compute_flops
        shape = tuple(bp.get("input_shape", input_shape or [3, 32, 32]))
        flops = compute_flops(bp, shape)
        if flops is None: raise ValueError("FLOPs failed")
    except Exception:
        # Fallback Heuristic
        # Estimate based on filter volume
        est_flops = 0
        stages = bp.get("stages", [])
        c_in = bp.get("input_shape", [3,32,32])[0]
        for s in stages:
            c_out = s.get("filters", 32)
            d = s.get("depth", 1)
            k = s.get("kernel", 3)
            # Volume * Approx Resolution (Assuming 1/2 reduction every 2 stages)
            est_flops += (k*k * c_in * c_out * d * 16 * 16) 
            c_in = c_out
        flops = est_flops

    # 2. Estimate Params & Layer Count
    try:
        from src.generator.heuristic import estimate_params_heuristic
        params = estimate_params_heuristic(bp)
    except ImportError:
        params = flops // 20 # Crude fallback
        
    layers = sum(s.get("depth", 1) for s in bp.get("stages", []))

    # 3. Calculate Latency
    ms = calculate_theoretical_latency(flops, params, layers, device)

    return {
        "est_flops": int(flops),
        "est_params": int(params),
        "est_latency_ms": float(ms)
    }

def flops_to_ms(flops: float, device: str = "cpu") -> float:
    return calculate_theoretical_latency(flops, 0, 0, device)
