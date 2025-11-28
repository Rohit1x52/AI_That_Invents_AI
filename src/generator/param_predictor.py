from typing import Dict, Any
import math

def extract_simple_features(bp: Dict[str, Any]) -> Dict[str, float]:
    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    total_filters = sum(int(s.get("filters") or 0) * int(s.get("depth",1)) for s in stages)
    avg_kernel = (sum(int(s.get("kernel",3)) for s in stages) / max(1, len(stages))) if stages else 3
    stages_count = len(stages)
    input_ch = bp.get("input_shape", [3])[0]
    return {
        "depth": float(depth),
        "total_filters": float(total_filters),
        "avg_kernel": float(avg_kernel),
        "stages_count": float(stages_count),
        "input_ch": float(input_ch)
    }

def rule_based_param_estimate(bp: Dict[str, Any]) -> int:
    feats = extract_simple_features(bp)
    stages = bp.get("stages", [])
    in_ch = bp.get("input_shape", [3])[0]
    total = 0
    for s in stages:
        out_ch = int(s.get("filters", 32))
        kernel = int(s.get("kernel", 3))
        depth = int(s.get("depth", 1))
        # per-layer conv params (approx): k^2 * in_ch * out_ch
        per_layer = (kernel ** 2) * in_ch * out_ch
        total += per_layer * depth
        in_ch = out_ch
    # add head params (fc)
    num_classes = bp.get("num_classes", 10)
    total += in_ch * num_classes
    # safety floor
    return max(1, int(total))

def estimate_params(bp: Dict[str, Any]) -> Dict[str, Any]:
    est = rule_based_param_estimate(bp)
    feats = extract_simple_features(bp)
    return {"est_params": int(est), "features": feats}
