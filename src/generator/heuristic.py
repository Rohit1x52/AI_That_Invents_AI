import random
import copy
import math
from typing import Dict, List, Any, Optional

SUPPORTED_BACKBONES = ["convnet", "resnet", "mobile_net", "transformer_lite"]

DEFAULT_STAGES = [
    {"type": "conv_block", "filters": 32, "depth": 1, "kernel": 3, "stride": 1},
    {"type": "bottleneck_block", "filters": 64, "depth": 2, "kernel": 3, "stride": 2},
    {"type": "bottleneck_block", "filters": 128, "depth": 2, "kernel": 3, "stride": 2},
]

def _align_to_hardware(val: int, multiple: int = 8) -> int:
    return max(multiple, math.ceil(val / multiple) * multiple)

def _select_target_stage(stages: List[Dict], mode: str, rng) -> int:
    if not stages: return -1
    
    costs = [(s.get("filters", 32) ** 2) * s.get("depth", 1) for s in stages]
    
    if mode == "shrink":
        return costs.index(max(costs))
    elif mode == "expand":
        return costs.index(min(costs))
    
    return rng.randrange(len(stages))

def extract_features_from_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    
    total_filters = 0
    for s in stages:
        f = int(s.get("filters", 32))
        d = int(s.get("depth", 1))
        total_filters += f * d
        
    return {
        "depth": depth, 
        "total_filters": total_filters, 
        "stages_count": len(stages)
    }

def estimate_params_heuristic(bp: Dict[str, Any]) -> int:
    total = 0
    in_ch = bp.get("input_shape", [3, 32, 32])[0]
    
    stem_f = bp.get("stem", {}).get("filters", 32)
    total += 3 * 3 * in_ch * stem_f
    in_ch = stem_f
    
    for s in bp.get("stages", []):
        out_ch = int(s.get("filters", 32))
        k = int(s.get("kernel", 3))
        d = int(s.get("depth", 1))
        expansion = 4 if s.get("type") == "bottleneck_block" else 1
        
        pw_in = in_ch * (out_ch * expansion) if expansion > 1 else 0
        spatial = (out_ch * expansion) * (k*k)
        pw_out = (out_ch * expansion) * out_ch if expansion > 1 else 0
        
        block_params = pw_in + spatial + pw_out
        if block_params == 0: block_params = in_ch * out_ch * k * k
        
        total += block_params * d
        in_ch = out_ch * expansion
        
    classes = bp.get("num_classes", 10)
    total += in_ch * classes
    
    return int(total)

estimate_params = estimate_params_heuristic

def satisfies_constraints(bp: Dict[str, Any], params_max: Optional[int] = None) -> bool:
    if params_max is None:
        return True
    return estimate_params_heuristic(bp) <= params_max

def enforce_mvp_compat(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    bp = copy.deepcopy(blueprint)

    if bp.get("backbone") not in SUPPORTED_BACKBONES:
        bp["backbone"] = "convnet"

    if "input_shape" not in bp: bp["input_shape"] = [3, 32, 32]
    if "num_classes" not in bp: bp["num_classes"] = 10

    if not bp.get("stages"):
        bp["stages"] = copy.deepcopy(DEFAULT_STAGES)

    for i, s in enumerate(bp["stages"]):
        if not isinstance(s, dict): s = {}
        
        valid_types = ["conv_block", "bottleneck_block", "inverted_residual", "se_block"]
        s["type"] = s.get("type", "conv_block")
        if s["type"] not in valid_types: s["type"] = "conv_block"
        
        s["filters"] = max(8, int(s.get("filters", 32)))
        s["depth"] = max(1, int(s.get("depth", 1)))
        
        k = int(s.get("kernel", 3))
        s["kernel"] = k if k in [1, 3, 5, 7] else 3
        
        s["stride"] = int(s.get("stride", 1))
        s["se_ratio"] = float(s.get("se_ratio", 0.0))
        
        bp["stages"][i] = s

    if "name" not in bp:
        bp["name"] = f"net_{random.randint(1000,9999)}"

    return bp

def mutate_blueprint(
    bp: Dict[str, Any], 
    rng: random.Random, 
    prefer: str = "balanced"
) -> Dict[str, Any]:
    
    new_bp = copy.deepcopy(bp)
    stages = new_bp.get("stages", [])
    
    if not stages:
        return {"blueprint": new_bp, "mutation": {"type": "noop"}}

    idx = _select_target_stage(stages, prefer, rng)
    s = stages[idx]
    
    mutation_info = {"target_stage_idx": idx, "mode": prefer}

    if prefer == "expand":
        old_f = s.get("filters", 32)
        new_f = _align_to_hardware(int(old_f * 1.25))
        s["filters"] = new_f
        mutation_info["type"] = "widen"
        mutation_info["delta"] = f"{old_f}->{new_f}"

    elif prefer == "shrink":
        if s.get("depth", 1) > 1:
            s["depth"] -= 1
            mutation_info["type"] = "prune_depth"
        else:
            old_f = s.get("filters", 32)
            new_f = max(8, _align_to_hardware(int(old_f * 0.75)))
            s["filters"] = new_f
            mutation_info["type"] = "prune_width"

    elif prefer == "stabilize":
        s["kernel"] = 3
        if "bottleneck" not in s.get("type", ""):
            s["type"] = "bottleneck_block"
            mutation_info["type"] = "upgrade_to_bottleneck"
        else:
            mutation_info["type"] = "kernel_reset"

    elif prefer == "hardware_aware":
        old_f = s.get("filters", 32)
        s["filters"] = _align_to_hardware(old_f, 8)
        s["kernel"] = 3 
        mutation_info["type"] = "hardware_align"

    elif prefer == "deepen":
        s["depth"] = s.get("depth", 1) + 1
        mutation_info["type"] = "add_layer"

    else: 
        roll = rng.random()
        if roll < 0.4:
            s["filters"] = _align_to_hardware(int(s.get("filters", 32) * 1.125))
            mutation_info["type"] = "gentle_widen"
        elif roll < 0.8:
            s["depth"] = s.get("depth", 1) + 1
            mutation_info["type"] = "add_layer"
        else:
            choices = [3, 5]
            current = s.get("kernel", 3)
            s["kernel"] = 5 if current == 3 else 3
            mutation_info["type"] = "kernel_swap"

    new_bp["stages"][idx] = s
    new_bp["_mutation"] = mutation_info
    
    return {"blueprint": new_bp, "mutation": mutation_info}

def sample_candidates(
    seed_bp: Dict[str, Any],
    n: int = 10,
    seed: Optional[int] = None,
    params_max: Optional[int] = None,
    mutation_mode: str = "balanced",
) -> List[Dict[str, Any]]:
    
    rng = random.Random(seed)
    candidates = []
    
    parent = enforce_mvp_compat(seed_bp)
    parent_size = estimate_params_heuristic(parent)
    
    if params_max and parent_size > params_max:
        actual_mode = "shrink"
    else:
        actual_mode = mutation_mode

    if not params_max or parent_size <= params_max:
        candidates.append(parent)

    attempts = 0
    while len(candidates) < n and attempts < n * 20:
        result = mutate_blueprint(parent, rng, prefer=actual_mode)
        child = result["blueprint"]
        mutation_info = result["mutation"]
        
        child_size = estimate_params_heuristic(child)
        
        if params_max and child_size > params_max:
            if actual_mode != "shrink":
                result = mutate_blueprint(parent, rng, prefer="shrink")
                child = result["blueprint"]
                mutation_info = result["mutation"]
                child_size = estimate_params_heuristic(child)
        
        if not params_max or child_size <= params_max:
            child["_mutation"] = mutation_info
            candidates.append(child)
            
        attempts += 1
        
    return candidates