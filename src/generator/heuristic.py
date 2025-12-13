import random
import copy
from typing import Dict, List, Any, Optional, Tuple

SUPPORTED_BACKBONES = ["convnet", "resnet", "mobile_net", "transformer_lite"]

DEFAULT_STAGES = [
    {"type": "conv_block", "filters": 32, "depth": 1, "kernel": 3, "stride": 1},
    {"type": "bottleneck_block", "filters": 64, "depth": 2, "kernel": 3, "stride": 2},
    {"type": "bottleneck_block", "filters": 128, "depth": 2, "kernel": 3, "stride": 2},
]

def extract_features_from_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    
    total_filters = 0
    for s in stages:
        f = int(s.get("filters", 32))
        d = int(s.get("depth", 1))
        # Weighted by depth to represent capacity
        total_filters += f * d
        
    return {
        "depth": depth, 
        "total_filters": total_filters, 
        "stages_count": len(stages)
    }

def estimate_params_heuristic(bp: Dict[str, Any]) -> int:
    # Heuristic: Params ~= Sum(Cin * Cout * K * K * Depth)
    # This is rough but fast for pre-filtering
    total = 0
    in_ch = bp.get("input_shape", [3, 32, 32])[0]
    
    # Stem
    stem_f = bp.get("stem", {}).get("filters", 32)
    total += 3 * 3 * in_ch * stem_f
    in_ch = stem_f
    
    for s in bp.get("stages", []):
        out_ch = int(s.get("filters", 32))
        k = int(s.get("kernel", 3))
        d = int(s.get("depth", 1))
        expansion = 4 if s.get("type") == "bottleneck_block" else 1
        
        # Approximate block params
        # 1. Pointwise in (if bottleneck)
        pw_in = in_ch * (out_ch * expansion) if expansion > 1 else 0
        # 2. Spatial
        spatial = (out_ch * expansion) * (k*k)
        # 3. Pointwise out (if bottleneck)
        pw_out = (out_ch * expansion) * out_ch if expansion > 1 else 0
        
        block_params = pw_in + spatial + pw_out
        # Fallback for simple blocks
        if block_params == 0: block_params = in_ch * out_ch * k * k
        
        total += block_params * d
        in_ch = out_ch * expansion
        
    # Head
    classes = bp.get("num_classes", 10)
    total += in_ch * classes
    
    return int(total)

def enforce_mvp_compat(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    bp = copy.deepcopy(blueprint)

    if bp.get("backbone") not in SUPPORTED_BACKBONES:
        bp["backbone"] = "convnet"

    if "input_shape" not in bp: bp["input_shape"] = [3, 32, 32]
    if "num_classes" not in bp: bp["num_classes"] = 10

    if not bp.get("stages"):
        bp["stages"] = copy.deepcopy(DEFAULT_STAGES)

    # Sanitize Stages
    for i, s in enumerate(bp["stages"]):
        if not isinstance(s, dict): s = {}
        
        # Valid Types
        valid_types = ["conv_block", "bottleneck_block", "inverted_residual", "se_block"]
        s["type"] = s.get("type", "conv_block")
        if s["type"] not in valid_types: s["type"] = "conv_block"
        
        # Valid Numerics
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

def mutate_stage_smart(stage: Dict[str, Any], rng: random.Random, mode: str) -> Dict[str, Any]:
    s = stage.copy()
    
    # Probabilities change based on mode
    ops = ["depth", "filters", "kernel", "type", "se", "stride"]
    weights = [20, 30, 10, 10, 10, 5] # Default
    
    if mode == "shrink":
        weights = [40, 40, 5, 5, 5, 5] # Focus on reducing depth/width
    elif mode == "grow":
        weights = [30, 30, 10, 10, 10, 10]
        
    op = rng.choices(ops, weights=weights, k=1)[0]
    
    if op == "depth":
        delta = -1 if mode == "shrink" else (1 if mode == "grow" else rng.choice([-1, 1]))
        s["depth"] = max(1, int(s.get("depth", 1)) + delta)
        
    elif op == "filters":
        current = int(s.get("filters", 32))
        factors = [0.5, 0.75] if mode == "shrink" else ([1.25, 1.5, 2.0] if mode == "grow" else [0.75, 1.25])
        s["filters"] = max(8, int(current * rng.choice(factors)))
        
    elif op == "kernel":
        # Usually keep odd kernels
        choices = [1, 3, 5]
        s["kernel"] = rng.choice(choices)
        
    elif op == "type":
        types = ["conv_block", "bottleneck_block", "inverted_residual"]
        s["type"] = rng.choice(types)
        
    elif op == "se":
        # Toggle Squeeze-Excitation
        current_se = float(s.get("se_ratio", 0.0))
        s["se_ratio"] = 0.0 if current_se > 0 else 0.25
        
    elif op == "stride":
        # Dangerous mutation, keep rare
        s["stride"] = 2 if s.get("stride", 1) == 1 else 1
        
    return s

def mutate_blueprint_smart(bp: Dict[str, Any], rng: random.Random, mode: str = "balanced") -> Dict[str, Any]:
    out = copy.deepcopy(bp)
    stages = out.get("stages", [])
    
    # Decide Strategy: Mutate existing, Add new, Remove existing
    r = rng.random()
    
    # 60% chance: Modify a stage
    if r < 0.6 and stages:
        idx = rng.randrange(len(stages))
        stages[idx] = mutate_stage_smart(stages[idx], rng, mode)
        
    # 20% chance: Add a stage (only if not shrinking)
    elif r < 0.8 and mode != "shrink" and len(stages) < 8:
        idx = rng.randrange(len(stages))
        # Clone neighbor
        new_stage = copy.deepcopy(stages[idx])
        # Slightly vary it
        new_stage["depth"] = 1
        stages.insert(idx + 1, new_stage)
        
    # 20% chance: Remove a stage (only if not growing)
    elif mode != "grow" and len(stages) > 1:
        idx = rng.randrange(len(stages))
        stages.pop(idx)
        
    out["stages"] = stages
    out["name"] = f"{out.get('name','net')}_m{rng.randint(10,99)}"
    
    # Mutate hyperparameters (Learning Rate, Optimizer) occasionally
    if rng.random() < 0.3:
        if "optimizer" not in out: out["optimizer"] = {}
        out["optimizer"]["lr"] = rng.choice([1e-3, 5e-4, 1e-4])
        
    return out

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
    
    # Intelligent Mode Switching
    # If parent is already larger than max_params, force "shrink" mode
    if params_max and parent_size > params_max:
        actual_mode = "shrink"
    else:
        actual_mode = mutation_mode

    # Include parent if valid
    if not params_max or parent_size <= params_max:
        candidates.append(parent)

    attempts = 0
    while len(candidates) < n and attempts < n * 20:
        child = mutate_blueprint_smart(parent, rng, mode=actual_mode)
        child_size = estimate_params_heuristic(child)
        
        if params_max and child_size > params_max:
            # If child is invalid, try to fix it immediately instead of discarding
            if actual_mode != "shrink":
                # Retry with shrink
                child = mutate_blueprint_smart(parent, rng, mode="shrink")
                child_size = estimate_params_heuristic(child)
        
        if not params_max or child_size <= params_max:
            candidates.append(child)
            
        attempts += 1
        
    return candidates