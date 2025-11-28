import random
import copy
from typing import Dict, List, Any, Optional, Tuple

SUPPORTED_BACKBONES = ["convnet", "conv_mixer", "simple"]

DEFAULT_STAGES = [
    {"type": "conv_block", "filters": 32, "depth": 1, "kernel": 3},
    {"type": "conv_block", "filters": 64, "depth": 1, "kernel": 3},
]

def extract_features_from_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    total_filters = sum(int(s.get("filters", 0) or 0) * int(s.get("depth",1)) for s in stages)
    return {"depth": depth, "total_filters": total_filters, "stages_count": len(stages)}


def estimate_params(bp: Dict[str, Any]) -> int:
    feats = extract_features_from_blueprint(bp)
    est = int(feats["total_filters"] * max(1, feats["depth"]) * 10)
    return max(0, est)


def enforce_mvp_compat(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    bp = copy.deepcopy(blueprint)

    # Force backbone compatibility
    if bp.get("backbone") not in SUPPORTED_BACKBONES:
        bp["backbone"] = SUPPORTED_BACKBONES[0]

    # Ensure input_shape and num_classes exist (reasonable defaults if missing)
    if "input_shape" not in bp:
        bp["input_shape"] = [3, 32, 32]
    if "num_classes" not in bp:
        bp["num_classes"] = 10

    # Ensure stages present
    stages = bp.get("stages")
    if not stages:
        stages = copy.deepcopy(DEFAULT_STAGES)
        bp["stages"] = stages

    # Sanitize each stage
    for i, s in enumerate(bp["stages"]):
        # ensure dictionary
        if not isinstance(s, dict):
            s = {}
        # defaults
        s["type"] = s.get("type", "conv_block")
        s["filters"] = max(1, int(s.get("filters", 32) or 32))
        s["depth"] = max(1, int(s.get("depth", 1) or 1))
        k = int(s.get("kernel", 3) or 3)
        s["kernel"] = k if k in (1, 3, 5, 7) else 3
        bp["stages"][i] = s

    # Ensure name exists
    if "name" not in bp or not bp["name"]:
        bp["name"] = f"arch_{random.randint(0,9999)}"

    return bp


def _get_op_probs_for_mode(mode: str) -> Dict[str, float]:
    mode = (mode or "balanced").lower()
    if mode == "widen":
        return {"depth": 0.2, "filters": 0.65, "kernel": 0.15}
    if mode == "deepen":
        return {"depth": 0.65, "filters": 0.2, "kernel": 0.15}
    return {"depth": 0.4, "filters": 0.4, "kernel": 0.2}

def mutate_stage(stage: Dict[str, Any], rng: random.Random, prefer: str = "balanced") -> Dict[str, Any]:
    s = stage.copy()
    op = rng.choice(["depth", "filters", "kernel", "block"])
    if prefer == "widen" and rng.random() < 0.6:
        op = "filters"
    if prefer == "deepen" and rng.random() < 0.6:
        op = "depth"
    if op == "depth":
        s["depth"] = max(1, int(s.get("depth",1)) + rng.choice([-1, 1]))
    elif op == "filters":
        f = int(s.get("filters", 32))
        factor = rng.choice([0.5, 0.75, 1.25, 1.5, 2])
        s["filters"] = max(1, int(f * factor))
    elif op == "kernel":
        s["kernel"] = int(rng.choice([1,3,5,7]))
    elif op == "block":
        # replace block type: conv_block, depthwise, bottleneck, residual
        s["type"] = rng.choice(["conv_block", "depthwise_conv", "bottleneck_block", "residual_block"])
    return s


def mutate_blueprint(bp: Dict[str, Any], rng: random.Random, prefer: str = "balanced") -> Dict[str, Any]:
    out = copy.deepcopy(bp)
    stages = out.get("stages", [])
    if not stages:
        # add a stage
        stages = [{"type":"conv_block","filters":32,"depth":1,"kernel":3}]
        out["stages"] = stages
    # choose mutation: mutate a stage or add/remove stage
    r = rng.random()
    if r < 0.6:
        idx = rng.randrange(len(stages))
        stages[idx] = mutate_stage(stages[idx], rng, prefer=prefer)
    elif r < 0.8 and len(stages) < 6:
        # add new stage after random idx
        idx = rng.randrange(len(stages))
        new_stage = {"type":"conv_block", "filters": max(16, int(stages[idx]["filters"]*0.5)), "depth":1, "kernel":3}
        stages.insert(idx+1, new_stage)
    else:
        if len(stages) > 1:
            idx = rng.randrange(len(stages))
            stages.pop(idx)
    out["stages"] = stages
    out["name"] = out.get("name","mut") + f"_m{rng.randint(0,9999)}"
    return out

def sample_candidates(
    seed_bp: Dict[str, Any],
    n: int = 10,
    seed: Optional[int] = None,
    params_max: Optional[int] = None,
    mutation_mode: str = "balanced",
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []

    original = enforce_mvp_compat(seed_bp)
    original["est_params"] = estimate_params(original)
    candidates.append(copy.deepcopy(original))

    attempts = 0
    while len(candidates) < n and attempts < max(50, n * 10):
        c = mutate_blueprint(original, rng, prefer=mutation_mode)
        if params_max is not None:
            if not satisfies_constraints(c, params_max=params_max):
                attempts += 1
                continue
        candidates.append(c)
        attempts += 1

    return candidates

# Constraint checker
def satisfies_constraints(bp: Dict[str, Any], params_max: Optional[int] = None) -> bool:
    est_params = estimate_params(bp)
    if params_max is not None and est_params > params_max:
        return False
    return True
