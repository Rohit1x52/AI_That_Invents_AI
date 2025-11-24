import random
import copy
from typing import Dict, List, Any, Optional, Tuple

# MVP-supported backbones for Phase-1 renderer
SUPPORTED_BACKBONES = ["convnet", "conv_mixer", "simple"]

# Default stage template used when input blueprint is missing stages
DEFAULT_STAGES = [
    {"type": "conv_block", "filters": 32, "depth": 1, "kernel": 3},
    {"type": "conv_block", "filters": 64, "depth": 1, "kernel": 3},
]


def extract_features_from_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    total_filters = sum(int(s.get("filters", 0) or 0) for s in stages)
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
    # balanced
    return {"depth": 0.4, "filters": 0.4, "kernel": 0.2}


def mutate_blueprint(
    bp: Dict[str, Any],
    rng: random.Random,
    op_probs: Optional[Dict[str, float]] = None,
    mutation_mode: str = "balanced",
) -> Dict[str, Any]:
    if op_probs is None:
        op_probs = _get_op_probs_for_mode(mutation_mode)

    out = copy.deepcopy(bp)
    stages = out.get("stages", [])
    if not stages:
        # nothing to mutate; return enforced default copy instead
        out = enforce_mvp_compat(out)
        # attach est params
        out["est_params"] = estimate_params(out)
        return out

    stage_idx = rng.randrange(len(stages))
    s = dict(stages[stage_idx])  # copy stage
    r = rng.random()
    if r < op_probs["depth"]:
        # mutate depth +/-1 (favor increasing if deepen mode)
        d = int(s.get("depth", 1))
        if mutation_mode == "deepen":
            # prefer +1
            delta = 1
        elif mutation_mode == "widen":
            delta = rng.choice([-1, 1])
        else:
            delta = rng.choice([-1, 1])
        newd = max(1, d + delta)
        s["depth"] = newd
    elif r < op_probs["depth"] + op_probs["filters"]:
        # mutate filters: multiply by 0.5 or 2 (rounded)
        f = int(s.get("filters", 32))
        if mutation_mode == "widen":
            factor = rng.choice([1.25, 1.5, 2.0])  # favor widening
        elif mutation_mode == "deepen":
            factor = rng.choice([0.75, 0.5, 1.0])  # slight tendency to shrink
        else:
            factor = rng.choice([0.5, 2.0])
        newf = max(1, int(max(1, round(f * factor))))
        s["filters"] = newf
    else:
        # mutate kernel size
        s["kernel"] = rng.choice([1, 3, 5, 7])

    out["stages"][stage_idx] = s
    # update name to reflect mutation (keeps uniqueness)
    base_name = out.get("name", "mut")
    out["name"] = f"{base_name}_m{rng.randint(0,9999)}"

    # enforce MVP compatibility before returning
    out = enforce_mvp_compat(out)
    # attach est params
    out["est_params"] = estimate_params(out)
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
    # generate until we have n candidates or we tried enough times
    while len(candidates) < n and attempts < max(50, n * 10):
        c = mutate_blueprint(original, rng, mutation_mode=mutation_mode)
        # apply constraint filter if requested
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
