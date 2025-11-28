from typing import List, Dict, Any, Optional
from src.generator.heuristic import sample_candidates, mutate_blueprint
from src.generator.param_predictor import estimate_params
from src.generator.latency_model import estimate_latency_from_blueprint

def sample_and_filter(seed_bp: Dict[str,Any],
                      n: int = 10,
                      params_max: Optional[int] = None,
                      latency_max_ms: Optional[float] = None,
                      device: str = "cpu",
                      seed: Optional[int] = None,
                      mutation_mode: str = "balanced") -> List[Dict[str, Any]]:
    # sample raw candidates
    cands = sample_candidates(seed_bp, n=n, seed=seed)
    kept = []
    for c in cands:
        p = estimate_params(c)
        c["est_params"] = p["est_params"]
        c["features"] = p["features"]
        lat = estimate_latency_from_blueprint(c, device=device)
        c["est_flops"] = lat.get("est_flops")
        c["est_latency_ms"] = lat.get("est_latency_ms")
        # constraints
        if params_max is not None and c["est_params"] > params_max:
            continue
        if latency_max_ms is not None and c["est_latency_ms"] > latency_max_ms:
            continue
        kept.append(c)
    # if none kept, relax constraints: return top-k by params asc
    if not kept:
        sorted_c = sorted(cands, key=lambda x: x.get("est_params", 1e12))
        kept = sorted_c[:min(len(sorted_c), n)]
    return kept
