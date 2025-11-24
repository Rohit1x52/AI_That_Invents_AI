from typing import List, Dict, Any, Optional
import copy

from src.generator.heuristic import sample_candidates, estimate_params as heuristic_estimate
from src.generator.predictor import ParamPredictor
from src.generator.latency_model import LatencyModel

# optional compute_flops utility (Phase-1: src.eval.flops_utils)
try:
    from src.eval.flops_utils import compute_flops
    _HAS_FLOPS_UTIL = True
except Exception:
    _HAS_FLOPS_UTIL = False

def sample_and_filter(seed_bp: Dict[str, Any],
                      n: int = 10,
                      params_max: Optional[int] = None,
                      latency_max_ms: Optional[float] = None,
                      device: str = "cpu",
                      mutation_mode: str = "balanced",
                      seed: Optional[int] = None) -> List[Dict[str, Any]]:
    predictor = ParamPredictor()
    latency_model = LatencyModel()

    raw_cands = sample_candidates(seed_bp, n=n, seed=seed, params_max=None, mutation_mode=mutation_mode)
    out = []
    for c in raw_cands:
        cand = copy.deepcopy(c)
        # estimate params
        est_params = predictor.estimate_params(cand)
        cand["est_params"] = est_params

        # estimate flops if utility available; fallback to heuristic: est_flops = est_params * 20
        est_flops = None
        if _HAS_FLOPS_UTIL:
            try:
                est_flops = compute_flops_from_blueprint(cand)
            except Exception:
                est_flops = None
        if est_flops is None:
            # crude fallback: multiply params by factor to get FLOPs (~20)
            est_flops = int(est_params * 20)
        cand["est_flops"] = est_flops

        # predict latency
        est_latency_ms = latency_model.predict_latency_ms(est_flops, device=device)
        cand["est_latency_ms"] = est_latency_ms

        # filter by params and latency if provided
        if params_max is not None and est_params > params_max:
            continue
        if latency_max_ms is not None and est_latency_ms is not None and est_latency_ms > latency_max_ms:
            continue

        out.append(cand)
    return out

# small helper that wraps compute_flops util (if present)
def compute_flops_from_blueprint(bp: Dict[str, Any]) -> int:
    if not _HAS_FLOPS_UTIL:
        raise RuntimeError("compute_flops unavailable")
    # render model and compute flops
    return compute_flops(bp, tuple(bp.get("input_shape", [3,32,32])))
