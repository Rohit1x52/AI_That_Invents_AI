import copy
import json
import hashlib
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.generator.heuristic import sample_candidates
from src.generator.param_predictor import estimate_params
from src.generator.latency_model import estimate_latency_from_blueprint

try:
    from src.eval.flops_utils import compute_flops
    _HAS_FLOPS_UTIL = True
except ImportError:
    _HAS_FLOPS_UTIL = False

def compute_arch_hash(bp: Dict[str, Any]) -> str:
    relevant_keys = {"backbone", "stages", "head"}
    subset = {k: v for k, v in bp.items() if k in relevant_keys}
    s = json.dumps(subset, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

def evaluate_candidate(cand: Dict[str, Any], device: str) -> Dict[str, Any]:
    cand["est_params"] = estimate_params(cand)
    
    if _HAS_FLOPS_UTIL:
        try:
            input_shape = tuple(cand.get("input_shape", [3, 32, 32]))
            cand["est_flops"] = compute_flops(cand, input_shape)
        except Exception:
            cand["est_flops"] = 0
    else:
        cand["est_flops"] = 0

    lat_res = estimate_latency_from_blueprint(cand, device=device)
    cand["est_latency_ms"] = lat_res.get("est_latency_ms", 0.0)
    cand["hash"] = compute_arch_hash(cand)
    
    return cand

def sample_and_filter(
    seed_bp: Dict[str, Any],
    n: int = 10,
    params_max: Optional[int] = None,
    latency_max_ms: Optional[float] = None,
    device: str = "cpu",
    mutation_mode: str = "balanced",
    seed: Optional[int] = None,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    
    oversample = n * 2
    raw_cands = sample_candidates(seed_bp, n=oversample, seed=seed, mutation_mode=mutation_mode)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        evaluated_cands = list(executor.map(lambda c: evaluate_candidate(c, device), raw_cands))

    valid_cands = []
    seen_hashes = set()

    for cand in evaluated_cands:
        if cand["hash"] in seen_hashes:
            continue
        seen_hashes.add(cand["hash"])

        if params_max is not None and cand["est_params"] > params_max:
            continue
        if latency_max_ms is not None and cand["est_latency_ms"] > latency_max_ms:
            continue
        
        valid_cands.append(cand)
        if len(valid_cands) >= n:
            break
    
    if not valid_cands:
        def violation_score(c):
            p_score = (c["est_params"] / params_max) if params_max else 0
            l_score = (c["est_latency_ms"] / latency_max_ms) if latency_max_ms else 0
            return p_score + l_score
        
        evaluated_cands.sort(key=violation_score)
        return evaluated_cands[:n]

    return valid_cands