from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math

from src.generator.heuristic import sample_candidates
from src.generator.param_predictor import estimate_params
from src.generator.latency_model import estimate_latency_from_blueprint

def compute_arch_hash(bp: Dict[str, Any]) -> str:
    s = json.dumps(bp.get("stages", []), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

def evaluate_candidate(c: Dict[str, Any], device: str) -> Dict[str, Any]:
    p = estimate_params(c)
    lat = estimate_latency_from_blueprint(c, device=device)
    c["est_params"] = p.get("est_params", 0)
    c["est_flops"] = lat.get("est_flops", 0)
    c["est_latency_ms"] = lat.get("est_latency_ms", 0.0)
    c["hash"] = compute_arch_hash(c)
    return c

def sample_and_filter(
    seed_bp: Dict[str, Any],
    n: int = 10,
    params_max: Optional[int] = None,
    latency_max_ms: Optional[float] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    mutation_mode: str = "balanced",
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    
    oversample_n = n * 3
    raw_candidates = sample_candidates(seed_bp, n=oversample_n, seed=seed)
    evaluated_candidates = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_candidate, c, device) for c in raw_candidates]
        for f in futures:
            evaluated_candidates.append(f.result())
    valid = []
    seen_hashes = set()

    for c in evaluated_candidates:
        if c["hash"] in seen_hashes:
            continue
            
        is_valid = True
        if params_max and c["est_params"] > params_max:
            is_valid = False
        if latency_max_ms and c["est_latency_ms"] > latency_max_ms:
            is_valid = False
            
        if is_valid:
            valid.append(c)
            seen_hashes.add(c["hash"])
        
        if len(valid) >= n:
            break

    if len(valid) >= n:
        return valid[:n]
    def violation_score(x):
        p_score = max(0, (x["est_params"] - (params_max or 0))) / (params_max or 1)
        l_score = max(0, (x["est_latency_ms"] - (latency_max_ms or 0))) / (latency_max_ms or 1)
        return p_score + l_score

    remaining_needed = n - len(valid)
    others = [c for c in evaluated_candidates if c["hash"] not in seen_hashes]
    others.sort(key=violation_score)
    
    return valid + others[:remaining_needed]