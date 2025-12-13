import random
import math
import hashlib
import json
import statistics
from typing import List, Dict, Any, Optional, Tuple, Set

from src.generator.heuristic import sample_candidates
from src.generator.param_predictor import estimate_params as heuristic_estimate
from src.generator.latency_model import estimate_latency_from_blueprint

def compute_blueprint_hash(bp: Dict[str, Any]) -> str:
    relevant_data = {
        "backbone": bp.get("backbone"),
        "stages": bp.get("stages", []),
        "head": bp.get("head", {})
    }
    s = json.dumps(relevant_data, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def extract_feature_vector(bp: Dict[str, Any]) -> List[float]:
    stages = bp.get("stages", [])
    
    total_depth = sum(s.get("depth", 1) for s in stages)
    
    total_filters = sum(s.get("filters", 32) * s.get("depth", 1) for s in stages)
    avg_width = total_filters / max(1, total_depth)
    
    kernels = [s.get("kernel", 3) for s in stages]
    avg_kernel = statistics.mean(kernels) if kernels else 3.0
    
    num_stages = len(stages)
    
    est_params = bp.get("est_params", 0)
    if est_params == 0:
        est_params = heuristic_estimate(bp)
    
    return [
        float(total_depth),
        float(avg_width),
        float(avg_kernel),
        float(num_stages),
        math.log(max(1, est_params))
    ]

def normalize_feature_vectors(vectors: List[List[float]]) -> List[List[float]]:
    if not vectors:
        return []
        
    dims = len(vectors[0])
    normalized = []
    
    min_max = []
    for d in range(dims):
        vals = [v[d] for v in vectors]
        min_max.append((min(vals), max(vals)))
        
    for v in vectors:
        nv = []
        for d, val in enumerate(v):
            mn, mx = min_max[d]
            if mx - mn < 1e-9:
                nv.append(0.5)
            else:
                nv.append((val - mn) / (mx - mn))
        normalized.append(nv)
        
    return normalized

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def select_most_diverse(
    candidates: List[Dict[str, Any]], 
    k: int, 
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    
    if len(candidates) <= k:
        return candidates
        
    rng = random.Random(seed)
    unique_candidates = []
    seen_hashes = set()
    for c in candidates:
        h = compute_blueprint_hash(c)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_candidates.append(c)
            
    if len(unique_candidates) <= k:
        return unique_candidates
    raw_vectors = [extract_feature_vector(c) for c in unique_candidates]
    norm_vectors = normalize_feature_vectors(raw_vectors)
    center_idx = 0
    min_dist_sum = float('inf')
    
    for i, v in enumerate(norm_vectors):
        d_sum = sum(euclidean_distance(v, other) for other in norm_vectors)
        if d_sum < min_dist_sum:
            min_dist_sum = d_sum
            center_idx = i
            
    selected_indices = [center_idx]
    while len(selected_indices) < k:
        max_min_dist = -1.0
        best_candidate_idx = -1
        
        for i, v in enumerate(norm_vectors):
            if i in selected_indices:
                continue
            min_dist_to_selected = min(
                euclidean_distance(v, norm_vectors[s_idx]) 
                for s_idx in selected_indices
            )
            
            if min_dist_to_selected > max_min_dist:
                max_min_dist = min_dist_to_selected
                best_candidate_idx = i
        
        if best_candidate_idx != -1:
            selected_indices.append(best_candidate_idx)
        else:
            remaining = [i for i in range(len(unique_candidates)) if i not in selected_indices]
            if remaining:
                selected_indices.append(rng.choice(remaining))
            else:
                break
                
    return [unique_candidates[i] for i in selected_indices]

def diverse_sample_candidates(
    seed_bp: Dict[str, Any],
    pool_n: int = 50,
    select_k: int = 10,
    params_max: Optional[int] = None,
    latency_max_ms: Optional[float] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    mutation_mode: str = "balanced"
) -> List[Dict[str, Any]]:
    raw_pool = sample_candidates(seed_bp, n=pool_n, seed=seed, mutation_mode=mutation_mode)
    valid_pool = []
    for bp in raw_pool:
        bp["est_params"] = heuristic_estimate(bp)
        lat_info = estimate_latency_from_blueprint(bp, device=device)
        bp["est_latency_ms"] = lat_info.get("est_latency_ms", 0.0)
        if params_max and bp["est_params"] > params_max:
            continue
        if latency_max_ms and bp["est_latency_ms"] > latency_max_ms:
            continue
            
        valid_pool.append(bp)
    if not valid_pool:
        raw_pool.sort(key=lambda x: x.get("est_params", 1e9))
        return raw_pool[:select_k]
    return select_most_diverse(valid_pool, k=select_k, seed=seed)