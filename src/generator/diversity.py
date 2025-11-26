import random
import math
import copy
from typing import List, Dict, Any, Optional, Tuple

from src.generator.heuristic import sample_candidates, estimate_params as heuristic_estimate, enforce_mvp_compat, extract_features_from_blueprint

def blueprint_features(bp: Dict[str, Any]) -> Tuple[float, float, float]:
    feats = extract_features_from_blueprint(bp)
    return (float(feats["depth"]), float(feats["total_filters"]), float(feats["stages_count"]))


def _normalize_vectors(vectors: List[Tuple[float,float,float]]) -> List[Tuple[float,float,float]]:
    if not vectors:
        return []
    xs = [v[0] for v in vectors]
    ys = [v[1] for v in vectors]
    zs = [v[2] for v in vectors]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    def norm(val, lo, hi):
        if hi - lo <= 0:
            return 0.0
        return (val - lo) / (hi - lo)
    out = []
    for a,b,c in vectors:
        out.append((norm(a,xmin,xmax), norm(b,ymin,ymax), norm(c,zmin,zmax)))
    return out


def pairwise_distance(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])


def diverse_select(pool: List[Dict[str,Any]], k: int, seed: Optional[int] = None) -> List[Dict[str,Any]]:
    if not pool or k <= 0:
        return []
    rng = random.Random(seed)
    feats = [blueprint_features(p) for p in pool]
    norm_feats = _normalize_vectors(feats)
    start_idx = 0
    try:
        ests = [p.get("est_params", heuristic_estimate(p)) for p in pool]
        median_val = sorted(ests)[len(ests)//2]
        start_idx = min(range(len(ests)), key=lambda i: abs(ests[i]-median_val))
    except Exception:
        start_idx = rng.randrange(len(pool))

    selected = [pool[start_idx]]
    selected_idx = {start_idx}
    selected_norm = [norm_feats[start_idx]]

    while len(selected) < min(k, len(pool)):
        best_idx = None
        best_score = -1.0
        for i, nf in enumerate(norm_feats):
            if i in selected_idx:
                continue
            # min distance to selected
            md = min(pairwise_distance(nf, s) for s in selected_norm)
            if md > best_score:
                best_score = md
                best_idx = i
        if best_idx is None:
            break
        selected.append(pool[best_idx])
        selected_idx.add(best_idx)
        selected_norm.append(norm_feats[best_idx])
    return selected


def diverse_sample_candidates(seed_bp: Dict[str,Any],
                              pool_n: int = 20,
                              select_k: int = 5,
                              seed: Optional[int] = None,
                              params_max: Optional[int] = None,
                              mutation_mode: str = "balanced") -> List[Dict[str,Any]]:
    pool = sample_candidates(seed_bp, n=pool_n, seed=seed, mutation_mode=mutation_mode)
    for p in pool:
        if "est_params" not in p:
            p["est_params"] = heuristic_estimate(p)
        if "est_flops" not in p:
            p["est_flops"] = int(p["est_params"] * 20)
    if params_max is not None:
        pool = [p for p in pool if p.get("est_params", 0) <= params_max]
    if len(pool) <= select_k:
        return pool

    selected = diverse_select(pool, select_k, seed=seed)
    return selected
