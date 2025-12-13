import json
import pytest
from src.generator.diversity import diverse_sample_candidates

SEED_BP = {
    "name": "test_seed",
    "input_shape": [3, 32, 32],
    "num_classes": 10,
    "backbone": "convnet",
    "stages": [
        {"type": "conv_block", "filters": 16, "depth": 1, "kernel": 3}
    ]
}

def test_diversity_logic():
    target_k = 5
    max_params = 1000000 
    
    cands = diverse_sample_candidates(
        SEED_BP,
        pool_n=50,
        select_k=target_k,
        seed=123,
        params_max=max_params
    )
    
    assert isinstance(cands, list)
    assert len(cands) == target_k
    
    signatures = set()
    for c in cands:
        assert c.get("est_params", 0) <= max_params
        
        stages_str = json.dumps(c.get("stages", []), sort_keys=True)
        signatures.add(stages_str)
        
    assert len(signatures) == len(cands)

def test_diversity_fallback():
    tiny_max_params = 100 
    
    cands = diverse_sample_candidates(
        SEED_BP,
        pool_n=10,
        select_k=5,
        seed=42,
        params_max=tiny_max_params
    )
    
    assert len(cands) > 0