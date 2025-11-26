from src.generator.diversity import diverse_sample_candidates
import json
bp = json.load(open("examples/blueprints/blueprint_convnet.json"))
def test_diverse_sample_shape():
    cands = diverse_sample_candidates(bp, pool_n=20, select_k=5, seed=42, params_max=500000)
    assert isinstance(cands, list)
    assert len(cands) <= 5
