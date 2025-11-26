from src.generator.heuristic import sample_candidates, mutate_blueprint, satisfies_constraints
import copy

def make_simple_bp():
    return {
        "name":"seed",
        "input_shape":[3,32,32],
        "num_classes":10,
        "backbone":"convnet",
        "stages":[
            {"type":"conv_block","filters":32,"depth":2,"kernel":3},
            {"type":"conv_block","filters":64,"depth":1,"kernel":3}
        ]
    }

def test_mutate_and_sample():
    bp = make_simple_bp()
    cand = mutate_blueprint(bp, rng=__import__("random").Random(0))
    assert "name" in cand
    cands = sample_candidates(bp, n=5, seed=0)
    assert len(cands) == 5

def test_constraints():
    bp = make_simple_bp()
    ok = satisfies_constraints(bp, params_max=10_000_000)
    assert ok is True
    # absurdly small limit
    ok2 = satisfies_constraints(bp, params_max=1)
    assert ok2 is False
