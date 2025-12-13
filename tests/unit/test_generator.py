import unittest
import random
import copy
import json
from src.generator.heuristic import sample_candidates, mutate_blueprint, satisfies_constraints

class TestHeuristicGenerator(unittest.TestCase):

    def setUp(self):
        self.seed_bp = {
            "name": "seed_arch",
            "input_shape": [3, 32, 32],
            "num_classes": 10,
            "backbone": "convnet",
            "stages": [
                {"type": "conv_block", "filters": 32, "depth": 1, "kernel": 3},
                {"type": "conv_block", "filters": 64, "depth": 1, "kernel": 3}
            ]
        }
        self.rng = random.Random(42)

    def test_mutation_changes_architecture(self):
        mutated = mutate_blueprint(self.seed_bp, self.rng, prefer="balanced")
        
        self.assertNotEqual(mutated["name"], self.seed_bp["name"])
        
        seed_json = json.dumps(self.seed_bp["stages"], sort_keys=True)
        mutated_json = json.dumps(mutated["stages"], sort_keys=True)
        self.assertNotEqual(seed_json, mutated_json)

    def test_sampling_population_size(self):
        population_size = 10
        candidates = sample_candidates(self.seed_bp, n=population_size, seed=1337)
        
        self.assertEqual(len(candidates), population_size)
        
        unique_names = {c["name"] for c in candidates}
        self.assertEqual(len(unique_names), population_size)

    def test_constraints_logic(self):
        # 1. High limit (Should Pass)
        self.assertTrue(satisfies_constraints(self.seed_bp, params_max=10_000_000))
        
        # 2. Impossible limit (Should Fail)
        self.assertFalse(satisfies_constraints(self.seed_bp, params_max=100))

    def test_mutation_modes(self):
        # Test "Widen" mode (should prefer increasing filters)
        widen_rng = random.Random(1)
        # Force a specific mutation path if possible, or run statistically
        # Here we just ensure it doesn't crash and returns valid structure
        mutated = mutate_blueprint(self.seed_bp, widen_rng, prefer="widen")
        self.assertIn("stages", mutated)
        self.assertTrue(len(mutated["stages"]) > 0)

    def test_input_immutability(self):
        original_copy = copy.deepcopy(self.seed_bp)
        _ = mutate_blueprint(self.seed_bp, self.rng)
        
        self.assertEqual(self.seed_bp, original_copy)

if __name__ == "__main__":
    unittest.main()