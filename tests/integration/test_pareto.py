import unittest
from typing import List, Dict, Tuple, Any

# --- Reference Implementation (Logic being tested) ---
def dominates(p1: Dict, p2: Dict, objectives: List[Tuple[str, str]]) -> bool:
    better_in_any = False
    for key, direction in objectives:
        val1 = p1.get(key)
        val2 = p2.get(key)
        
        # Handle missing data by assuming non-dominance
        if val1 is None or val2 is None:
            return False

        if direction == "max":
            if val1 > val2: better_in_any = True
            elif val1 < val2: return False
        elif direction == "min":
            if val1 < val2: better_in_any = True
            elif val1 > val2: return False
            
    return better_in_any

def compute_pareto_front(entries: List[Dict], objectives: List[Tuple[str, str]]) -> List[Dict]:
    frontier = []
    for candidate in entries:
        is_dominated = False
        for opponent in entries:
            if candidate["id"] == opponent["id"]:
                continue
            if dominates(opponent, candidate, objectives):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(candidate)
    return frontier

# --- Enhanced Test Suite ---
class TestParetoFrontier(unittest.TestCase):
    
    def test_simple_dominance(self):
        # Model A is strictly better than Model B in both metrics
        entries = [
            {"id": "A", "acc": 0.90, "params": 100}, # Champion
            {"id": "B", "acc": 0.80, "params": 200}, # Dominated (Low acc, High params)
        ]
        objectives = [("acc", "max"), ("params", "min")]
        
        front = compute_pareto_front(entries, objectives)
        ids = {x["id"] for x in front}
        
        self.assertIn("A", ids)
        self.assertNotIn("B", ids)
        self.assertEqual(len(front), 1)

    def test_tradeoff_retention(self):
        # Model A is accurate but heavy. Model B is light but less accurate.
        # Both should be kept.
        entries = [
            {"id": "A", "acc": 0.95, "params": 1000}, # Best Acc
            {"id": "B", "acc": 0.85, "params": 100},  # Best Params
        ]
        objectives = [("acc", "max"), ("params", "min")]
        
        front = compute_pareto_front(entries, objectives)
        ids = {x["id"] for x in front}
        
        self.assertEqual(len(front), 2)
        self.assertTrue({"A", "B"} == ids)

    def test_three_objectives(self):
        # Testing Accuracy (Max), Params (Min), Latency (Min)
        entries = [
            {"id": 1, "acc": 0.90, "params": 100, "lat": 10}, # Baseline
            {"id": 2, "acc": 0.80, "params": 100, "lat": 10}, # Dominated by 1 (Lower Acc)
            {"id": 3, "acc": 0.90, "params": 110, "lat": 10}, # Dominated by 1 (Higher Params)
            {"id": 4, "acc": 0.90, "params": 100, "lat": 15}, # Dominated by 1 (Higher Latency)
            {"id": 5, "acc": 0.92, "params": 200, "lat": 20}, # Tradeoff (High Acc)
        ]
        objectives = [("acc", "max"), ("params", "min"), ("lat", "min")]
        
        front = compute_pareto_front(entries, objectives)
        ids = {x["id"] for x in front}
        
        self.assertIn(1, ids)
        self.assertIn(5, ids)
        self.assertNotIn(2, ids)
        self.assertNotIn(3, ids)
        self.assertNotIn(4, ids)

    def test_empty_input(self):
        front = compute_pareto_front([], [("acc", "max")])
        self.assertEqual(front, [])

if __name__ == "__main__":
    unittest.main()
    import unittest
from typing import List, Dict, Tuple, Any

# --- Reference Implementation (Logic being tested) ---
def dominates(p1: Dict, p2: Dict, objectives: List[Tuple[str, str]]) -> bool:
    better_in_any = False
    for key, direction in objectives:
        val1 = p1.get(key)
        val2 = p2.get(key)
        
        # Handle missing data by assuming non-dominance
        if val1 is None or val2 is None:
            return False

        if direction == "max":
            if val1 > val2: better_in_any = True
            elif val1 < val2: return False
        elif direction == "min":
            if val1 < val2: better_in_any = True
            elif val1 > val2: return False
            
    return better_in_any

def compute_pareto_front(entries: List[Dict], objectives: List[Tuple[str, str]]) -> List[Dict]:
    frontier = []
    for candidate in entries:
        is_dominated = False
        for opponent in entries:
            if candidate["id"] == opponent["id"]:
                continue
            if dominates(opponent, candidate, objectives):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(candidate)
    return frontier

# --- Enhanced Test Suite ---
class TestParetoFrontier(unittest.TestCase):
    
    def test_simple_dominance(self):
        # Model A is strictly better than Model B in both metrics
        entries = [
            {"id": "A", "acc": 0.90, "params": 100}, # Champion
            {"id": "B", "acc": 0.80, "params": 200}, # Dominated (Low acc, High params)
        ]
        objectives = [("acc", "max"), ("params", "min")]
        
        front = compute_pareto_front(entries, objectives)
        ids = {x["id"] for x in front}
        
        self.assertIn("A", ids)
        self.assertNotIn("B", ids)
        self.assertEqual(len(front), 1)

    def test_tradeoff_retention(self):
        # Model A is accurate but heavy. Model B is light but less accurate.
        # Both should be kept.
        entries = [
            {"id": "A", "acc": 0.95, "params": 1000}, # Best Acc
            {"id": "B", "acc": 0.85, "params": 100},  # Best Params
        ]
        objectives = [("acc", "max"), ("params", "min")]
        
        front = compute_pareto_front(entries, objectives)
        ids = {x["id"] for x in front}
        
        self.assertEqual(len(front), 2)
        self.assertTrue({"A", "B"} == ids)

    def test_three_objectives(self):
        # Testing Accuracy (Max), Params (Min), Latency (Min)
        entries = [
            {"id": 1, "acc": 0.90, "params": 100, "lat": 10}, # Baseline
            {"id": 2, "acc": 0.80, "params": 100, "lat": 10}, # Dominated by 1 (Lower Acc)
            {"id": 3, "acc": 0.90, "params": 110, "lat": 10}, # Dominated by 1 (Higher Params)
            {"id": 4, "acc": 0.90, "params": 100, "lat": 15}, # Dominated by 1 (Higher Latency)
            {"id": 5, "acc": 0.92, "params": 200, "lat": 20}, # Tradeoff (High Acc)
        ]
        objectives = [("acc", "max"), ("params", "min"), ("lat", "min")]
        
        front = compute_pareto_front(entries, objectives)
        ids = {x["id"] for x in front}
        
        self.assertIn(1, ids)
        self.assertIn(5, ids)
        self.assertNotIn(2, ids)
        self.assertNotIn(3, ids)
        self.assertNotIn(4, ids)

    def test_empty_input(self):
        front = compute_pareto_front([], [("acc", "max")])
        self.assertEqual(front, [])

if __name__ == "__main__":
    unittest.main()