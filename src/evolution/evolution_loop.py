import random
import statistics
import copy
from typing import List, Dict, Any, Optional
try:
    from src.generator.sample_pipeline import evolve_from_parent
    from src.evolution.dna import dna_distance
    from src.eval.pareto import pareto_front
except ImportError:
    evolve_from_parent = lambda **kwargs: kwargs.get("parent_bp")
    dna_distance = lambda a, b: 0.1
    pareto_front = lambda pop: sorted(pop, key=lambda x: x["metrics"]["val_acc"], reverse=True)[:5]

class EvolutionController:
    def __init__(
        self,
        max_generations: int = 20,
        population_size: int = 20,
        stagnation_limit: int = 5,
        novelty_weight: float = 0.0,
        seed: int = 42,
    ):
        self.max_generations = max_generations
        self.pop_size = population_size
        self.stagnation_limit = stagnation_limit
        self.novelty_weight = novelty_weight
        
        self.rng = random.Random(seed)
        self.history: List[Dict] = []
        self.best_global_acc = 0.0
        self.stagnation_counter = 0

    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calculates average DNA distance between all pairs."""
        if len(population) < 2: return 0.0
        sample = population if len(population) < 50 else self.rng.sample(population, 50)
        
        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                d = dna_distance(
                    sample[i].get("dna", {}), 
                    sample[j].get("dna", {})
                )
                distances.append(d)
        
        return statistics.mean(distances) if distances else 0.0

    def _tournament_selection(self, candidates: List[Dict], k: int = 3) -> Dict:
        """Selects the best parent from k random individuals."""
        pool = self.rng.sample(candidates, min(k, len(candidates)))
        return max(pool, key=lambda x: x.get("metrics", {}).get("val_acc", 0))

    def evolve(
        self,
        initial_population: List[Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        population = initial_population
        current_mutation_rate = "balanced"

        print(f"--- Evolution Start: {len(population)} individuals ---")

        for gen in range(1, self.max_generations + 1):
            elites = pareto_front(population)
            gen_best_acc = max((p["metrics"].get("val_acc", 0) for p in population), default=0)
            if gen_best_acc > self.best_global_acc + 0.001: 
                self.best_global_acc = gen_best_acc
                self.stagnation_counter = 0
                print(f"Gen {gen}: New Best Accuracy: {self.best_global_acc:.4f}")
            else:
                self.stagnation_counter += 1
                print(f"Gen {gen}: Stagnation {self.stagnation_counter}/{self.stagnation_limit}")

            if self.stagnation_counter >= self.stagnation_limit:
                print("Early Stopping: Algorithm Stagnated.")
                break
            diversity = self._calculate_diversity(population)
            if diversity < 0.1:
                current_mutation_rate = "expand" 
                print("-> Diversity Low: Switching to EXPLORATION mode")
            else:
                current_mutation_rate = "balanced"
            next_generation = []
            sorted_elites = sorted(elites, key=lambda x: x["metrics"].get("val_acc", 0), reverse=True)
            next_generation.extend(sorted_elites[:2])
            while len(next_generation) < self.pop_size:
                parent = self._tournament_selection(elites, k=3)
                child = evolve_from_parent(
                    parent_bp=parent["blueprint"],
                    critic=parent.get("critic", {}),
                    metrics=parent.get("metrics", {}),
                    constraints=constraints,
                    seed=self.rng.randint(0, 1_000_000),
                )
                next_generation.append(child)
            self.history.append({
                "gen": gen,
                "best_acc": gen_best_acc,
                "diversity": diversity,
                "pop_size": len(population)
            })
            
            population = next_generation

        return sorted_elites