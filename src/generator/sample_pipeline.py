from typing import Dict, Any, Optional
import random
from src.agents.mutation_policy import choose_mutation_strategy
from src.generator.heuristic import mutate_blueprint

def _map_strategy_to_heuristic_mode(strategy_mode: str) -> str:
    """
    Maps the high-level strategy mode from mutation_policy to the 
    low-level prefer mode supported by heuristic.mutate_blueprint.
    """
    mapping = {
        "shrink_width": "shrink",
        "reduce_depth": "shrink",
        "add_skip_connections": "stabilize",
        "deepen": "deepen",
        "widen_efficiently": "hardware_aware",
        "widen": "expand",
        "prune": "shrink"
    }
    return mapping.get(strategy_mode, "balanced")

def evolve_from_parent(
    parent_bp: Dict[str, Any], 
    critic: Dict[str, Any], 
    metrics: Dict[str, Any], 
    rng: random.Random, 
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evolves a parent blueprint based on critic feedback and performance metrics.
    """
    strategy = choose_mutation_strategy(
        critic=critic,
        metrics=metrics,
        constraints=constraints
    )

    heuristic_mode = _map_strategy_to_heuristic_mode(strategy["mode"])

    result = mutate_blueprint(
        parent_bp,
        rng=rng,
        prefer=heuristic_mode
    )

    # Inject the policy decision into the blueprint for tracking
    if "blueprint" in result:
        result["blueprint"]["_mutation_policy"] = strategy
        # Ensure _mutation is present
        if "_mutation" not in result["blueprint"]:
             result["blueprint"]["_mutation"] = result.get("mutation", {})

    return result
