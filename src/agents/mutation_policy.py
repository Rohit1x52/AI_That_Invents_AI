from typing import Dict, Any, Optional, List

def choose_mutation_strategy(
    critic: Dict[str, Any],
    metrics: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Intelligently selects the best mutation strategy based on performance gaps.
    Prioritizes Hard Constraints > Stability > Accuracy > Efficiency.
    """
    constraints = constraints or {}
    
    scores = critic.get("scores", {})
    suggestions = critic.get("suggestions", []) 
    latency = metrics.get("latency_cpu_ms", 0.0)
    params = metrics.get("params", 0)
    acc = metrics.get("val_acc", 0.0)
    max_latency = constraints.get("latency_max_ms", float("inf"))
    max_params = constraints.get("params_max", float("inf"))
    min_acc = constraints.get("min_accuracy", 0.0)
    if latency > max_latency:
        return {
            "mode": "shrink_width",
            "reason": f"Latency ({latency:.1f}ms) exceeds limit ({max_latency}ms).",
            "priority": "critical"
        }
    
    if params > max_params:
        return {
            "mode": "reduce_depth",
            "reason": f"Size ({params//1e6}M) exceeds limit ({max_params//1e6}M).",
            "priority": "critical"
        }
    if scores.get("training_stability", 10) < 4 or acc < 0.15:
        return {
            "mode": "add_skip_connections", 
            "reason": "Model failed to converge/stabilize.",
            "priority": "high"
        }
    if acc < min_acc or scores.get("expressiveness", 10) < 5:
        if latency < max_latency * 0.7:
            return {
                "mode": "deepen", 
                "reason": "Underfitting detected, ample compute budget available.",
                "priority": "medium"
            }
        else:
            return {
                "mode": "widen_efficiently", 
                "reason": "Underfitting, but latency is tight.",
                "priority": "medium"
            }
    if scores.get("efficiency", 10) < 5:
        return {
            "mode": "bottleneck_insertion",
            "reason": "Model is accurate but inefficient.",
            "priority": "low"
        }
    if scores.get("overall", 0) > 8.0:
        return {
            "mode": "fine_tune", 
            "reason": "Excellent candidate found, exploring local neighborhood.",
            "priority": "low"
        }
    return {
        "mode": "balanced",
        "reason": "No critical defects, proceeding with balanced exploration.",
        "priority": "default"
    }