import json, tempfile
from src.eval.pareto import pareto_front

def test_pareto_simple():
    entries = [
        {"arch_id":1,"val_acc":0.9,"params":100,"latency_cpu_ms":10},
        {"arch_id":2,"val_acc":0.85,"params":80,"latency_cpu_ms":8},
        {"arch_id":3,"val_acc":0.92,"params":200,"latency_cpu_ms":12},
        {"arch_id":4,"val_acc":0.88,"params":90,"latency_cpu_ms":20},
    ]
    objectives = [("val_acc","max"),("params","min"),("latency_cpu_ms","min")]
    front = pareto_front(entries, objectives)
    # expects that arch 3 (highest acc) and arch 2 (lowest params/latency) may be on frontier
    assert isinstance(front, list)
    assert len(front) >= 1
