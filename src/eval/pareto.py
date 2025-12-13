import json
import csv
import argparse
import math
import os
from typing import List, Dict, Any, Tuple
from src.dkb.client_sqlite import DKBClient

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

def load_entries_from_dkb(dkb_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(dkb_path):
        return []
    
    dkb = DKBClient(dkb_path)
    arches = dkb.query_architectures()
    entries = []

    for a in arches:
        arch_id = a.get("id")
        entry = {
            "arch_id": arch_id,
            "name": a.get("name") or f"arch_{arch_id}",
            "params": int(a.get("params") or 0),
            "flops": int(a.get("flops") or 0),
            "val_acc": 0.0,
            "latency_cpu_ms": float('inf'),
            "latency_cuda_ms": float('inf')
        }

        trials = dkb.latest_trials_for_arch(arch_id, limit=1)
        if trials:
            t = trials[0]
            metrics = dkb.get_metrics_for_trial(t["id"])
            if metrics:
                best_m = max(metrics, key=lambda x: x.get("val_acc") or 0.0)
                entry["val_acc"] = float(best_m.get("val_acc") or 0.0)
                entry["latency_cpu_ms"] = float(best_m.get("latency_cpu_ms") or 0.0)
                entry["latency_cuda_ms"] = float(best_m.get("latency_cuda_ms") or 0.0)
        
        if entry["val_acc"] > 0:
            entries.append(entry)
            
    dkb.close()
    return entries

def dominates(p1: Dict, p2: Dict, objectives: List[Tuple[str, str]]) -> bool:
    better_in_any = False
    for key, direction in objectives:
        v1 = p1.get(key, 0)
        v2 = p2.get(key, 0)
        
        if direction == "max":
            if v1 > v2: better_in_any = True
            elif v1 < v2: return False
        else:
            if v1 < v2: better_in_any = True
            elif v1 > v2: return False
            
    return better_in_any

def compute_pareto_front(entries: List[Dict], objectives: List[Tuple[str, str]]) -> List[Dict]:
    population = [e for e in entries]
    pareto_front = []
    
    for candidate in population:
        is_dominated = False
        for opponent in population:
            if candidate["arch_id"] == opponent["arch_id"]:
                continue
            if dominates(opponent, candidate, objectives):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(candidate)
            
    return sorted(pareto_front, key=lambda x: x.get(objectives[0][0], 0), reverse=(objectives[0][1]=="max"))

def plot_interactive(entries: List[Dict], champions: List[Dict], out_path: str):
    if not _HAS_PLOTLY:
        return

    champ_ids = {c["arch_id"] for c in champions}
    for e in entries:
        e["type"] = "Champion" if e["arch_id"] in champ_ids else "Candidate"
        e["size_marker"] = 15 if e["arch_id"] in champ_ids else 5

    fig = px.scatter(
        entries,
        x="latency_cpu_ms",
        y="val_acc",
        color="type",
        size="params",
        hover_data=["arch_id", "name", "params", "flops"],
        title="Pareto Frontier: Accuracy vs Latency",
        labels={"val_acc": "Validation Accuracy", "latency_cpu_ms": "Latency (CPU ms)"},
        color_discrete_map={"Champion": "red", "Candidate": "blue"}
    )
    
    out_html = out_path.replace(".png", ".html")
    fig.write_html(out_html)
    print(f"Saved interactive plot: {out_html}")

def parse_objectives(obj_str: str) -> List[Tuple[str, str]]:
    objs = []
    for part in obj_str.split(","):
        key, mode = part.split(":")
        objs.append((key.strip(), mode.strip().lower()))
    return objs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkb", type=str, default="dkb.sqlite")
    parser.add_argument("--out", type=str, default="champions.json")
    parser.add_argument("--objectives", type=str, default="val_acc:max,params:min,latency_cpu_ms:min")
    parser.add_argument("--plot", type=str, default="pareto_plot.png")
    args = parser.parse_args()

    data = load_entries_from_dkb(args.dkb)
    if not data:
        print("No valid training data found.")
        return

    objectives = parse_objectives(args.objectives)
    champions = compute_pareto_front(data, objectives)

    print(f"Analyzed {len(data)} models. Found {len(champions)} on Pareto Frontier.")
    
    with open(args.out, "w") as f:
        json.dump(champions, f, indent=2)

    print(f"{'ID':<6} {'Name':<25} {'Val Acc':<10} {'Params (M)':<12} {'Latency':<10}")
    print("-" * 70)
    for c in champions:
        print(f"{c['arch_id']:<6} {c['name'][:24]:<25} {c['val_acc']:.2%}     {c['params']/1e6:<12.2f} {c['latency_cpu_ms']:.2f}ms")

    if _HAS_PLOTLY:
        plot_interactive(data, champions, args.plot)
    elif _HAS_MATPLOTLIB:
        print("Plotly missing, falling back to static plot.")

if __name__ == "__main__":
    main()
