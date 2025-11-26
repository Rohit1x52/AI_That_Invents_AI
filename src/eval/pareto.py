import json
import csv
import argparse
from typing import List, Dict, Any, Optional, Tuple
from src.dkb.client_sqlite import DKBClient
import math
import os
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


def load_entries_from_dkb(dkb_path: str) -> List[Dict[str, Any]]:
    dkb = DKBClient(dkb_path)
    arches = dkb.query_architectures()
    entries = []
    for a in arches:
        arch_id = a.get("id")
        name = a.get("name") or f"arch_{arch_id}"
        params = a.get("params")
        flops = a.get("flops")
        trials = dkb.latest_trials_for_arch(arch_id, limit=1)
        val_acc = None
        latency_cpu = None
        latency_cuda = None
        if trials:
            t = trials[0]
            metrics = dkb.get_metrics_for_trial(t["id"])
            sel = None
            for m in metrics:
                if m.get("epoch") == -1:
                    sel = m
                    break
            if sel is None and metrics:
                sel = metrics[-1]
            if sel:
                val_acc = sel.get("val_acc")
                latency_cpu = sel.get("latency_cpu_ms")
                latency_cuda = sel.get("latency_cuda_ms")
        entries.append({
            "arch_id": arch_id,
            "name": name,
            "params": None if params is None else int(params),
            "flops": None if flops is None else int(flops),
            "val_acc": None if val_acc is None else float(val_acc),
            "latency_cpu_ms": None if latency_cpu is None else float(latency_cpu),
            "latency_cuda_ms": None if latency_cuda is None else float(latency_cuda),
        })
    dkb.close()
    return entries


def _safe_compare(a: Optional[float], b: Optional[float], mode: str) -> int:
    if a is None and b is None:
        return 0
    if mode == "max":
        a_v = -math.inf if a is None else a
        b_v = -math.inf if b is None else b
        if a_v > b_v: return 1
        if a_v < b_v: return -1
        return 0
    else:
        # mode == "min"
        a_v = math.inf if a is None else a
        b_v = math.inf if b is None else b
        if a_v < b_v: return 1
        if a_v > b_v: return -1
        return 0


def dominates(x: Dict[str, Any], y: Dict[str, Any], objectives: List[Tuple[str,str]]) -> bool:
    at_least_one_strict = False
    for key, direction in objectives:
        cmp = _safe_compare(x.get(key), y.get(key), mode=direction)
        if cmp == -1:
            return False
        if cmp == 1:
            at_least_one_strict = True
    return at_least_one_strict


def pareto_front(entries: List[Dict[str, Any]], objectives: List[Tuple[str,str]]) -> List[Dict[str, Any]]:
    frontier = []
    for i, x in enumerate(entries):
        dominated = False
        to_remove = []
        for j, y in enumerate(frontier):
            if dominates(y, x, objectives):
                dominated = True
                break
            if dominates(x, y, objectives):
                to_remove.append(y)
        if not dominated:
            frontier = [f for f in frontier if f not in to_remove]
            frontier.append(x)
    return frontier


def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str, entries: List[Dict[str, Any]]):
    if not entries:
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arch_id","name","params","flops","val_acc","latency_cpu_ms","latency_cuda_ms"])
        return
    keys = ["arch_id","name","params","flops","val_acc","latency_cpu_ms","latency_cuda_ms"]
    with open(path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for e in entries:
            writer.writerow([e.get(k) for k in keys])


def plot_params_vs_acc(entries: List[Dict[str, Any]], out_path: str):
    if not _HAS_MATPLOTLIB:
        print("matplotlib not available; skipping plot.")
        return
    xs = []
    ys = []
    colors = []
    labels = []
    for e in entries:
        if e.get("params") is None or e.get("val_acc") is None:
            continue
        xs.append(e["params"])
        ys.append(e["val_acc"])
        lat = e.get("latency_cpu_ms")
        colors.append(lat if lat is not None else max(1.0, max(xs) * 0.001))
        labels.append(e.get("name", str(e.get("arch_id"))))

    if not xs:
        print("No numeric (params,val_acc) pairs to plot.")
        return

    plt.figure(figsize=(8,6))
    sc = plt.scatter(xs, ys, c=colors, cmap="viridis", s=60, alpha=0.9)
    plt.xscale("log")
    plt.xlabel("Params (log scale)")
    plt.ylabel("Validation Accuracy")
    plt.title("Params vs Val-Acc (color=latency_cpu_ms)")
    cbar = plt.colorbar(sc)
    cbar.set_label("latency_cpu_ms")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkb", type=str, required=True, help="Path to dkb.sqlite")
    parser.add_argument("--out", type=str, default="pareto_champions.json", help="Output JSON path")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--plot", type=str, default=None, help="Optional plot path (params vs val_acc)")
    args = parser.parse_args()

    entries = load_entries_from_dkb(args.dkb)
    print(f"Loaded {len(entries)} architectures from DKB")

    objectives = [
        ("val_acc", "max"),
        ("params", "min"),
        ("latency_cpu_ms", "min")
    ]

    frontier = pareto_front(entries, objectives)
    print(f"Found {len(frontier)} Pareto-optimal architectures")

    frontier_sorted = sorted(frontier, key=lambda e: (-(e.get("val_acc") or -1.0), (e.get("params") or 10**18)))

    write_json(args.out, frontier_sorted)
    print(f"Wrote champions JSON: {args.out}")
    if args.csv:
        write_csv(args.csv, frontier_sorted)
        print(f"Wrote champions CSV: {args.csv}")
    if args.plot:
        plot_params_vs_acc(entries, args.plot)

    for e in frontier_sorted:
        print(f"arch_id={e['arch_id']}, name={e['name']}, val_acc={e.get('val_acc')}, params={e.get('params')}, latency_cpu_ms={e.get('latency_cpu_ms')}")

if __name__ == "__main__":
    main()
