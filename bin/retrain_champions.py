import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dkb.client_sqlite import DKBClient
from src.orchestrator.local_runner import run_campaign

def get_blueprints_for_champions(dkb_path, champions_json):
    champions = json.load(open(champions_json))
    arch_ids = [c["arch_id"] for c in champions]
    dkb = DKBClient(dkb_path)
    blueprints = []
    for arch in dkb.query_architectures():
        if arch["id"] in arch_ids:
            bp = arch.get("blueprint_json")
            if isinstance(bp, str):
                bp = json.loads(bp)
            blueprints.append(bp)
    dkb.close()
    return blueprints

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dkb", default="dkb.sqlite")
    p.add_argument("--champions", default="champions.json")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--parallel", type=int, default=1, help=">1 uses parallel runner if available")
    args = p.parse_args()

    bps = get_blueprints_for_champions(args.dkb, args.champions)
    print(f"Retrieved {len(bps)} blueprints for retraining")

    high_cfg = {"epochs": args.epochs, "batch_size": args.batch, "lr": 0.01, "patience": 10, "use_synthetic": False}

    if args.parallel and args.parallel > 1:
        try:
            from src.orchestrator.parallel_runner import run_campaign_parallel
            summary = run_campaign_parallel(bps, dkb_path=args.dkb, device=args.device, max_workers=args.parallel, low_fidelity_cfg=high_cfg)
            print("Parallel retrain summary:", summary)
            return
        except Exception as e:
            print("Parallel runner not available or failed:", e)
    summary = run_campaign(bps, dkb_path=args.dkb, low_fidelity_cfg=high_cfg, device=args.device)
    print("Retrain summary:", summary)

if __name__ == "__main__":
    main()
