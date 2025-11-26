import argparse
import json
from pathlib import Path
from pprint import pprint

from src.generator.filtering import sample_and_filter
from src.generator.heuristic import sample_candidates
from src.orchestrator.local_runner import run_campaign as run_campaign_seq

def load_blueprint(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blueprint", type=str, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--dkb", type=str, default="dkb.sqlite")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--params_max", type=int, default=None)
    p.add_argument("--latency_max", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mutation_mode", type=str, default="balanced")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--parallel", type=int, default=1, help=">1 uses parallel runner")
    args = p.parse_args()

    bp = load_blueprint(Path(args.blueprint))
    print(f"Loaded blueprint: {bp.get('name','<unnamed>')}")
    # sample
    if sample_and_filter is not None:
        candidates = sample_and_filter(
            seed_bp=bp, n=args.n, params_max=args.params_max, latency_max_ms=args.latency_max,
            device=args.device, mutation_mode=args.mutation_mode, seed=args.seed
        )
    else:
        candidates = sample_candidates(bp, n=args.n, seed=args.seed)

    print(f"Sampled: {len(candidates)} candidates (after filtering).")
    for i,c in enumerate(candidates):
        print(i, c.get("name"), c.get("est_params"), c.get("est_latency_ms"))

    if args.dry_run:
        print("Dry run - exiting")
        return

    if args.parallel and args.parallel > 1:
        try:
            from src.orchestrator.parallel_runner import run_campaign_parallel
            summary = run_campaign_parallel(candidates, dkb_path=args.dkb, device=args.device, max_workers=args.parallel)
        except Exception as e:
            print(f"[WARN] parallel runner failed, falling back to sequential: {e}")
            summary = run_campaign_seq(candidates, dkb_path=args.dkb, device=args.device)
    else:
        summary = run_campaign_seq(candidates, dkb_path=args.dkb, device=args.device)
    print("Campaign summary:")
    pprint(summary)

if __name__ == "__main__":
    main()
