import argparse
import json
import sys
from pathlib import Path
from pprint import pprint

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import from src
try:
    from src.generator.filtering import sample_and_filter
except Exception:
    sample_and_filter = None
from src.generator.heuristic import sample_candidates
from src.orchestrator.local_runner import run_campaign

def load_blueprint(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blueprint", type=str, required=True, help="Path to seed blueprint JSON")
    p.add_argument("--n", type=int, default=5, help="Number of candidates to sample")
    p.add_argument("--dkb", type=str, default="dkb.sqlite", help="Path to DKB sqlite")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--params_max", type=int, default=None, help="Filter: max params allowed (est)")
    p.add_argument("--latency_max", type=float, default=None, help="Filter: max latency (ms)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mutation_mode", type=str, default="balanced", choices=["balanced","widen","deepen"])
    p.add_argument("--dry_run", action="store_true", help="Only sample & filter, do not run training")
    args = p.parse_args()

    bp_path = Path(args.blueprint)
    bp = load_blueprint(bp_path)
    print(f"Loaded blueprint: {bp.get('name', '<unnamed>')}")
    print("Sampling candidates:", args.n)
    print("Params max:", args.params_max, "Latency max(ms):", args.latency_max)

    # Prefer the filtering pipeline if available
    if sample_and_filter is not None:
        print("Using sample_and_filter pipeline (predictor + latency model).")
        candidates = sample_and_filter(
            seed_bp=bp,
            n=args.n,
            params_max=args.params_max,
            latency_max_ms=args.latency_max,
            device=args.device,
            mutation_mode=args.mutation_mode,
            seed=args.seed,
        )
    else:
        print("WARNING: sample_and_filter not available; falling back to sample_candidates (no filtering).")
        candidates = sample_candidates(bp, n=args.n, seed=args.seed)

    print(f"Total sampled: {args.n}; after filtering: {len(candidates)}")
    # print brief summary of candidates
    for i, c in enumerate(candidates):
        name = c.get("name","<noname>")
        est_params = c.get("est_params", "N/A")
        est_flops = c.get("est_flops", "N/A")
        est_lat = c.get("est_latency_ms", "N/A")
        print(f"  [{i}] name={name}, est_params={est_params}, est_flops={est_flops}, est_lat_ms={est_lat}")

    if args.dry_run:
        print("Dry run requested — skipping training. Exiting.")
        return

    # Run campaign on the filtered set
    print("Starting campaign (training survivors)...")
    summary = run_campaign(candidates, dkb_path=args.dkb, low_fidelity_cfg=None, device=args.device, seed=args.seed)
    print("Campaign summary:")
    pprint(summary)

if __name__ == "__main__":
    main()
