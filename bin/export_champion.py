import argparse, json, torch, os
from src.dkb.client_sqlite import DKBClient
from src.codegen.renderer import render_blueprint

def get_blueprint_for_arch(dkb_path, arch_id):
    dkb = DKBClient(dkb_path)
    arches = dkb.query_architectures()
    for a in arches:
        if a["id"] == arch_id:
            bp = a.get("blueprint_json")
            if isinstance(bp, str):
                bp = json.loads(bp)
            dkb.close()
            return bp
    dkb.close()
    return None

def get_best_checkpoint_for_trial(dkb_path, trial_id):
    # Look up trials table to find checkpoint path (if stored). Fallback: search logs.
    dkb = DKBClient(dkb_path)
    trials = dkb.query_trials(trial_id=trial_id)
    dkb.close()
    if trials:
        return trials[0].get("checkpoint_path") or None
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dkb", default="dkb.sqlite")
    p.add_argument("--arch", type=int, required=True)
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--out", default="exports")
    args = p.parse_args()

    bp = get_blueprint_for_arch(args.dkb, args.arch)
    if bp is None:
        raise SystemExit("Blueprint not found for arch_id=%s" % args.arch)

    # if trial supplied, try to get checkpoint; else try common log locations
    ckpt = None
    if args.trial:
        ckpt = get_best_checkpoint_for_trial(args.dkb, args.trial)

    if not ckpt:
        # fallback: search logs for campaign_* directories with best.pth and matching blueprint name
        import glob
        candidates = glob.glob("logs/**/best.pth", recursive=True)
        if candidates:
            ckpt = candidates[-1]  # best-effort
    if not ckpt:
        raise SystemExit("No checkpoint found; ensure best.pth exists for the trained model")

    model = render_blueprint(type("B", (), {})())  # placeholder: we need Blueprint class instance
    # Better: use same Blueprint dataclass: (import and build)
    from src.codegen.blueprint import Blueprint
    bp_obj = Blueprint.from_dict(bp)
    model = render_blueprint(bp_obj)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    # TorchScript
    dummy = torch.randn(1, *bp["input_shape"])
    try:
        ts = torch.jit.trace(model, dummy)
        ts_path = os.path.join(args.out, f"arch_{args.arch}.pt")
        ts.save(ts_path)
        print("Saved TorchScript:", ts_path)
    except Exception as e:
        print("TorchScript trace failed:", e)

    # ONNX
    onnx_path = os.path.join(args.out, f"arch_{args.arch}.onnx")
    try:
        torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=["input"], output_names=["output"])
        print("Saved ONNX:", onnx_path)
    except Exception as e:
        print("ONNX export failed:", e)

if __name__ == "__main__":
    main()
