import argparse, json, sys
from src.trainer.train import train_one_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True, help="Training config JSON string")
    args = p.parse_args()
    cfg = json.loads(args.cfg)
    try:
        res = train_one_model(cfg)
        print(json.dumps(res))
        sys.exit(0)
    except Exception as e:
        print(f"TRAIN_WORKER_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
