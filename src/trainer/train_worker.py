import argparse
import json
import sys
import traceback
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.trainer.train import train_one_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True, help="Training config JSON string")
    p.add_argument("--out", type=str, default=None, help="Path to write result JSON")
    args = p.parse_args()

    # 1. Parse Config
    try:
        # Support loading config from file OR raw string (handles OS CLI limits)
        if os.path.exists(args.cfg):
            cfg = json.loads(Path(args.cfg).read_text())
        else:
            cfg = json.loads(args.cfg)
    except Exception as e:
        _fail(args.out, f"Config Parsing Error: {e}", traceback.format_exc())
        return

    # 2. Run Training
    try:
        # Redirect C-level logs if possible or just python logs to avoid noise
        # Note: True isolation requires capturing streams, but this helps.
        print(f"[Worker] Starting training for {cfg.get('blueprint', {}).get('name', 'unnamed')}")
        
        # Run the actual heavy lifting
        result = train_one_model(cfg)
        
        # 3. Save Success
        _success(args.out, result)

    except Exception as e:
        # 4. Handle Crash
        _fail(args.out, str(e), traceback.format_exc())

def _success(out_path, result_data):
    """Writes success JSON to file or stdout"""
    if out_path:
        Path(out_path).write_text(json.dumps(result_data, indent=2))
    else:
        # Legacy fallback
        print(json.dumps(result_data))
    sys.exit(0)

def _fail(out_path, error_msg, tb_str):
    """Writes failure JSON to file and stderr"""
    error_data = {
        "status": "FAILED",
        "best_val_acc": 0.0,
        "error": error_msg,
        "traceback": tb_str
    }
    
    # Write structured error to result file so Orchestrator knows what happened
    if out_path:
        Path(out_path).write_text(json.dumps(error_data, indent=2))
    
    # Also print to stderr for immediate visibility in logs
    print(f"TRAIN_WORKER_CRASH: {error_msg}\n{tb_str}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()