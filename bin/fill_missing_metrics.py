import argparse
import json
import sqlite3
import subprocess
import tempfile
import os
import sys
from pathlib import Path

def estimate_params_from_blueprint(bp: dict) -> int:
    stages = bp.get("stages", [])
    depth = sum(int(s.get("depth", 1)) for s in stages)
    total_filters = sum(int(s.get("filters", 0) or 0) for s in stages)
    return int(total_filters * max(1, depth) * 10)

def estimate_flops_from_params(params: int) -> int:
    return int(params * 20)

def query_missing_archs(dkb_path: str):
    conn = sqlite3.connect(dkb_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, blueprint_json FROM architectures WHERE params IS NULL OR flops IS NULL")
    rows = cur.fetchall()
    conn.close()
    return rows

def update_arch_row_with_estimates(dkb_path: str, arch_id: int, params: int, flops: int, summary: dict = None):
    conn = sqlite3.connect(dkb_path)
    cur = conn.cursor()
    if summary is None:
        summary = {"est_params": params, "est_flops": flops}
    cur.execute("UPDATE architectures SET params = ?, flops = ?, summary = ? WHERE id = ?", (int(params), int(flops), json.dumps(summary), int(arch_id)))
    conn.commit()
    conn.close()

def run_post_metrics(dkb_path: str, arch_id: int, trial_id: int, blueprint_dict: dict, device: str, runs: int):
    with tempfile.TemporaryDirectory() as td:
        bp_path = Path(td) / f"blueprint_{arch_id}.json"
        bp_path.write_text(json.dumps(blueprint_dict, indent=2))
        cmd = [
            sys.executable, "-m", "src.orchestrator.post_metrics",
            "--dkb", dkb_path,
            "--arch_id", str(arch_id),
            "--trial_id", str(trial_id) if trial_id is not None else "0",
            "--blueprint", str(bp_path),
            "--device", device,
            "--runs", str(runs)
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        print("rc:", proc.returncode)
        if proc.stdout:
            print("--- stdout ---")
            print(proc.stdout)
        if proc.stderr:
            print("--- stderr ---")
            print(proc.stderr)
        return proc.returncode == 0

def find_latest_trial_for_arch(dkb_path: str, arch_id: int):
    conn = sqlite3.connect(dkb_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM trials WHERE arch_id = ? ORDER BY start_ts DESC LIMIT 1", (arch_id,))
    row = cur.fetchone()
    conn.close()
    return (row[0] if row else None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkb", type=str, default="dkb.sqlite")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true", help="Skip rows that already have params/flops (default true behavior)")
    args = parser.parse_args()

    dkb_path = args.dkb
    device = args.device
    runs = args.runs

    rows = query_missing_archs(dkb_path)
    print(f"Found {len(rows)} architectures with missing params or flops.")

    if not rows:
        print("Nothing to do.")
        return

    updated = []
    failed = []
    for arch_id, name, blueprint_json in rows:
        print(f"\n=== arch_id={arch_id} name={name} ===")
        try:
            bp = json.loads(blueprint_json) if blueprint_json else None
        except Exception:
            bp = None

        trial_id = find_latest_trial_for_arch(dkb_path, arch_id)
        print("Latest trial id:", trial_id)

        if bp is None:
            print("[WARN] blueprint_json missing or invalid; skipping post_metrics; marking with estimates=0")
            est_params = 0
            est_flops = 0
            update_arch_row_with_estimates(dkb_path, arch_id, est_params, est_flops, summary={"note":"no_blueprint"})
            updated.append(arch_id)
            continue

        try:
            ok = run_post_metrics(dkb_path, arch_id, trial_id, bp, device, runs)
            if ok:
                print(f"[OK] post_metrics succeeded for arch {arch_id}")
                updated.append(arch_id)
                continue
            else:
                print(f"[WARN] post_metrics failed (non-zero exit). Will write heuristic estimates for arch {arch_id}.")
        except Exception as e:
            print(f"[ERROR] running post_metrics for arch {arch_id}: {e}")

        try:
            est_params = estimate_params_from_blueprint(bp)
            est_flops = estimate_flops_from_params(est_params)
            update_arch_row_with_estimates(dkb_path, arch_id, est_params, est_flops, summary={"est_params": est_params, "est_flops": est_flops, "note": "fallback_estimate"})
            updated.append(arch_id)
            print(f"[FALLBACK] wrote est_params={est_params}, est_flops={est_flops}")
        except Exception as e:
            print(f"[ERROR] fallback update failed for arch {arch_id}: {e}")
            failed.append(arch_id)

    print("\nSummary:")
    print("Updated:", updated)
    print("Failed:", failed)

if __name__ == "__main__":
    main()
