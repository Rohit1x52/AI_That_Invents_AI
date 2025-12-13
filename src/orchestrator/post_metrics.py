import argparse
import json
import time
import torch
import gc
import statistics
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Local imports
try:
    from src.codegen.renderer import render_blueprint
    from src.codegen.blueprint import Blueprint
except ImportError:
    print("[ERROR] Source modules not found. Ensure run from project root.")
    exit(1)

try:
    from src.dkb.client_sqlite import DKBClient
except ImportError:
    DKBClient = None

# --- Measurement Utilities ---

def compute_model_stats(model: torch.nn.Module, input_shape: Tuple[int, ...], device: str) -> Dict[str, Any]:
    """
    Comprehensive hardware profiling: Params, FLOPs, Latency, Throughput, VRAM.
    """
    stats = {}
    
    # 1. Parameter Count
    stats["params"] = sum(p.numel() for p in model.parameters())
    stats["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. FLOPs (using fvcore or thop)
    try:
        from fvcore.nn import FlopCountAnalysis
        model_cpu = model.cpu()
        dummy_cpu = torch.randn(1, *input_shape)
        flops = FlopCountAnalysis(model_cpu, dummy_cpu)
        flops.unsupported_ops_warnings(False)
        stats["flops"] = int(flops.total())
    except ImportError:
        # Fallback approximation: Params * 20 (rough heuristic)
        stats["flops"] = int(stats["params"] * 20)
    except Exception as e:
        print(f"[WARN] FLOPs calculation failed: {e}")
        stats["flops"] = 0

    # Move to target device for timing/memory
    device = torch.device(device if torch.cuda.is_available() or device=="cpu" else "cpu")
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, *input_shape, device=device)

    # 3. Peak VRAM Usage
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input)
        stats["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        stats["peak_mem_mb"] = 0.0

    # 4. Precision Latency Measurement
    latencies = []
    warmup = 10
    runs = 50
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(runs):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event)) # Returns ms
    else:
        # CPU Timing
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    stats["latency_mean"] = statistics.mean(latencies)
    stats["latency_std"] = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    
    # 5. Throughput (Images / Sec)
    # We use a larger batch size to saturate the GPU
    batch_size = 32
    try:
        batch_input = torch.randn(batch_size, *input_shape, device=device)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10): # Run 10 batches
                _ = model(batch_input)
            if device.type == "cuda": torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
        stats["throughput_img_per_sec"] = (batch_size * 10) / total_time
    except RuntimeError:
        # OOM on large batch
        stats["throughput_img_per_sec"] = 0.0

    return stats

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkb", type=str, required=True, help="Path to dkb.sqlite")
    parser.add_argument("--arch_id", type=int, required=True, help="Architecture ID")
    parser.add_argument("--trial_id", type=int, required=False, help="Trial ID to update")
    parser.add_argument("--blueprint", type=str, required=True, help="Path to blueprint.json")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    args = parser.parse_args()

    # 1. Load Blueprint
    bp_path = Path(args.blueprint)
    if not bp_path.exists():
        print(f"[ERROR] Blueprint not found: {bp_path}")
        return

    try:
        bp_dict = json.loads(bp_path.read_text())
        bp = Blueprint.from_dict(bp_dict)
    except Exception as e:
        print(f"[ERROR] Invalid blueprint JSON: {e}")
        return

    # 2. Build Model
    print(f"--- Profiling Arch {args.arch_id} ---")
    try:
        model = render_blueprint(bp)
    except Exception as e:
        print(f"[ERROR] Model build failed: {e}")
        return

    # 3. Profile
    # Check if requested device is actually available
    target_device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {target_device}")

    try:
        stats = compute_model_stats(model, bp.input_shape, target_device)
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"[ERROR] Profiling crashed: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # Cleanup VRAM
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Update Database
    if DKBClient:
        with DKBClient(args.dkb) as dkb:
            # Update Architecture Metadata (Static info)
            conn = dkb._conn 
            # Note: We access _conn directly for custom updates or add specific methods to DKBClient
            # Ideally DKBClient should have update_architecture_stats method.
            # Here is a direct update for brevity/compatibility with your schema:
            
            summary = {
                "profiled_at": time.time(),
                "metrics": stats
            }
            
            conn.execute(
                "UPDATE architectures SET params=?, flops=?, summary=? WHERE id=?",
                (stats["params"], stats["flops"], json.dumps(summary), args.arch_id)
            )
            
            # Update Trial Metrics (Dynamic info)
            if args.trial_id:
                # We check if we profiled on CPU or CUDA to fill the right column
                lat_cpu = stats["latency_mean"] if target_device == "cpu" else None
                lat_cuda = stats["latency_mean"] if target_device.startswith("cuda") else None
                
                # We insert a summary row (epoch=-1) or update existing
                # Fetch existing val_acc to preserve it
                existing = dkb.get_metrics_for_trial(args.trial_id)
                val_acc = 0.0
                val_loss = 0.0
                if existing:
                    # Get best
                    best = max(existing, key=lambda x: x["val_acc"] or 0)
                    val_acc = best["val_acc"]
                    val_loss = best["val_loss"]

                dkb.add_metrics(
                    args.trial_id, 
                    epoch=-1,
                    metrics={
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "latency_cpu_ms": lat_cpu,
                        "latency_cuda_ms": lat_cuda
                    }
                )
            
            print("[SUCCESS] Database updated.")
    else:
        print("[WARN] DKBClient missing, skipping DB update.")

if __name__ == "__main__":
    main()