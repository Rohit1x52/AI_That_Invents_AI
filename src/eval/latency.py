import torch
import time
import numpy as np
from typing import Tuple, Dict, Optional, Union

def measure_latency_precise(
    model: torch.nn.Module, 
    input_shape: Tuple[int, ...], 
    device: str = "cpu", 
    runs: int = 100, 
    warmup: int = 20
) -> Dict[str, float]:
    
    if "cuda" in device and not torch.cuda.is_available():
        return {"mean": -1.0, "std": 0.0, "p99": 0.0}

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(1, *input_shape, device=device)

    is_cuda = device.type == "cuda"
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    if is_cuda:
        torch.cuda.synchronize()
    
    timings = []

    start_event = torch.cuda.Event(enable_timing=True) if is_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if is_cuda else None

    with torch.no_grad():
        for _ in range(runs):
            if is_cuda:
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            else:
                start_t = time.perf_counter()
                _ = model(input_tensor)
                end_t = time.perf_counter()
                timings.append((end_t - start_t) * 1000.0)

    timings = np.array(timings)
    
    return {
        "mean": float(np.mean(timings)),
        "median": float(np.median(timings)),
        "std": float(np.std(timings)),
        "min": float(np.min(timings)),
        "max": float(np.max(timings)),
        "p99": float(np.percentile(timings, 99))
    }

def benchmark_throughput(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    device: str = "cuda",
    duration_sec: float = 5.0
) -> float:
    
    if "cuda" in device and not torch.cuda.is_available():
        return 0.0

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    num_batches = 0
    
    while (time.perf_counter() - start_time) < duration_sec:
        with torch.no_grad():
            _ = model(dummy_input)
        num_batches += 1
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    total_time = time.perf_counter() - start_time
    total_images = num_batches * batch_size
    
    return total_images / total_time

def estimate_theoretical_latency(
    model: torch.nn.Module, 
    input_shape: Tuple[int, ...], 
    device_flops: float
) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
        
        model_cpu = model.cpu()
        dummy = torch.randn(1, *input_shape)
        
        flops = FlopCountAnalysis(model_cpu, dummy)
        flops.unsupported_ops_warnings(False)
        total_flops = flops.total()
        
        latency_sec = total_flops / device_flops
        return latency_sec * 1000.0
        
    except ImportError:
        return None
    except Exception:
        return None