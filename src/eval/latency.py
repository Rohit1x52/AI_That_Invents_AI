import time
import torch

def measured_latency(model, input_shape, device="cpu", runs=50, warmup=10):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this machine.")

    model = model.to(device)
    model.eval()
    inp = torch.randn((1, *input_shape), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(inp)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.time() - start)

    import statistics
    return statistics.mean(times) * 1000.0  # return ms


def measured_latency_device_list(model, input_shape, devices=("cpu", "cuda"),
                                 runs=30, warmup=5):
    results = {}
    for d in devices:
        try:
            if d == "cuda" and not torch.cuda.is_available():
                results[d] = None
                continue
            results[d] = measured_latency(
                model, input_shape, device=d, runs=runs, warmup=warmup
            )
        except Exception:
            results[d] = None
    return results


def estimate_latency_theoretical(model, input_shape, device_peak_flops=None):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        print("fvcore required for theoretical latency computation.")
        return None

    import torch
    model_cpu = model.cpu()
    try:
        dummy = torch.randn(1, *input_shape)
        flops = FlopCountAnalysis(model_cpu, dummy).total()
        if device_peak_flops is None:
            print("Pass device_peak_flops (e.g., 10e12 for 10TFLOPS)")
            return None
        return flops / float(device_peak_flops)
    except Exception as e:
        print("FLOPs analysis failed:", e)
        return None
