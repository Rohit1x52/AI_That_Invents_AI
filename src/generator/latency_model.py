from typing import Dict, Optional

DEFAULT_DEVICE_PROFILES = {
    # conservative example numbers (adjust to your hardware via benchmarking)
    "cpu": {"flops_per_sec": 50e9, "overhead_ms": 1.0, "fudge": 1.5},   # 50 GFLOPS/s
    "mobile_cpu": {"flops_per_sec": 10e9, "overhead_ms": 2.0, "fudge": 2.0}, # 10 GFLOPS/s
    "cuda": {"flops_per_sec": 1e12, "overhead_ms": 0.5, "fudge": 1.2},  # 1 TFLOP/s
}

class LatencyModel:
    def __init__(self, profiles: Optional[Dict[str, Dict]] = None):
        self.profiles = profiles or DEFAULT_DEVICE_PROFILES

    def predict_latency_ms(self, flops: Optional[int], device: str = "cpu") -> Optional[float]:
        if flops is None:
            return None
        prof = self.profiles.get(device)
        if prof is None:
            # try simple fallback to 'cpu'
            prof = self.profiles.get("cpu")
            if prof is None:
                return None
        flops_per_sec = prof["flops_per_sec"]
        overhead = prof.get("overhead_ms", 0.0)
        fudge = prof.get("fudge", 1.0)
        # avoid division by zero
        if flops_per_sec <= 0:
            return None
        seconds = (float(flops) / flops_per_sec) * fudge
        ms = seconds * 1000.0 + float(overhead)
        return float(ms)
