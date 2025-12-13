import unittest
import torch
import torch.nn as nn
from src.orchestrator.post_metrics import (
    compute_num_params, 
    compute_flops_fallback, 
    measured_latency_ms
)

# Define a tiny dummy model for reliable testing without external deps
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

class TestMetricsProfiler(unittest.TestCase):

    def setUp(self):
        self.model = TinyModel()
        self.input_shape = (3, 8, 8)

    def test_compute_num_params(self):
        """Ensure parameter counting is exact."""
        params = compute_num_params(self.model)
        
        # Calculation:
        # Conv: (3 * 8 * 3*3) weights + 8 bias = 216 + 8 = 224
        # FC: (512 * 10) weights + 10 bias = 5120 + 10 = 5130
        # Total: 5354
        expected = 5354
        self.assertEqual(params, expected, f"Expected {expected} params, got {params}")

    def test_flops_fallback(self):
        """Ensure fallback FLOPs logic is consistent."""
        params = 1000
        flops = compute_flops_fallback(params)
        
        # Logic is params * 20
        self.assertEqual(flops, 20000)
        self.assertIsInstance(flops, int)

    def test_latency_measurement_cpu(self):
        """Ensure latency returns a reasonable float value."""
        # Run a short benchmark
        lat = measured_latency_ms(
            self.model, 
            self.input_shape, 
            device="cpu", 
            runs=5, 
            warmup=2
        )
        
        self.assertIsNotNone(lat)
        self.assertIsInstance(lat, float)
        self.assertGreater(lat, 0.0)
        self.assertLess(lat, 1000.0, "Tiny model took >1s, something is wrong")

if __name__ == "__main__":
    unittest.main()