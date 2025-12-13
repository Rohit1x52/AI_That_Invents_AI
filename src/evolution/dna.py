import hashlib
import numpy as np
from typing import Dict, Any, List

def calculate_receptive_field(stages: List[Dict[str, Any]]) -> int:
    """
    Estimates the theoretical receptive field size.
    RF_l = RF_{l-1} + (kernel_size - 1) * stride_{total}
    """
    rf = 1
    current_stride = 1
    
    for s in stages:
        k = int(s.get("kernel", 3))
        stride = int(s.get("stride", 1))
        depth = int(s.get("depth", 1))
        for _ in range(depth):
            rf += (k - 1) * current_stride
        current_stride *= stride
            
    return rf

def generate_fingerprint(stages: List[Dict[str, Any]]) -> str:
    """Creates a short, readable ID like 's3_w64-128-256_d1-2-2'"""
    widths = [str(int(s.get("filters", 0))) for s in stages]
    depths = [str(int(s.get("depth", 1))) for s in stages]
    
    signature = f"S{len(stages)}_W{'-'.join(widths)}_D{'-'.join(depths)}"
    return signature

def architecture_dna_enhanced(bp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts rich features for Meta-Learning and Profiling.
    """
    stages = bp.get("stages", [])
    backbone = bp.get("backbone", "custom")
    depths = [int(s.get("depth", 1)) for s in stages]
    widths = [int(s.get("filters", 32)) for s in stages]
    kernels = [int(s.get("kernel", 3)) for s in stages]
    strides = [int(s.get("stride", 1)) for s in stages]
    total_depth = sum(depths)
    avg_width = float(np.mean(widths)) if widths else 0.0
    rf_size = calculate_receptive_field(stages)
    total_stride = float(np.prod(strides))
    compute_intensity = sum((w**2) * d for w, d in zip(widths, depths))
    memory_proxy = sum(w * d for w, d in zip(widths, depths))

    dna = {
        "fingerprint": generate_fingerprint(stages),
        "backbone": backbone,
        "num_stages": len(stages),
        "total_depth": total_depth,
        "max_width": max(widths) if widths else 0,
        "min_width": min(widths) if widths else 0,
        "avg_width": avg_width,
        "receptive_field_est": rf_size,
        "downsample_factor": total_stride,
        "compute_intensity_score": compute_intensity,
        "memory_pressure_score": memory_proxy,
        "kernel_entropy": len(set(kernels)),
        "is_bottleneck": any("bottleneck" in s.get("type", "") for s in stages),
        "has_attention": any("se" in s.get("type", "") or s.get("se_ratio", 0) > 0 for s in stages)
    }

    return dna

def diff_dna(parent: dict, child: dict) -> dict:
    """
    Return only the genetic differences between parent and child DNA.
    """
    delta = {}

    keys = set(parent.keys()).union(child.keys())
    for k in keys:
        pv = parent.get(k)
        cv = child.get(k)
        if pv != cv:
            delta[k] = {"from": pv, "to": cv}

    return delta

architecture_dna = architecture_dna_enhanced
