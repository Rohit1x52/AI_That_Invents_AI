import torch
import torch.nn as nn
from typing import Dict, Any
import traceback
import gc

from .renderer import render_blueprint
from .blueprint import Blueprint


def validate_blueprint_dict(
    bp_dict: dict,
    device: str = "cpu",
    strict_shape: bool = True,
    batch: int = 2,
    timeout: float = None,
    debug: bool = False,
) -> Dict[str, Any]:
    result = {
        "valid": False,
        "error": None,
        "params": 0,
        "flops_approx": 0,
        "out_shape": None,
        "debug_trace": None,
    }

    try:
        bp = Blueprint.from_dict(bp_dict)
    except Exception as e:
        result["error"] = f"Blueprint Parsing Failed: {str(e)}"
        if debug:
            result["debug_trace"] = traceback.format_exc()
        return result

    model = None
    dummy_input = None

    try:
        model = render_blueprint(bp)
        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()
    except Exception as e:
        result["error"] = f"Model Compilation Failed: {str(e)}"
        if debug:
            result["debug_trace"] = traceback.format_exc()
        # cleanup
        if model is not None:
            try:
                del model
            except Exception:
                pass
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    try:
        try:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            result["params"] = int(params)
        except Exception:
            result["params"] = 0

        input_shape = bp.input_shape
        if not (isinstance(input_shape, list) and len(input_shape) == 3):
            raise ValueError("Blueprint input_shape must be [C, H, W]")

        dummy_input = torch.randn(batch, *input_shape, device=torch_device)

        with torch.no_grad():
            out = model(dummy_input)

        out_flat = out.view(out.size(0), -1)
        result["out_shape"] = tuple(out.shape)

        if strict_shape:
            expected_flat_shape = (batch, int(bp.num_classes))
            if out_flat.shape != expected_flat_shape:
                result["error"] = (
                    f"Shape Mismatch: Expected flattened output {expected_flat_shape}, got {tuple(out_flat.shape)}."
                    " Head configuration might be wrong."
                )
                if debug:
                    result["debug_trace"] = traceback.format_exc()
                return result

        try:
            from thop import profile
            try:
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                result["flops_approx"] = int(flops)
            except Exception:
                result["flops_approx"] = 0
        except Exception:
            result["flops_approx"] = 0

        result["valid"] = True
        return result

    except RuntimeError as e:
        err_msg = str(e)
        low = err_msg.lower()
        if "size mismatch" in low or "mat1" in low or "matmul" in low:
            result["error"] = "Layer Size Mismatch: Check channel counts between stages."
        elif "output size is too small" in low or "computed output size" in low or "would be negative" in low:
            result["error"] = "Resolution Crash: Image shrunk to 0x0 (too many pooling/strided layers)."
        elif "out of memory" in low or "cuda out of memory" in low:
            result["error"] = "OOM: Model too large for VRAM."
        else:
            result["error"] = f"Runtime Error: {err_msg}"
        if debug:
            result["debug_trace"] = traceback.format_exc()
        return result

    except Exception as e:
        result["error"] = f"Unknown Error during Forward: {str(e)}"
        if debug:
            result["debug_trace"] = traceback.format_exc()
        return result

    finally:
        if 'model' in locals() and model is not None:
            try:
                del model
            except Exception:
                pass
        if 'dummy_input' in locals() and dummy_input is not None:
            try:
                del dummy_input
            except Exception:
                pass
        try:
            gc.collect()
        except Exception:
            pass
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    test_bp = {
        "name": "test_net",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "stages": [
            {"filters": 64, "kernel": 3, "stride": 2},
            {"filters": 128, "kernel": 3, "stride": 2},
            {"filters": 256, "kernel": 3, "stride": 2},
        ],
    }

    print("Validating...")
    res = validate_blueprint_dict(test_bp, device="cpu", debug=True)

    if res["valid"]:
        print(f"Success! Params: {res['params']:,}")
    else:
        print(f"Failed: {res['error']}")
        if res.get("debug_trace"):
            print(res["debug_trace"])