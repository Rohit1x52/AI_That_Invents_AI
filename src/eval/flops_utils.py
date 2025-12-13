import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Any

def compute_flops(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Optional[float]:
    model_mode = model.training
    model.eval()
    
    original_device = next(model.parameters()).device
    model.to(device)
    
    dummy_input = torch.randn(1, *input_shape).to(device)
    flops = None

    try:
        from fvcore.nn import FlopCountAnalysis
        flops_counter = FlopCountAnalysis(model, dummy_input)
        flops_counter.unsupported_ops_warnings(False)
        flops = flops_counter.total()
    except ImportError:
        try:
            import thop
            flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        except ImportError:
            pass
    except Exception:
        pass
    finally:
        model.to(original_device)
        model.train(model_mode)

    return flops