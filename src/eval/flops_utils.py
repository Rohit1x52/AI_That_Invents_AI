def compute_flops(model, input_shape):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        print("fvcore not installed. Install with: pip install fvcore")
        return None

    import torch
    model_cpu = model.cpu()
    try:
        dummy = torch.randn(1, *input_shape)
        flops = FlopCountAnalysis(model_cpu, dummy)
        return flops.total()
    except Exception as e:
        print("FLOPs computation failed:", e)
        return None
