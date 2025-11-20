import torch
from .renderer import render_blueprint
from .blueprint import Blueprint

def validate_blueprint_dict(bp_dict: dict, device="cpu"):
    bp = Blueprint.from_dict(bp_dict)
    model = render_blueprint(bp)
    model.to(device)
    batch = torch.randn(2, *bp.input_shape, device=device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, bp.num_classes), f"Unexpected output shape: {out.shape}"
    # quick param count
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "out_shape": out.shape}
