import pytest
import torch
from src.codegen.blueprint import Blueprint
from src.codegen.renderer import render_blueprint
from src.codegen.validator import validate_blueprint_dict

# 1. Happy Path: Test various valid configurations
@pytest.mark.parametrize("backbone", ["convnet", "simple"])
def test_valid_blueprints(backbone):
    bp_dict = {
        "name": f"test_{backbone}",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "backbone": backbone,
        "stages": [
            {"type": "conv_block", "filters": 16, "depth": 1, "kernel": 3},
            {"type": "conv_block", "filters": 32, "depth": 1, "kernel": 3}
        ]
    }
    
    # Step 1: Validation
    meta = validate_blueprint_dict(bp_dict, device="cpu")
    assert meta["params"] > 0
    # validate_blueprint_dict typically uses a dummy batch of size 2
    assert meta["out_shape"] == (2, 10)

    # Step 2: Rendering (The real test)
    bp = Blueprint.from_dict(bp_dict)
    model = render_blueprint(bp)
    
    assert isinstance(model, torch.nn.Module)
    
    # Step 3: Forward Pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10)

# 2. Negative Path: Test invalid inputs
def test_invalid_blueprint_structure():
    # Missing 'stages' should fail
    bad_bp = {
        "name": "broken",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "backbone": "convnet"
        # stages missing
    }
    
    with pytest.raises((AssertionError, KeyError, ValueError)):
        validate_blueprint_dict(bad_bp, device="cpu")

# 3. Shape Mismatch Test
def test_blueprint_shape_propagation():
    """Ensure depth/width changes actually affect the model."""
    bp_dict = {
        "name": "shape_test",
        "input_shape": [1, 28, 28], # MNIST style
        "num_classes": 5,
        "backbone": "convnet",
        "stages": [
            {"type": "conv_block", "filters": 8, "depth": 1}
        ]
    }
    
    bp = Blueprint.from_dict(bp_dict)
    model = render_blueprint(bp)
    
    dummy = torch.randn(4, 1, 28, 28) # Batch 4
    out = model(dummy)
    
    assert out.shape == (4, 5) # (Batch, Num_Classes)